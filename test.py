import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor
import json
import os
import logging
import gc
from config import config
from dataset import SEP28kDataset, get_collate_fn
from model import SOTAHybridDysfluencyModel
from utils import compute_advanced_metrics, brutal_memory_cleanup, setup_logger

def run_testing():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    setup_logger(config.LOG_DIR)
    logging.info("[START] Benchmarking SOTA architecture on pristine hold-out Test partition")
    
    if config.VERBOSE and torch.cuda.is_available():
        print(f"[VERBOSE] Initial GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated, {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")
    
    run_log_dir = config.get_run_dir()
    eval_checkpoint_file = os.path.join(run_log_dir, "Testing_Checkpoint.json")
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    pad_token_id = processor.tokenizer.pad_token_id
    test_data = SEP28kDataset("test", processor, augment=False)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=get_collate_fn(processor), pin_memory=config.PIN_MEMORY, num_workers=config.NUM_WORKERS)
    model = SOTAHybridDysfluencyModel(config, processor.tokenizer.vocab_size, pad_token_id).to(config.DEVICE)
    model.load_state_dict(torch.load(os.path.join(config.CHECKPOINT_DIR, "best_sota_model_ema.pt"), map_location=config.DEVICE))
    model.eval()
    all_preds, all_labels, all_ctc_preds, all_text_labels = [], [], [], []

    with torch.no_grad():
        for test_idx, batch in enumerate(test_loader):
            progress = (test_idx / len(test_loader)) * 100 if len(test_loader) > 0 else 0
            if config.VERBOSE and test_idx % max(1, len(test_loader) // 10) == 0:
                logging.info(f"[PROGRESS] Testing batch {test_idx}/{len(test_loader)} [{progress:.1f}%]")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            inputs = batch["input_values"].to(config.DEVICE, non_blocking=True)
            masks = batch["attention_mask"].to(config.DEVICE, non_blocking=True)
            stutter_labels = batch["stutter_labels"].to(config.DEVICE, non_blocking=True)
            labels = batch["labels"].to(config.DEVICE, non_blocking=True)

            with torch.autocast('cuda', enabled=config.USE_AMP):
                ctc_logits, _, stutter_logits, _ = model(inputs, masks)
            
            all_preds.append(stutter_logits.cpu())
            all_labels.append(stutter_labels.cpu())
            all_ctc_preds.append(torch.argmax(ctc_logits, dim=-1).cpu())
            
            labels[labels == -100] = pad_token_id
            all_text_labels.append(labels.cpu())
            del inputs, masks, labels, stutter_labels, ctc_logits, stutter_logits, batch
            if test_idx % max(10, len(test_loader) // 5) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    if len(all_preds) == 0:
        logging.error("[CRITICAL] No test batches were processed. Cannot compute metrics.")
        print("[ERROR] No test data processed!")
        return
    
    if config.VERBOSE:
        print(f"\n[VERBOSE] Processing {len(all_preds)} test batches for final metrics...")
        
    metrics = compute_advanced_metrics(torch.cat(all_preds), torch.cat(all_labels), torch.cat(all_ctc_preds), torch.cat(all_text_labels), processor)
    
    report = {
        "Project_Architecture": config.PROJECT_NAME,
        "SOTA_Evaluation_Summary": {
            "Word_Error_Rate_WER": float(metrics["wer"]),
            "Stutter_Classification_F1_Macro": float(metrics["f1_macro"]),
            "Stutter_ROC_AUC": float(metrics["roc_auc"]),
            "Stutter_Precision": float(metrics["precision"]),
            "Stutter_Recall": float(metrics["recall"])
        }
    }

    results_file = os.path.join(config.RESULTS_DIR, "final_sota_evaluation_report.json")
    with open(results_file, "w") as f: json.dump(report, f, indent=4)  
    with open(eval_checkpoint_file, "w") as f: json.dump(report, f, indent=4)
    logging.info(f"[SUCCESS] Test run complete. Logs saved to: {results_file}")
    print(json.dumps(report, indent=4))
    
    del all_preds, all_labels, all_ctc_preds, all_text_labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    brutal_memory_cleanup()

if __name__ == "__main__":
    run_testing()