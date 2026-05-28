import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor, get_cosine_schedule_with_warmup
import logging
import os
import gc
from config import config
from dataset import SEP28kDataset, get_collate_fn
from model import SOTAHybridDysfluencyModel
from loss import SOTAMultiTaskLoss
from utils import ModelEMA, compute_advanced_metrics, brutal_memory_cleanup, setup_logger

def log_gpu_memory(stage: str, epoch: int = 0):
    """Log GPU memory usage for leak detection"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        free = torch.cuda.mem_get_info()[0] / 1e9
        if config.VERBOSE:
            print(f"[MEMORY] {stage} (Epoch {epoch}): Allocated={allocated:.2f}GB | Reserved={reserved:.2f}GB | Free={free:.2f}GB")
        logging.debug(f"[MEMORY] {stage} (Epoch {epoch}): Allocated={allocated:.2f}GB | Reserved={reserved:.2f}GB | Free={free:.2f}GB")

def run_training():
    config.setup_dirs()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    setup_logger(config.LOG_DIR)
    logging.info("[START] Initiating Global Maximum Training Routine...")
    required_files = ["train.csv", "val.csv"]
    for req_file in required_files:
        fpath = os.path.join(config.DATA_DIR, req_file)
        if not os.path.exists(fpath):
            logging.error(f"[CRITICAL] Required data file missing: {fpath}. Please run data_prep.py first.")
            raise FileNotFoundError(f"Missing required file: {fpath}")
    
    torch.set_float32_matmul_precision('high')
    
    logging.info("[VERBOSE] Loading pre-trained speech processor...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    vocab_size = processor.tokenizer.vocab_size
    pad_token_id = processor.tokenizer.pad_token_id
    logging.info(f"[VERBOSE] Processor loaded. Vocab size: {vocab_size}")
    logging.info("[VERBOSE] Initializing datasets...")
    train_data = SEP28kDataset("train", processor, augment=True)
    val_data = SEP28kDataset("val", processor, augment=False)
    logging.info(f"[VERBOSE] Datasets initialized. Train: {len(train_data)}, Val: {len(val_data)}")
    
    collate = get_collate_fn(processor)
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate, pin_memory=config.PIN_MEMORY, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate, pin_memory=config.PIN_MEMORY, num_workers=config.NUM_WORKERS)
    model = SOTAHybridDysfluencyModel(config, vocab_size, pad_token_id).to(config.DEVICE)
    
    if config.USE_FLASH_ATTENTION:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        
    ema = ModelEMA(model, config.EMA_DECAY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, fused=True)
    total_steps = len(train_loader) * config.EPOCHS // config.GRADIENT_ACCUMULATION_STEPS
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * config.WARMUP_RATIO), num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)
    criterion = SOTAMultiTaskLoss(config, blank_idx=pad_token_id)
    best_val_f1 = 0.0
    run_log_dir = config.get_run_dir()

    for epoch in range(1, config.EPOCHS + 1):
        if config.VERBOSE:
            print(f"\n{'='*80}")
            print(f"[VERBOSE] EPOCH {epoch}/{config.EPOCHS} | Batch Size: {config.BATCH_SIZE} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"{'='*80}")
        log_gpu_memory("Epoch Start", epoch)
        logging.info(f"Epoch {epoch}/{config.EPOCHS} started")
        model.train()
        train_loss = 0
        batch_count = 0
        optimizer.zero_grad(set_to_none=True) 
        
        for batch_idx, batch in enumerate(train_loader):
            batch_count += 1
            if config.VERBOSE and batch_idx % max(1, len(train_loader) // 10) == 0:
                progress = (batch_idx / len(train_loader)) * 100
                logging.debug(f"[VERBOSE] Training batch {batch_idx}/{len(train_loader)} [{progress:.1f}%]")
            try:
                inputs = batch["input_values"].to(config.DEVICE, non_blocking=True)
                masks = batch["attention_mask"].to(config.DEVICE, non_blocking=True)
                labels = batch["labels"].to(config.DEVICE, non_blocking=True)
                stutter_labels = batch["stutter_labels"].to(config.DEVICE, non_blocking=True)
                label_lengths = batch["label_lengths"].to(config.DEVICE, non_blocking=True)
                
                audio_lengths = torch.clamp(batch["audio_lengths"] // config.ENCODER_SUBSAMPLING_FACTOR, min=1).to(config.DEVICE)

                with torch.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, enabled=config.USE_AMP):
                    ctc_logits, asr_logits, stutter_logits, attn_weights = model(inputs, masks, labels)
                    loss = criterion(ctc_logits, asr_logits, labels, stutter_logits, stutter_labels, attn_weights, label_lengths, audio_lengths)
                    loss = loss / config.GRADIENT_ACCUMULATION_STEPS
                scaler.scale(loss).backward()
                if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(model)
                train_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
                del inputs, masks, labels, stutter_labels, label_lengths, audio_lengths, ctc_logits, asr_logits, stutter_logits, loss, attn_weights, batch
                if batch_idx % max(10, len(train_loader) // 5) == 0:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                logging.error(f"[ERROR] Training batch {batch_idx} failed: {e}")
                raise
        ema.shadow.eval()
        all_stutter_preds, all_stutter_labels, all_ctc_preds, all_text_labels = [], [], [], []
        
        with torch.no_grad():
            for val_idx, batch in enumerate(val_loader):
                if config.VERBOSE and val_idx % max(1, len(val_loader) // 10) == 0:
                    logging.debug(f"[VERBOSE] Validation batch {val_idx}/{len(val_loader)}")   
                inputs = batch["input_values"].to(config.DEVICE, non_blocking=True)
                masks = batch["attention_mask"].to(config.DEVICE, non_blocking=True)
                labels = batch["labels"].to(config.DEVICE, non_blocking=True)
                with torch.autocast('cuda', enabled=config.USE_AMP):
                    ctc_logits, _, stutter_logits, _ = ema.shadow(inputs, masks)
                
                all_stutter_preds.append(stutter_logits.cpu())
                all_stutter_labels.append(batch["stutter_labels"].cpu())
                all_ctc_preds.append(torch.argmax(ctc_logits, dim=-1).cpu())
                labels[labels == -100] = pad_token_id
                all_text_labels.append(labels.cpu())
                del inputs, masks, labels, ctc_logits, stutter_logits, batch
                if val_idx % max(5, len(val_loader) // 5) == 0:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    gc.collect()

        log_gpu_memory("Validation Complete", epoch)
        if len(all_stutter_preds) == 0:
            logging.warning(f"[WARNING] No validation batches processed in epoch {epoch}")
            metrics = {"f1_macro": 0.0, "roc_auc": 0.0, "wer": 1.0, "precision": 0.0, "recall": 0.0}
        else:
            if config.VERBOSE:
                print(f"[VERBOSE] Computing metrics for {len(all_stutter_preds)} validation batches...")
            metrics = compute_advanced_metrics(torch.cat(all_stutter_preds), torch.cat(all_stutter_labels), torch.cat(all_ctc_preds), torch.cat(all_text_labels), processor)
        if epoch % config.CHECKPOINT_INTERVAL_EPOCHS == 0 or epoch == config.EPOCHS:
            checkpoint_path = os.path.join(run_log_dir, f"Training_Epoch_{epoch:03d}_Checkpoint.pt")
            torch.save(ema.shadow.state_dict(), checkpoint_path)
            
        logging.info(f"Epoch {epoch:03d} | Train Loss: {train_loss/len(train_loader):.4f} | Val F1: {metrics['f1_macro']:.4f} | Val AUC: {metrics['roc_auc']:.4f} | Val WER: {metrics['wer']:.4f}")

        if metrics['f1_macro'] > best_val_f1:
            best_val_f1 = metrics['f1_macro']
            torch.save(ema.shadow.state_dict(), os.path.join(config.CHECKPOINT_DIR, "best_sota_model_ema.pt"))
            logging.info(">>> SOTA Checkpoint Secured.") 
        del all_stutter_preds, all_stutter_labels, all_ctc_preds, all_text_labels, metrics
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        log_gpu_memory("Epoch End", epoch)
        brutal_memory_cleanup()

if __name__ == "__main__":
    run_training()