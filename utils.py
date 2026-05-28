import torch
import gc
import copy
import logging
import os
from jiwer import wer, cer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

class ModelEMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model.module if hasattr(model, 'module') else model)
        self.shadow.eval()
        for param in self.shadow.parameters():
            param.requires_grad = False

    def update(self, model):
        model_params = (model.module if hasattr(model, 'module') else model).state_dict()
        shadow_params = self.shadow.state_dict()
        with torch.no_grad():
            for name, param in model_params.items():
                if param.dtype.is_floating_point:
                    shadow_params[name].sub_((1.0 - self.decay) * (shadow_params[name] - param))
        self.shadow.load_state_dict(shadow_params)

def brutal_memory_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def compute_advanced_metrics(stutter_preds, stutter_labels, ctc_preds, text_labels, processor):
    probs = torch.sigmoid(stutter_preds).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    labels = stutter_labels.cpu().numpy()
    pred_str = processor.batch_decode(ctc_preds)
    label_str = processor.batch_decode(text_labels, group_tokens=False)
    valid_pairs = [(p, l) for p, l in zip(pred_str, label_str) if len(l.strip()) > 0]
    p_texts = [p[0] if len(p[0].strip()) > 0 else "<sil>" for p in valid_pairs]
    l_texts = [p[1] if len(p[1].strip()) > 0 else "<sil>" for p in valid_pairs]
    
    try: auc = roc_auc_score(labels, probs, average='macro')
    except ValueError: auc = 0.0   
    if not p_texts or not l_texts:
        return {
            "f1_macro": f1_score(labels, preds, average='macro', zero_division=0),
            "precision": precision_score(labels, preds, average='macro', zero_division=0),
            "recall": recall_score(labels, preds, average='macro', zero_division=0),
            "roc_auc": auc, "wer": 1.0, "cer": 1.0
        }
    
    return {
        "f1_macro": f1_score(labels, preds, average='macro', zero_division=0),
        "precision": precision_score(labels, preds, average='macro', zero_division=0),
        "recall": recall_score(labels, preds, average='macro', zero_division=0),
        "roc_auc": auc,
        "wer": wer(l_texts, p_texts),
        "cer": cer(l_texts, p_texts)
    }

def setup_logger(log_dir):
    from config import config
    level = logging.DEBUG if config.VERBOSE else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s' if config.VERBOSE else '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        filename=os.path.join(log_dir, 'pipeline.log'), filemode='a',
        format=log_format, level=level
    )
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger('').handlers.clear()
    logging.getLogger('').addHandler(console)
    if config.VERBOSE:
        print(f"[VERBOSE] Logger initialized | Level: DEBUG | Batch Size: {config.BATCH_SIZE} | Checkpoint Intervals: {config.CHECKPOINT_INTERVAL_STEPS} steps, {config.CHECKPOINT_INTERVAL_EPOCHS} epochs")