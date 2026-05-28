import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class RelaxedGaussianAlignmentLoss(nn.Module):
    def __init__(self, sigma: float):
        super().__init__()
        self.sigma = sigma

    def forward(self, attention_weights, target_lengths, acoustic_lengths):
        attn = attention_weights.mean(dim=1) 
        batch_size, max_ty, max_tx = attn.size()
        device = attn.device
        
        total_loss = torch.tensor(0.0, device=device)
        valid_batches = 0
        
        for b in range(batch_size):
            ty, tx = target_lengths[b].item(), acoustic_lengths[b].item()
            if ty == 0 or tx == 0: continue
                
            j_idx = torch.arange(ty, device=device).float()
            i_idx = torch.arange(tx, device=device).float()
            
            phi_j = j_idx * (tx / ty)
            dist_matrix = (i_idx.unsqueeze(0) - phi_j.unsqueeze(1)) ** 2
            
            penalty_matrix = 1.0 - torch.exp(-dist_matrix / (2 * self.sigma ** 2))
            alignment_penalty = torch.sum(attn[b, :ty, :tx] * penalty_matrix)
            
            total_loss += alignment_penalty / (ty * tx)
            valid_batches += 1
            
        return total_loss / max(valid_batches, 1)

class SOTAMultiTaskLoss(nn.Module):
    def __init__(self, cfg, blank_idx):
        super().__init__()
        self.cfg = cfg
        self.ctc_loss_fn = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=cfg.LABEL_SMOOTHING)
        self.focal_loss_fn = AsymmetricFocalLoss(alpha=cfg.FOCAL_ALPHA, gamma=cfg.FOCAL_GAMMA)
        self.align_loss_fn = RelaxedGaussianAlignmentLoss(sigma=cfg.RELAXATION_SIGMA)
        
    def forward(self, ctc_logits, asr_logits, labels, stutter_logits, stutter_labels, attn_weights, label_lengths, audio_lengths):
        log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
        ctc_targets = [l[l != -100] for l in labels]
        ctc_target_lens = torch.tensor([len(t) for t in ctc_targets], dtype=torch.long, device=labels.device)
        ctc_targets_flat = torch.cat(ctc_targets) if len(ctc_targets) > 0 else torch.empty(0, dtype=torch.long, device=labels.device)
        
        l_ctc = self.ctc_loss_fn(log_probs, ctc_targets_flat, audio_lengths, ctc_target_lens)
        l_ce = self.ce_loss_fn(asr_logits.reshape(-1, asr_logits.size(-1)), labels.reshape(-1))
        l_class = self.focal_loss_fn(stutter_logits, stutter_labels)
        l_align = self.align_loss_fn(attn_weights, label_lengths, audio_lengths)
        
        total_loss = (self.cfg.LOSS_WEIGHT_ASR_CTC * l_ctc) + \
                     (self.cfg.LOSS_WEIGHT_ASR_CE * l_ce) + \
                     (self.cfg.LOSS_WEIGHT_CLASS * l_class) + \
                     (self.cfg.LOSS_WEIGHT_ALIGN * l_align)
                     
        return total_loss