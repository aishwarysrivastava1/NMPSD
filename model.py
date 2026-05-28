import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel
from peft import LoraConfig, get_peft_model

class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )
    def forward(self, x, mask=None):
        attn_weights = self.attention(x).squeeze(-1) 
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask.bool(), float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        return torch.sum(x * attn_weights.unsqueeze(-1), dim=1) 

class SOTAHybridDysfluencyModel(nn.Module):
    def __init__(self, cfg, vocab_size, pad_token_id):
        super().__init__()
        self.config = cfg
        self.pad_token_id = pad_token_id
        
        base_encoder = WavLMModel.from_pretrained(cfg.ENCODER_ID)
        lora_config = LoraConfig(
            r=cfg.LORA_R, lora_alpha=cfg.LORA_ALPHA,
            target_modules=["q_proj", "v_proj"], lora_dropout=cfg.LORA_DROPOUT, bias="none"
        )
        self.encoder = get_peft_model(base_encoder, lora_config)
        
        if cfg.GRADIENT_CHECKPOINTING:
            self.encoder.gradient_checkpointing_enable() 
            
        self.ctc_head = nn.Linear(cfg.HIDDEN_DIM, vocab_size)
        self.asp = AttentiveStatisticsPooling(cfg.HIDDEN_DIM)
        
        self.stutter_head = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM // 2),
            nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.HIDDEN_DIM // 2, len(cfg.STUTTER_CLASSES))
        )
        
        self.text_embedding = nn.Embedding(vocab_size, cfg.HIDDEN_DIM)
        self.pos_encoder = nn.Parameter(torch.randn(1, cfg.MAX_TEXT_LENGTH, cfg.HIDDEN_DIM) * 0.02)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.HIDDEN_DIM, nhead=cfg.NUM_HEADS,
            dim_feedforward=cfg.HIDDEN_DIM * 4, dropout=cfg.DROPOUT, batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.DECODER_LAYERS)
        self.asr_head = nn.Linear(cfg.HIDDEN_DIM, vocab_size)

    def forward(self, input_values, audio_padding_mask, text_targets=None):
        enc_out = self.encoder(input_values=input_values, attention_mask=audio_padding_mask)
        memory = enc_out.last_hidden_state 
        
        ctc_logits = self.ctc_head(memory)
        seq_len = memory.size(1)
        sub_mask = F.interpolate(
            audio_padding_mask.unsqueeze(1).float(), 
            size=seq_len, 
            mode='nearest'
        ).squeeze(1).bool()
            
        pooled_memory = self.asp(memory, sub_mask)
        stutter_logits = self.stutter_head(pooled_memory)
        
        asr_logits, attn_weights = None, None
        if text_targets is not None:
            safe_targets = text_targets.clone()
            safe_targets[safe_targets == -100] = self.pad_token_id
            tgt_seq_len = min(safe_targets.size(1), self.config.MAX_TEXT_LENGTH)
            safe_targets = safe_targets[:, :tgt_seq_len]
            
            tgt_embed = self.text_embedding(safe_targets) + self.pos_encoder[:, :tgt_seq_len, :]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(self.config.DEVICE)
            
            decoder_out = self.decoder(tgt=tgt_embed, memory=memory, tgt_mask=tgt_mask)
            attn_logits = torch.bmm(decoder_out, memory.transpose(1, 2)) / (self.config.HIDDEN_DIM ** 0.5)
            attn_weights = torch.softmax(attn_logits, dim=-1).unsqueeze(1)
            asr_logits = self.asr_head(decoder_out)
            
        return ctc_logits, asr_logits, stutter_logits, attn_weights