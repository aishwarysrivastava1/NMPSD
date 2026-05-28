import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import os
from config import config

class AdvancedAcousticAugment(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mask = T.TimeMasking(time_mask_param=80)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=27)

    def forward(self, x):
        if torch.rand(1).item() > 0.5: x = self.time_mask(x)
        if torch.rand(1).item() > 0.5: x = self.freq_mask(x)
        return x

class SEP28kDataset(Dataset):
    def __init__(self, split: str, processor, augment: bool = False):
        self.data = pd.read_csv(os.path.join(config.DATA_DIR, f"{split}.csv"))
        self.processor = processor
        self.augment = augment
        self.aug_pipe = AdvancedAcousticAugment()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data) or idx < 0:
            raise IndexError(f"Dataset index {idx} out of range [0, {len(self.data)})")
            
        row = self.data.iloc[idx]
        audio_path = os.path.join(config.DATA_DIR, "wavs", row['audio_file'])
        
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"[WARNING] Failed to load {audio_path}: {e}. Using silence.")
            waveform = torch.zeros(1, config.MAX_AUDIO_LENGTH)
            sr = config.SAMPLE_RATE
            
        if sr != config.SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, config.SAMPLE_RATE)
            
        waveform = waveform.squeeze(0)[:config.MAX_AUDIO_LENGTH]
        
        if self.augment:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
            waveform = self.aug_pipe(waveform.unsqueeze(0)).squeeze(0)
            
        labels = self.processor.tokenizer(row['transcript']).input_ids
        stutter_labels = torch.tensor(row[config.STUTTER_CLASSES].values.astype(float), dtype=torch.float32)
        
        return {"waveform": waveform, "labels": torch.tensor(labels, dtype=torch.long), "stutter_labels": stutter_labels}

def get_collate_fn(processor):
    def collate_fn(batch):
        waveforms = [item["waveform"] for item in batch]
        labels = [item["labels"] for item in batch]
        stutter_labels = torch.stack([item["stutter_labels"] for item in batch])
        
        audio_lengths = torch.tensor([len(w) for w in waveforms], dtype=torch.long)
        label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

        inputs = processor(waveforms, sampling_rate=config.SAMPLE_RATE, return_tensors="pt", padding=True)
        padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        return {
            "input_values": inputs.input_values,
            "attention_mask": inputs.attention_mask,
            "labels": padded_labels,
            "stutter_labels": stutter_labels,
            "audio_lengths": audio_lengths,
            "label_lengths": label_lengths
        }
    return collate_fn