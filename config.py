import os
import torch
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Infrastructure & Paths
    PROJECT_NAME: str = "WavLM_ASP_Dysfluency_SOTA"
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    RESULTS_DIR: str = os.path.join(BASE_DIR, "results")
    LOG_DIR: str = os.path.join(BASE_DIR, "logs")
    CHECKPOINT_DIR: str = os.path.join(BASE_DIR, "checkpoints")

    # SEP-28k Parameters
    LABELS_URL: str = "https://raw.githubusercontent.com/apple/ml-stuttering-events-dataset/main/SEP-28k_labels.csv"
    STUTTER_CLASSES: List[str] = field(default_factory=lambda: ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection'])
    MIN_ANNOTATOR_AGREEMENT: int = 2
    SAMPLE_RATE: int = 16000

    # Inherent Data Split
    TRAIN_SPLIT: float = 0.80
    VAL_SPLIT: float = 0.10
    TEST_SPLIT: float = 0.10

    # SOTA Architectural Upgrades
    ENCODER_ID: str = "microsoft/wavlm-large"
    HIDDEN_DIM: int = 512  
    NUM_HEADS: int = 4 
    DECODER_LAYERS: int = 2  
    DROPOUT: float = 0.1
    ENCODER_SUBSAMPLING_FACTOR: int = 320 
    
    # Fine-Tuning (LoRA)
    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05

    # Optimization & Scaling Guardrails
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP: bool = True
    USE_FLASH_ATTENTION: bool = True 
    GRADIENT_CHECKPOINTING: bool = True
    PIN_MEMORY: bool = True  
    NUM_WORKERS: int = 0 if os.name == 'nt' else 4  
    GRADIENT_ACCUMULATION_STEPS: int = 4 
    BATCH_SIZE: int = 2  
    MAX_AUDIO_LENGTH: int = 16000 * 3  
    MAX_TEXT_LENGTH: int = 64  

    # Multi-Task Loss Formulation
    RELAXATION_SIGMA: float = 5.0
    FOCAL_GAMMA: float = 2.0     
    FOCAL_ALPHA: float = 0.75    
    LOSS_WEIGHT_ASR_CE: float = 0.4
    LOSS_WEIGHT_ASR_CTC: float = 0.2
    LOSS_WEIGHT_ALIGN: float = 0.1
    LOSS_WEIGHT_CLASS: float = 1.5 
    LABEL_SMOOTHING: float = 0.1

    # Optimization Routine
    LEARNING_RATE: float = 3e-4
    WEIGHT_DECAY: float = 0.01
    WARMUP_RATIO: float = 0.1
    EPOCHS: int = 100  
    MAX_GRAD_NORM: float = 1.0
    EMA_DECAY: float = 0.999

    VERBOSE: bool = True  
    CHECKPOINT_INTERVAL_STEPS: int = 10000 
    CHECKPOINT_INTERVAL_EPOCHS: int = 25  

    def setup_dirs(self):
        for d in [self.DATA_DIR, self.RESULTS_DIR, self.LOG_DIR, self.CHECKPOINT_DIR]:
            os.makedirs(d, exist_ok=True)
            
    def get_run_dir(self):
        run_dir = os.environ.get("RUN_ID", "RUN_1")
        path = os.path.join(self.LOG_DIR, run_dir)
        os.makedirs(path, exist_ok=True)
        return path

config = Config()