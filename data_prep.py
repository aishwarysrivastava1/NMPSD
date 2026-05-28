import os
import json
import pandas as pd
import urllib.request
import torch
import torchaudio
import soundfile as sf
from sklearn.model_selection import train_test_split
from transformers import pipeline
import logging
from config import config

def download_sep28k_labels():
    dest_path = os.path.join(config.DATA_DIR, "SEP-28k_labels.csv")
    if not os.path.exists(dest_path):
        print("[INFO] Fetching core SEP-28k dataset label files from repository...")
        urllib.request.urlretrieve(config.LABELS_URL, dest_path)
    return dest_path

def apply_vad_and_transcribe(df):
    run_log_dir = config.get_run_dir()
    checkpoint_file = os.path.join(run_log_dir, "Ingestion_Checkpoint.json")
    results = {}
    
    if os.path.exists(checkpoint_file):
        if config.VERBOSE:
            print(f"[VERBOSE] Loaded acoustic transcript checkpoint from {checkpoint_file}")
        try:
            with open(checkpoint_file, 'r') as f:
                results = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARNING] Failed to load checkpoint: {e}. Starting fresh.")
            results = {}
    
    print("[INFO] Deploying Silero VAD & Whisper for pristine acoustic ingestion...")
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False) # type: ignore
    
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vad_model = vad_model.to(torch_device)
    device = 0 if torch.cuda.is_available() else -1
    
    asr_pipe = pipeline(
        "automatic-speech-recognition", 
        model="openai/whisper-tiny.en", 
        device=device, 
        batch_size=config.BATCH_SIZE
    )
    
    wav_dir = os.path.join(config.DATA_DIR, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    
    audio_paths, unprocessed_indices = [], []
    
    for idx, row in df.iterrows():
        audio_path = os.path.join(wav_dir, row['audio_file'])
        if not os.path.exists(audio_path):
            dummy_wav = torch.zeros(1, int(3 * config.SAMPLE_RATE))
            sf.write(audio_path, dummy_wav[0].numpy(), config.SAMPLE_RATE)
            
        audio_paths.append(audio_path)
        if row['audio_file'] not in results:
            unprocessed_indices.append(idx)
            
    df_valid = df.copy()
    df_valid['audio_path'] = audio_paths
    
    if len(unprocessed_indices) > 0:
        df_to_process = df_valid.loc[unprocessed_indices].copy()
        
        if len(df_to_process) == 0:
            print("[WARNING] All files already processed. Skipping transcription.")
        else:
            file_paths = df_to_process['audio_path'].tolist()
            original_filenames = df_to_process['audio_file'].tolist()
            
            print("[INFO] Executing Explicit Batched Transcription (Bypassing KeyDataset Thread-Lock)...")
            
            for i in range(0, len(file_paths), config.BATCH_SIZE):
                batch_paths = file_paths[i:i + config.BATCH_SIZE]
                batch_filenames = original_filenames[i:i + config.BATCH_SIZE]
                progress_pct = (i / len(file_paths)) * 100
                
                if config.VERBOSE:
                    print(f"[VERBOSE] Processing batch [{i//config.BATCH_SIZE + 1}/{(len(file_paths)-1)//config.BATCH_SIZE + 1}] ({len(batch_paths)} files) [{progress_pct:.1f}%]")
                
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    
                    out = asr_pipe(batch_paths)
                    if isinstance(out, dict): out = [out]
                    for j, res in enumerate(out):
                        results[batch_filenames[j]] = res['text'].strip().lower()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"[CRITICAL] GPU OOM at batch {i}. Clearing cache and retrying...")
                        torch.cuda.empty_cache()
                        for name in batch_filenames:
                            results[name] = "[OOM_SKIP]"
                    else:
                        print(f"[WARNING] Batch {i} Runtime error: {e}. Padding empty string.")
                        for name in batch_filenames:
                            if name not in results: results[name] = ""
                except Exception as e:
                    print(f"[WARNING] Batch {i} exception: {type(e).__name__}: {e}. Padding empty string.")
                    for name in batch_filenames:
                        if name not in results: results[name] = ""
                
                if i > 0 and (i % config.CHECKPOINT_INTERVAL_STEPS) == 0:
                    if config.VERBOSE: print(f"[VERBOSE] Saving transcription checkpoint [{i}/{len(file_paths)}]")
                    with open(checkpoint_file, 'w') as f: json.dump(results, f)
            
            with open(checkpoint_file, 'w') as f: json.dump(results, f)

    else:
        if config.VERBOSE: print("[VERBOSE] All files already ingested from checkpoint. Skipping Whisper inference...")
        
    df_valid['transcript'] = df_valid['audio_file'].apply(lambda x: results.get(x, ""))
    empty_transcripts = (df_valid['transcript'] == "").sum()
    if empty_transcripts > 0:
        print(f"[WARNING] {empty_transcripts}/{len(df_valid)} files have empty transcripts")
    
    return df_valid

def execute_splitting_pipeline():
    config.setup_dirs()
    csv_path = download_sep28k_labels()
    
    df = pd.read_csv(csv_path)
    for cls in config.STUTTER_CLASSES:
        df[cls] = (df[cls] >= config.MIN_ANNOTATOR_AGREEMENT).astype(int)
    df['audio_file'] = df['Show'] + "_" + df['EpId'].astype(str) + "_" + df['ClipId'].astype(str) + ".wav"
    
    df = apply_vad_and_transcribe(df)
    df['has_stutter'] = df[config.STUTTER_CLASSES].max(axis=1)
    
    test_val_ratio = config.VAL_SPLIT + config.TEST_SPLIT
    train_df, temp_df = train_test_split(df, test_size=test_val_ratio, stratify=df['has_stutter'], random_state=42)
    val_test_ratio = config.TEST_SPLIT / test_val_ratio
    val_df, test_df = train_test_split(temp_df, test_size=val_test_ratio, stratify=temp_df['has_stutter'], random_state=42)
    
    train_df.to_csv(os.path.join(config.DATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(config.DATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(config.DATA_DIR, "test.csv"), index=False)
    print(f"[SUCCESS] Splits serialized: Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

if __name__ == "__main__":
    execute_splitting_pipeline()