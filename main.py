import subprocess
import sys
import os
import time

def execute_phase(script_name: str):
    from config import config
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    print(f"\n" + "="*90)
    print(f"[ORCHESTRATOR] Commencing SOTA Operational Phase: {script_name}")
    print("="*90)
    if config.VERBOSE:
        print(f"[VERBOSE] Script path: {script_path}")
        print(f"[VERBOSE] Python executable: {sys.executable}")
        print(f"[VERBOSE] Device: {config.DEVICE}")
        print(f"[VERBOSE] Batch size: {config.BATCH_SIZE}")
        print(f"[VERBOSE] Checkpoint intervals: Steps={config.CHECKPOINT_INTERVAL_STEPS}, Epochs={config.CHECKPOINT_INTERVAL_EPOCHS}")
    
    env = os.environ.copy()
    start_time = time.time()
    process = None
    try:
        process = subprocess.Popen([sys.executable, script_path], stdout=sys.stdout, stderr=sys.stderr, env=env)
        process.wait()  # NO TIMEOUT - let processes complete naturally
        
        elapsed_time = time.time() - start_time
        if process.returncode != 0:
            print(f"\n[ERROR] Execution failed in {script_name} with return code {process.returncode}.")
            sys.exit(process.returncode)
        else:
            print(f"[SUCCESS] {script_name} fully resolved in {elapsed_time:.2f} seconds.")
    except Exception as e:
        print(f"\n[FATAL] Execution error in {script_name}: {e}")
        if process is not None:
            try:
                process.kill()
            except:
                pass
        sys.exit(1)

def main():
    print("[SYSTEM INITIALIZATION] Running SOTA WavLM + ASP Dysfluency Framework...")
    
    from config import config
    config.setup_dirs()
    existing_runs = [d for d in os.listdir(config.LOG_DIR) if d.startswith("RUN_")]
    max_run = 0
    for r in existing_runs:
        try:
            max_run = max(max_run, int(r.split("_")[1]))
        except ValueError:
            pass
    run_id = f"RUN_{max_run + 1}"
    os.environ["RUN_ID"] = run_id
    
    log_path = config.get_run_dir()
    print(f"[VERBOSE] Allocated Global Runtime Environment: {log_path}")
    
    execute_phase("data_prep.py")
    execute_phase("train.py")
    execute_phase("test.py")
    print("\n" + "="*90)
    print("[COMPLETED] All global maximum pipeline operations resolved. Exiting.")
    print("="*90)

if __name__ == "__main__":
    main()