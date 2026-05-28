# 🚀 SEP-28k SOTA: Hybrid WavLM + Attentive Statistics Pooling for Dysfluency Detection

## 1. Problem Statement & Motivation
Stuttering and dysfluency detection is a highly complex paralinguistic task. Unlike standard Automatic Speech Recognition (ASR), which maps phonemes to text, dysfluency detection requires the network to interpret the *manner* of speech delivery (prolongations, blocks, interjections). Standard acoustic models (like Wav2Vec2) often fail here because they are trained to *ignore* paralinguistics in favor of linguistic content. 

This project solves this by introducing a **Global Maximum Architecture**: combining `microsoft/wavlm-large` (explicitly SOTA for paralinguistics) with an **Attentive Statistics Pooling (ASP)** mechanism and a **Multi-Task Objective** to jointly learn dysfluency classification and ASR.

## 2. Dataset: SEP-28k
The system leverages the [Stuttering Events in Podcasts (SEP-28k)](https://github.com/apple/ml-stuttering-events-dataset) dataset, comprising ~28,000 clips across 5 stuttering sub-classes. Our pipeline dynamically handles ingestion, pseudo-transcript generation via Whisper, and inherent Stratified 80/10/10 splitting.

## 3. SOTA Architectural Innovations
* **WavLM Large Encoder:** Outperforms HuBERT and Wav2Vec2 due to its gated relative position bias and masked speech denoising pre-training.
* **Attentive Statistics Pooling (ASP):** Rather than mean-pooling acoustic representations across the time dimension, ASP learns a dynamic attention matrix. It intelligently weights frames containing stuttering events, discarding fluent background noise.
* **Flash Attention 2:** Deployed within the transformer decoder, reducing GPU VRAM allocation complexity from $O(N^2)$ to $O(N)$, ensuring stable gradient descent under tight constraints.
* **Silero Voice Activity Detection (VAD):** Mathematically purges absolute silence from acoustic vectors prior to ingestion, stabilizing CTC alignments.

## 4. Multi-Task Objective Formulation
The network is optimized using a weighted hybrid objective function containing four terms:

$$L_{total} = \lambda_1 L_{CTC} + \lambda_2 L_{CE} + \lambda_3 L_{Focal} + \lambda_4 L_{Align}$$

Where:
* $L_{CTC}$: Connectionist Temporal Classification loss for unsupervised acoustic-to-text alignment.
* $L_{CE}$: Label-smoothed Cross Entropy for autoregressive text decoding.
* $L_{Align}$: A relaxed Gaussian alignment loss enforcing monotonicity between acoustic space and semantic space.
* $L_{Focal}$: **Asymmetric Focal Loss** ($\gamma=2.0$). Because dysfluencies represent $<10\%$ of all frames, standard BCE collapses. Focal loss forces the optimizer to explicitly target hard-to-detect stutter blocks:

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

## 5. Brutal VRAM Optimization Logic
To completely eliminate CUDA Out-Of-Memory exceptions, this codebase utilizes:
1.  **Orchestrator OS-Level Isolation (`main.py`)**: Runs each script as an independent child process, guaranteeing total OS-level VRAM reclamation upon completion.
2.  **ZeRO-Inspired Graph Purging**: Calls `optimizer.zero_grad(set_to_none=True)` to deallocate memory references rather than leaving empty pointers.
3.  **BF16/FP16 Mixed Precision Native Scaling**.
4.  **Deep GC Sweeping**: Aggressive inner-loop garbage collection routines mapped via `utils.brutal_memory_cleanup()`.

## 6. Execution Command
```bash
# This will autonomously execute Ingestion, 100-Epoch Training, and Diagnostics:
python main.py