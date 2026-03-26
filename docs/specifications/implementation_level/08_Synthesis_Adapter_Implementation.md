# Implementation Specification: Synthesis Adapter (`synthesis_adapter.py`)

## 1. Overview
The Synthesis Adapter serves as an execution wrapper for older, foundational speech models (like Piper) providing them basic integration endpoints. 

## 2. The Problem It Solves
The core Piper TTS model lacks intrinsic voice cloning capabilities out of the box; it only generates high-quality standard voices. The system must find a mechanism to inject the user's specific Voice Profile data (like pitch and speed) into the final audio product. 

## 3. How It Works in the Code
It acts as a two-stage audio pipeline.
1. **The Piper Manager:** It automatically handles dependencies, bridging Piper binary instances with python scripts.
2. **The Speed Translator:** Since Piper commands utilize a conceptual `length_scale`, the Adapter performs reverse-math based on the user's desired "Words Per Minute" to calculate the exact float integer Piper needs to match the tempo. 
3. **The DSP Auto-Tuner:** It uses `librosa` to forcefully pitch the entire track up or down (`librosa.effects.pitch_shift()`) capping the shift so the audio doesn't glitch.

## 4. Technical Architecture

### 4.1 Inputs
- **Base Text:** Target sentences.
- **Voice Configurations:** Dictating the specific base voice tag to utilize.
- **Pitch Adjustments:** Integer step requirements for post-processing.

### 4.2 Execution Flow
1. Instantiate standard local Subprocess commands invoking Piper's `.exe` framework.
2. Pass arguments utilizing stdin stream mapping.
3. Collect resulting byte chunks, reconstructing into valid `np.array` waveforms.
4. Process output wav arrays over the `librosa` pitch shifters.

### 4.3 Data Transformations
- **Command Line Sub-processing:** Transfers execution data outside the python memory space completely, fetching resulting STDOUT audio pipes back in.
- **DSP Pitch Shifting:** Resamples internal frequency windows via Fast-Fourier processes scaling base frequencies up logarithmically.

### 4.4 Outputs
- **Resulting Array:** Standalone 1D TTS Generation Array + Sample Rate.

### 4.5 Current Limitations
- **Process Blocking:** Synchronous subprocessing freezes the global python thread executing Piper. 
- **Destructive Shifting:** Extreme values pushed into `librosa.pitch_shift` cause intense robotic artifacts and clipping.
