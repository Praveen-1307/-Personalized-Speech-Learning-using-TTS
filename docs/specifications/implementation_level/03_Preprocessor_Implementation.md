# Implementation Specification: Preprocessor (`preprocessor.py`)

## 1. Overview
The Audio Preprocessor is the "Cleaner." It is the first line of defense in the system, ensuring that the AI only receives high-quality audio data.

## 2. The Problem It Solves
Microphones record at totally different speeds, volumes, and qualities. Muffled, quiet, or static-filled audio will instantly ruin the AI's ability to clone a voice properly.

## 3. How It Works in the Code 
Before the AI or the Analyzers hear anything, this script uses mathematical filters to fix the audio:
1. **Resampling:** It forces all audio to play at the exact same uniform speed so the AI knows what to expect.
2. **Normalization:** It artificially raises or lowers the volume of the file so it's perfectly balanced.
3. **Trimming:** It automatically detects "dead air" and crops off the silence at the start and end of the recording.
4. **Pre-emphasis (Noise Reduction):** It turns down the low bass frequencies to remove background noise automatically.

## 4. Technical Architecture

### 4.1 Inputs
- **Raw Input Audio:** Unprocessed NumPy array representing imported WAV data.
- **Target Configurations:** Base sample rate, min/max duration, normalization db limits.

### 4.2 Execution Flow
1. Load audio matrix in floating point format.
2. Standardize sample rate using `librosa.resample`.
3. Detect absolute volume peak; amplify or attenuate array strictly to -3.0 dB.
4. Run `librosa.effects.trim` dynamically based on top_db amplitude thresholds.
5. Optionally apply signal pre-emphasis using a standard signal filtering formula.

### 4.3 Data Transformations
- **Continuous Re-mapping:** Interpolates sample coordinates to stretch or shrink the data array vertically (volume) or horizontally (resampling).
- **Array Slicing:** Dynamically splices boundaries off the 1D arrays removing silent indices.

### 4.4 Outputs
- **Processed Audio:** Clean, 1D NumPy float32 array ready for the feature extractor.
- **New Sample Rate:** Confirmed sampling rate integer.

### 4.5 Current Limitations
- **Destructive Operation:** If recording is overly saturated/clipped natively, the preprocessor normalizer cannot reconstruct destroyed frequencies.
- **Aggressive Trimming:** The top_db threshold on trimming may inadvertently cut off extremely soft but intentional whispering words at the end of loops.
