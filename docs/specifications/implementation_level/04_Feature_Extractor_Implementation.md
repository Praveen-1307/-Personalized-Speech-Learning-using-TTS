# Implementation Specification: Feature Extractor (`feature_extractor.py`)

## 1. Overview
The Feature Extractor calculates the "Physical Voice Profile." It acts as the mechanical ears of the system to figure out how high/low, loud, and fast someone speaks.

## 2. The Problem It Solves
A computer does not naturally know what "loud" or "deep" means; it only sees a massive array of 80,000 floating-point numbers.

## 3. How It Works in the Code 
1. **Pitch Tracking:** It runs advanced "Signal Processing" algorithms that scan those 80,000 numbers looking for repeating ripples. If the ripples happen fast, it mathematically logs that the person has a high pitch.
2. **Syllable Counting:** To measure if someone is talking fast, it counts the number of times the volume (RMS Energy) spikes up and down per second. By doing this, it can guess the speaking rate limit.
3. **Statistical Aggregation:** Finally, it takes all these thousands of fluctuating numbers and uses Python math functions (`np.mean()`) to average them out into single numbers.

## 4. Technical Architecture

### 4.1 Inputs
- **Cleaned Audio:** 1D NumPy array validated by the Preprocessor.
- **Sampling Rate:** Integer tracking audio time resolution.

### 4.2 Execution Flow
1. Ingest clean audio array.
2. Utilize `librosa.pyin` or `librosa.yin` to calculate Fundamental Frequency point-maps natively.
3. Utilize `librosa.feature.rms` to measure energy across frame boundaries.
4. Utilize `librosa.feature.spectral_centroid` to measure audio brightness.
5. Apply Numpy means and medians to condense array data into singular float variables.

### 4.3 Data Transformations
- **Time-Domain to Frequency-Domain:** Performs Short-Time Fourier Transforms (STFT) converting raw amplitude frames into interpretable frequency distributions.
- **Matrix Condensation:** Flattens 2D STFT representations down into 1-dimensional dictionary metadata files.

### 4.4 Outputs
- **Statistical Dictionary:** A mapped pure-python dictionary containing flat statistical points like `f0_mean`, `energy_std`, `speaking_rate`, and `spectral_centroid`.

### 4.5 Current Limitations
- **Processing Time:** `pyin` F0 extraction is highly accurate but computationally heavy, causing delays on longer files.
- **Accuracy on Noisy Files:** Fails significantly if polyphonic background noises seep into the recording (e.g. music playing in background).
