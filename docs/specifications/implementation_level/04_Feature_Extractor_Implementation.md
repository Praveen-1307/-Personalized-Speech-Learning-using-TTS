# Implementation Specification: Feature Extractor (`feature_extractor.py`)


## 1. Overview
The Feature Extractor calculates the "Physical Voice Profile." It acts as the mechanical ears of the system to figure out how high/low, loud, and fast someone speaks.

## 2. The Problem It Solves
A computer does not naturally know what "loud" or "deep" means; it only sees a massive array of 80,000 floating-point numbers.

## 3. How It Works in the Code 
1. **Pitch Tracking:** It runs advanced "Signal Processing" algorithms that scan those 80,000 numbers looking for repeating ripples. If the ripples happen fast, it mathematically logs that the person has a high pitch (Fundamental Frequency). 
2. **Syllable Counting:** To measure if someone is talking fast, it literally counts the number of times the volume (RMS Energy) spikes up and down per second. By doing this, it can guess exactly how many syllables the person spoke in that 5-second clip.
3. **Statistical Aggregation:** Finally, it takes all these thousands of fluctuating numbers and uses Python math functions (`np.mean()`) to average them out into single numbers (e.g., Average Pitch: 120Hz).
