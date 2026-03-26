# Implementation Specification: Synthesis Validator (`validator.py`)

## 1. Overview
The Synthesis Validator is an automatic "Quality Assurance (QA) Checker." It is the safety net of the application.

## 2. The Problem It Solves
AI models hallucinate. Sometimes the AI glitches and refuses to speak, or it randomly cuts off half the sentence, but the program thinks it worked because it successfully saved an MP3 file. How do we *prove* it actually generated what we asked?

## 3. How It Works in the Code
It acts as an automatic spell-checker for audio.
1. **The Math:** If the user types a long 200-word paragraph, the Validator calculates an estimate. It knows humans speak around 2.5 words per second. Therefore, it calculates that the final file *should* be around 80 seconds long.
2. **Audio Trimming:** Once the AI finishes generating the audio, the Validator opens it up and trims off all the dead silence so it can measure exactly how many seconds the AI spent actually talking.
3. **The Pass/Fail Check:** It measures the length of the actual audio against the estimated length. If the file is only 5 seconds long (when it should be 80), it instantly flags the generation as a **FAIL** because it knows the AI crashed or skipped the words.

## 4. Technical Architecture

### 4.1 Inputs
- **Synthesized Audio:** Output 1D NumPy waveform.
- **Original Text:** The target string we attempted to generate.
- **Sampling Rate:** Configuration audio int.

### 4.2 Execution Flow
1. Ingest finalized audio matrix.
2. Strip silent bookends using `librosa.effects.trim`.
3. Calculate physical duration of non-silent sequence in floating seconds.
4. Process original text, splitting by whitespace to acquire raw word_count integer.
5. Compute mathematical Expected Duration based on baseline 150 Words-Per-Minute metric.
6. Compare expected vs physical to render binary Boolean validation logic.

### 4.3 Data Transformations
- **Word Tracking:** Sub-divides simple strings into word token arrays via `.split()`.
- **Duration Mathematics:** Divides sequence length against Sample Rate to infer real-time physical lengths.

### 4.4 Outputs
- **Validation Report:** Dictionary emitting properties: `is_valid` (Boolean true/false), `score` (0-100 match accuracy), `estimated_duration` and `actual_duration`.

### 4.5 Current Limitations
- **Speech Rate Ignorance:** Hardcoded to assume human median speech rate. If the user intentionally sets the AI to speak very slowly, the validator will falsely assume the file is "too long" and fail it.
- **Phonetic Ignorance:** It does not actually listen to the phonemes with an active recognizer; it strictly relies on dimensional math duration.
