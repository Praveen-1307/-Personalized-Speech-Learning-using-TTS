# Implementation Specification: Synthesis Validator (`validator.py`)
*(Simple Explanation Version)*

## 1. Overview
The Synthesis Validator is an automatic "Quality Assurance (QA) Checker." It is the safety net of the application.

## 2. The Problem It Solves
AI models hallucinate. Sometimes the AI glitches and refuses to speak, or it randomly cuts off half the sentence, but the program thinks it worked because it successfully saved an MP3 file. How do we *prove* it actually generated what we asked?

## 3. How It Works in the Code (The Tricks)
It acts as an automatic spell-checker for audio.
1. **The Math:** If the user types a long 200-word paragraph, the Validator calculates an estimate. It knows humans speak around 2.5 words per second. Therefore, it calculates that the final MP3 file *should* be around 80 seconds long.
2. **Audio Trimming:** Once the AI finishes generating the audio file, the Validator opens it up and trims off all the dead silence so it can measure exactly how many seconds the AI spent actually talking.
3. **The Pass/Fail Check:** It measures the length of the actual audio against the estimated length. If the file is only 5 seconds long (when it should be 80), it instantly flags the generation as a **FAIL** because it knows the AI crashed or skipped the words. It then assigns a percentage "Word Match Score" to show the user how confident it is.
