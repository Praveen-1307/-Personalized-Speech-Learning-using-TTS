# Component Specification: Synthesis Validator (`validator.py`)

## 1. Overview
The Synthesis Validator serves as the Quality Assurance (QA) gatekeeper for the entire application. After the Qwen Adapter generates the final `.wav` file, the Validator runs a series of checks comparing the original input text to the generated audio's length and structure to ensure the artificial intelligence successfully generated all the words without randomly breaking or hallucinating.

## 2. Input Data & Structure
| Input Name | Data Type | Description |
| :--- | :--- | :--- |
| `text` | `String` | The original target text string that the user typed. |
| `audio_path` | `String` | The absolute path to the generated `.wav` output file. |

## 3. High-Level Sequence of Execution
1.  **Text Validation**: First, checks the basic validity of the text itself. Does it contain actual words, or is it just punctuation and empty spaces?
2.  **Audio Loading**: Reads the generated `.wav` file into memory using `librosa`.
3.  **Silence Filtering**: Identifies and mathematically trims away any ambient silence before the sentence starts and after it ends to find the precise "Active Speaking Duration".
4.  **Audio/Text Ratio Calculation**: Determines the "Estimated Word Match Score". It compares the character count and word count of the text to the absolute millisecond duration of the "Active Speech" to ensure they correlate. 
5.  **Status Check**: Applies a Pass/Fail threshold. If the audio is far too short for a 100-word essay, it flags the audio as invalid.

## 4. Internal Data Transformations
*   **Text $\rightarrow$ Array**: The text string is split by spaces into a List of strings (`words`).
*   **Files $\rightarrow$ Arrays**: The `.wav` file gets converted to an array of amplitudes to detect where voices start/end using Decibel thresholds (db=30).
*   **Audio Length $\rightarrow$ Statistics**: The raw samples generated are converted into seconds (`duration = samples / sample_rate`), and math is used to calculate "Syllables Per Second".

## 5. Output Data & Structure
| Output Name | Data Type | Description |
| :--- | :--- | :--- |
| `validation_report` | `Dictionary` | A report detailing if the generation worked, and by how much. |

Example structure:
```json
{
  "valid": true,
  "metrics": {
    "duration_seconds": 6.2,
    "active_duration": 5.8,
    "estimated_word_count": 14,
    "match_score": 0.98
  }
}
```
