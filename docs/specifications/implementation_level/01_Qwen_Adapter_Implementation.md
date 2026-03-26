# Implementation Specification: Qwen Adapter (`personalization_engine/qwen_adapter.py`)

## Overview
This component serves as the core text-to-speech generation engine. It processes textual input alongside a reference speaker embedding to generate personalized, high-fidelity audio streams utilizing batched transformer inference.

## Input Data and Structure
| Input | Type | Structure |
| :--- | :--- | :--- |
| `text` | `str` | The target text string to be synthesized. |\n| `reference_audio` | `np.ndarray` | Waveform of the speaker's voice used for zero-shot cloning. |\n| `speaker_embedding` | `np.ndarray` | Pre-computed x-vector representing the speaker profile. |\n| `batch_size` | `int` | Number of text chunks to process concurrently. |

## Sequence of Steps
1. Initialize the Qwen3-TTS model weights into VRAM.\n2. Compute or load the mathematical speaker embedding from the reference audio.\n3. Chunk the target text string based on semantic punctuation boundaries.\n4. Execute batched inference on all text chunks using the same speaker embedding.\n5. Concatenate the output waveform tensors into a continuous audio stream.

## Data Transformations
- `str -> list[str]`: Target paragraph is split into sentence chunks.\n- `np.ndarray -> np.ndarray`: Audio waveform is compressed into a fixed-length speaker embedding vector.\n- `str + np.ndarray -> np.ndarray`: Text chunks and speaker embedding generate raw audio tensors.

## Output Data and Structure
| Output | Type | Structure |
| :--- | :--- | :--- |
| `waveform` | `np.ndarray` | 1D float array of the synthesized audio. |\n| `sample_rate` | `int` | The sampling rate of the generated audio (e.g., 22050). |

## Notes
- Heavy VRAM utilization; batching strategy is critical to avoid out-of-memory errors.\n- Requires proper punctuation in the text for successful chunking.\n- Temperature is strictly set to lock the voice variance for consistency.
