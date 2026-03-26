# Implementation Specification: Profile Manager (`profile_manager.py`)

## 1. Overview
The Profile Manager is the "Database and Save System." It ensures the system doesn't forget the user's voice when the program closes.

## 2. The Problem It Solves
All the math done by the Feature Extractor and Emotion Detector takes computational time. We absolutely do not want to re-calculate the pitch and emotion every time the user generates a new sentence in the same session.

## 3. How It Works in the Code 
1. **Aggregation:** It gathers all the mathematical numbers from the Extractors, organizes them into a neat list (a Python Dictionary), and pairs it with the user's unique ID.
2. **The Type-Casting Trick:** Python's standard `json.dumps()` save function will physically crash and throw an error if you try to save an advanced NumPy number. It contains a custom loop that secretly converts every single NumPy float into a native python float before saving.
3. **File I/O:** It outputs a standard `.json` text file into the `profiles/` folder on the computer's hard drive.

## 4. Technical Architecture

### 4.1 Inputs
- **User Identifier:** Simple string tracking the profile name.
- **Metadata Dictionary:** Consolidated audio statistics and emotional context maps.
- **Operation Context:** File-path directory logic for I/O operations.

### 4.2 Execution Flow
1. Catch the aggregated dictionary coming from the Pipeline.
2. Run custom `_convert_numpy_types` function over every dictionary key to sanitize datatypes.
3. Generate metadata string payloads (timestamp, id, structure).
4. Run standard File I/O writing streams targeting `profiles/<ID>.json`.
5. (Optional) Zip/Backup past profile revisions to prevent accidental data overwrites.

### 4.3 Data Transformations
- **Dictionary to JSON Byte Stream:** Transcribes nested dynamic python memory structures into structured flat ASCII/UTF-8 JSON files.
- **Type Casting:** Forces `numpy.float32` -> `python.float` avoiding the "Object of type float32 is not JSON serializable" systemic crash.

### 4.4 Outputs
- **JSON File:** Physical `.json` file resting on disk natively.
- **Load Dict:** Emits cleanly loaded dictionary objects when executing `load_profile()`.

### 4.5 Current Limitations
- **Storage Bottleneck:** Synchronous JSON writing locks the main thread, potentially causing a minor UI stutter during bulk batch profile generations.
- **Concurrency Risks:** If two processes attempt to save/update the exact same profile ID simultaneously, race conditions may corrupt the JSON string.
