# Component Specification: Profile Manager (`profile_manager.py`)

## 1. Overview
The Profile Manager acts as the Database/Data Storage layer for the entire application. When a user records their voice, the Analysis Engine creates a complex dictionary of features and emotions. The Profile Manager's job is to serialize (save) that dictionary to the hard drive so the user's voice profile can be loaded again tomorrow without needing to be recalculated.

## 2. Input Data & Structure
| Input Name | Data Type | Description |
| :--- | :--- | :--- |
| `user_id` | `String` | A unique name or ID for the user (e.g., `'john_doe'`). |
| `features` | `Dictionary` | The physical voice data from `feature_extractor.py`. |
| `emotion` | `Dictionary` | The psychological voice data from `emotion_detector.py`. |

## 3. High-Level Sequence of Execution
1.  **Creation/Aggregation**: Combines the user's ID, physical features, and emotional state into one massive "Profile Dictionary".
2.  **Serialization**: Converts that Python dictionary into a JSON string format that can be written to a text file.
3.  **File I/O**: Opens a file inside the `profiles/` folder (e.g., `profiles/john_doe.json`) and writes the JSON string to disk.
4.  **Loading (Deserialization)**: If called to *load*, it reads a JSON file from disk, converts the string back into a Python dictionary, and returns it to the main application.

## 4. Internal Data Transformations
*   **Dictionary $\rightarrow$ JSON String**: Complex Python objects (like NumPy float values) are cast to standard Python floats and transformed into a `.json` formatted string via `json.dumps()`.
*   **String $\rightarrow$ File**: The raw string is transformed into bytes and written directly to the hard drive file system.

## 5. Output Data & Structure
| Output Name | Data Type | Description |
| :--- | :--- | :--- |
| `(File on Disk)` | `.json` File | The physical file holding all the combined voice metadata. |
| `loaded_profile` | `Dictionary` | If reading, it outputs a Python dictionary containing the parsed JSON data. |
