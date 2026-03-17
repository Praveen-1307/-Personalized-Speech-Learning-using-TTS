# Implementation Specification: Profile Manager (`profile_manager.py`)


## 1. Overview
The Profile Manager is the "Database and Save System." It ensures the system doesn't forget the user's voice when the program closes.

## 2. The Problem It Solves
All the math done by the Feature Extractor and Emotion Detector takes computational time. We absolutely do not want to re-calculate the pitch and emotion every time the user generates a new sentence in the same session.

## 3. How It Works in the Code 
1. **Aggregation:** It gathers all the mathematical numbers from the Extractors, organizes them into a neat list (a Python Dictionary), and pairs it with the user's unique ID.
2. **The Type-Casting Trick:** Python's standard `json.dumps()` save function will physically crash and throw an error if you try to save an advanced NumPy math number to a text file. The Profile Manager code contains a custom loop that acts as a translator—it secretly converts every single NumPy float into standard text before saving it to the disk.
3. **File I/O:** It outputs a standard `.json` text file into the `profiles/` folder on the computer's hard drive.
