# HIE Grading Using Signal Processing and Deep Learning

This repository contains the code and resources for the **HIE Grading Using Signal Processing and Deep Learning** paper. The project implements signal processing techniques combined with deep learning models for the automated grading of Hypoxic-Ischemic Encephalopathy (HIE).

---

## Repository Structure

The code and data are organized into the following directories and files:

### **1. Data**
- Contains a **sample of the FM/AM transformed data**, stored as `.wav` files (one per channel).
- Example: `Data/ ├── channel1.wav ├── channel2.wav └──`


### **2. Combined_data.csv**
- A CSV file that stores:
- Grading information
- Metadata corresponding to the `.wav` files.

### **3. Spectrograms**
- A sample of the **spectrograms** generated from the FM/AM transformed data.
- Stored as `.png` files (one per channel).
- Spectrograms are dynamically generated using the data loader in `Main_Inference.py`.

### **4. Model_Weights**
- Contains the **trained model weights** used in the inference pipeline.
- The model was evaluated using a **nested cross-validation (CV) framework**.

### **5. Main_Inference.py**
- The main script for performing inference. 
- Key functionalities:
1. Loading the data.
2. Generating spectrograms.
3. Loading the trained model.
4. Performing inference.

### **6. Optimised_thresholds.json**
- A JSON file storing optimized **rounding thresholds** for post-processing.
- These thresholds are based on the validation predictions from model training.

---
