# 🧠 LoRA-Based RoBERTa Fine-Tuning for AGNEWS Classification

**Developers**:  
- Ashutosh Agrawal  
- Siddhanth Rana  
- Aneesh Mokashi  

**Course**: Deep Learning – Spring 2025  
**Professor**: Chinmay Hegde  
**Project**: Mini Project 2 — Low-Rank Adaptation under 1M Parameters  


---

## 🚀 Project Overview

This project investigates **parameter-efficient fine-tuning** using **Low-Rank Adaptation (LoRA)** applied to the **RoBERTa-base** transformer model for text classification on the AGNEWS dataset. 

The key challenge was to modify the model such that **trainable parameters remain under 1 million**, while maximizing classification accuracy. LoRA provides a mechanism to adapt large pre-trained models by injecting low-rank matrices into attention submodules—achieving competitive results at a fraction of the training cost.

---

## 📊 Final Results Summary

| Metric                   | Value                             |
|--------------------------|------------------------------------|
| **Validation Accuracy**  | **94.22%**                         |
| **Trainable Parameters** | **814,852** (~0.65% of full model) |
| **Total Parameters**     | 125,463,560                        |
| **LoRA Rank (r)**        | 6                                  |
| **Scaling Factor (α)**   | 16                                 |
| **Dropout**              | 0.10                               |
| **Epochs**               | 5                                  |
| **Batch Size**           | 16                                 |
| **Optimizer**            | AdamW                              |
| **Hardware**             | NVIDIA RTX 4070 (Kaggle GPU)       |

---

## 🧪 Methodology

- **Model Architecture**: RoBERTa-base with LoRA adapters added to the attention layers (query and value projections only).
- **Filtering**: Only headlines with 25–90 words and vocabulary richness ≥ 0.70 were used. The filtered dataset contained ~110K examples.
- **Tokenization**: Performed using Hugging Face’s `RobertaTokenizer`, padded to max length of 256.
- **LoRA Integration**: Implemented using the `peft` library, targeting `query` and `value` projections in all attention heads.
- **Training Strategy**:
  - Optimizer: AdamW
  - Learning Rate: 5e-5
  - Epochs: 5
  - Evaluation Strategy: Per epoch
  - Loss: Categorical cross-entropy
- **Metrics**:
  - Classification accuracy
  - Class-wise precision, recall, and F1-score
  - Confusion matrix

---

## 📈 Key Observations

- **Loss Convergence**:
  - Training loss decreased from 0.52 to 0.16 over 5 epochs.
  - Validation loss decreased from 0.25 to 0.205, with no sign of overfitting.
- **Class-wise Performance**:
  - F1-scores > 0.90 across all categories.
  - Sports class had the highest precision and recall.
  - Minor confusion occurred between World and Business categories.
- **Prediction Distribution**:

| Class     | Count |
|-----------|-------|
| World     | 1,568 |
| Sports    | 2,002 |
| Business  | 1,793 |
| Sci/Tech  | 2,637 |
