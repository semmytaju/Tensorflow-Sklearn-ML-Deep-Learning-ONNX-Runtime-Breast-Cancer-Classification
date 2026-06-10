# Breast Cancer Classification using ML, DL and ONNX Runtime

This project compares Machine Learning (SVM, KNN, Random Forest) and Deep Learning (Neural Network) models for binary classification using the Wisconsin Breast Cancer dataset. All models are exported to ONNX format and evaluated using ONNX Runtime. ONNX (Open Neural Network Exchange) is an open-source standard created to represent machine learning and deep learning models.

---

## ⚙️ Data Set

| Attribute | Value |
|-----------|-------|
| **Name** | Wisconsin Breast Cancer Dataset |
| **Source** | UCI Machine Learning Repository |
| **Samples** | 569 |
| **Features** | 30 numeric features |
| **Classes** | 2 (Malignant, Benign) |

---

## ⚙️ Models Used

### Machine Learning Models
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest (RF)

### Deep Learning Model
- Fully Connected Neural Network (DNN)

---

## 🚀 Workflow Pipeline

Data → Preprocessing → Training → ONNX Conversion → ONNX Runtime → Evaluation

---

## 📥 Installation

pip install numpy pandas matplotlib seaborn scikit-learn
pip install tensorflow
pip install onnx onnxruntime
pip install skl2onnx tf2onnx

---

## 🔧 Data Preprocessing

- StandardScaler normalization
- Train/test split 80/20
- Stratified sampling

---

## 🧠 Deep Learning Architecture

Input (30 features)
→ Dense (64, ReLU)
→ Dense (32, ReLU)
→ Dense (1, Sigmoid)
→ Output (Binary Classification)

---

## 🔄 ONNX Conversion

### Machine Learning Models
Converted using skl2onnx:

from skl2onnx import convert_sklearn

---

### Deep Learning Model
Converted using tf2onnx:

import tf2onnx

---

## ⚡ ONNX Runtime Inference

import onnxruntime as ort

session = ort.InferenceSession("model.onnx")

---

## 📊 Evaluation Metrics

Accuracy:
TP + TN / (TP + TN + FP + FN)

Sensitivity:
TP / (TP + FN)

Specificity:
TN / (TN + FP)

MCC:
(TP×TN − FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

---

## 📈 Results Summary

| Model          | ACC   | SEN   | SPEC  | MCC   |
|----------------|-------|-------|-------|-------|
| SVM            | 0.982 | 0.986 | 0.976 | 0.961 |
| KNN            | 0.965 | 0.972 | 0.952 | 0.926 |
| Random Forest  | 0.974 | 0.986 | 0.952 | 0.945 |
| DNN            | 0.986 | 0.986 | 0.976 | 0.969 |

---

## 📉 Visualizations

- Confusion Matrix
- Accuracy Comparison
- Sensitivity Comparison
- Specificity Comparison
- MCC Comparison

---

## 🧪 Requirements

- Python 3.8+
- TensorFlow
- Scikit-learn
- ONNX
- ONNX Runtime

---

## 📚 References

- https://onnx.ai/
- https://scikit-learn.org/
- https://www.tensorflow.org/
- UCI Machine Learning Repository

---

## 📌 License

For educational and research purposes only.
