# Hybrid Intrusion Detection System (IDS)

A Deep Learning based Hybrid Intrusion Detection System for detecting network attacks using the CICIDS-2017 dataset.

This project combines:

- CNN for local feature extraction
- BiLSTM for sequential learning
- Attention Mechanism for focusing on important network features
- Autoencoder for anomaly detection
- Hybrid decision logic for improved attack detection

---

## Architecture
## Architecture

```
                   Network Traffic
                          │
                          ▼
                  Data Preprocessing
          (Cleaning + Feature Selection +
             Standardization)
                          │
        ┌─────────────────┴──────────────────┐
        │                                    │
        ▼                                    ▼
 Autoencoder                        CNN → BiLSTM
(Anomaly Detector)                 → Attention
        │                             │
        └──────────────┬──────────────┘
                       ▼
              Hybrid Decision Logic
                       ▼
              Benign / Attack
```

---

## Dataset

Dataset used:

- CICIDS-2017

The dataset contains both normal and malicious network traffic.

---

## Key Components

- Data Cleaning
- Binary Label Encoding
- Stratified Train/Test Split
- Mutual Information Feature Selection
- StandardScaler Normalization
- Autoencoder trained only on Benign traffic
- CNN + BiLSTM + Attention classifier
- Hybrid Prediction Logic
- ROC Curve
- Confusion Matrix
- Reconstruction Error Distribution

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

---

## Workflow

1. Load CICIDS-2017 Dataset
2. Data Cleaning
3. Binary Label Encoding
4. Train-Test Split
5. Feature Selection
6. Feature Scaling
7. Train Autoencoder
8. Train CNN-BiLSTM-Attention Classifier
9. Hybrid Prediction
10. Evaluate Model

---

## Hybrid Decision Logic

The system combines two models:

### Autoencoder

- Learns only normal traffic.
- Detects anomalies using reconstruction error.

### CNN + BiLSTM + Attention

- Performs binary classification.
- Predicts Benign or Attack.

### Final Decision

- Classifier prediction is the primary decision.
- Autoencoder only overrides predictions for highly anomalous traffic with low classifier confidence.

---

## Evaluation Metrics

The model reports:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

---

## Generated Files

After training, the following files are generated:

```
classifier.h5
autoencoder.h5
selector.pkl
scaler.pkl
training_curves.png
roc_curve.png
confusion_matrix.png
recon_error_dist.png
```

---

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python3 main.py
```

---

## Author

Danush S

B.Tech Computer Science and Engineering

