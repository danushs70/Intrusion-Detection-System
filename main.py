# ============================================================
# Anomaly-Based IDS — Architecture 1
# Module 1 : CNN → BiLSTM → Attention  (Classifier)
# Module 2 : Autoencoder               (Anomaly Detector)
# Combined : if autoencoder flags anomaly → "ATTACK"
#            else use classifier output
# Dataset  : CICIDS-2017
# ============================================================

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc)
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout,
                                     Conv1D, MaxPooling1D,
                                     Bidirectional, LSTM,
                                     Attention, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
#   LOAD DATASET
# ============================================================
df = pd.read_csv("dataset/cicids.csv")
#df = pd.read_csv("cicids.csv")
df.columns = df.columns.str.strip()
print("Full Dataset Shape:", df.shape)

# Stratified sample — keeps original class ratio
# Using train_test_split trick: sample 300k while preserving Label distribution
from sklearn.model_selection import train_test_split as _tts
sample_frac = min(300_000 / len(df), 1.0)
if sample_frac < 1.0:
    df, _ = _tts(df, train_size=sample_frac, random_state=42,
                 stratify=df["Label"])
    df = df.reset_index(drop=True)
print("Sampled Dataset Shape:", df.shape)
print("Label column present:", "Label" in df.columns)  # confirm

# ============================================================
#   DATA CLEANING
# ============================================================

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print("After Cleaning:", df.shape)

# ============================================================
#   LABEL ENCODING  (benign=0, attack=1)
# ============================================================

print("\nOriginal Label Distribution:")
print(df["Label"].value_counts())

df["Label"] = df["Label"].str.lower()
df["Label"] = df["Label"].apply(lambda x: 0 if x == "benign" else 1)

print("\nAfter Binary Conversion:")
print(df["Label"].value_counts())

# ============================================================
#   FEATURE / TARGET SPLIT
# ============================================================

X = df.drop("Label", axis=1)
y = df["Label"]

# ============================================================
#   STRATIFIED TRAIN-TEST SPLIT  (must come BEFORE fitting
#     any preprocessor — prevents data leakage)
# ============================================================

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain samples:", X_train_raw.shape[0])
print("Test  samples:", X_test_raw.shape[0])

# ============================================================
#   FEATURE SELECTION  (fit on TRAIN only)
# ============================================================

selector = SelectKBest(mutual_info_classif, k=40)
selector.fit(X_train_raw, y_train)           # <-- train only

X_train_sel = selector.transform(X_train_raw)
X_test_sel  = selector.transform(X_test_raw)

print("After Feature Selection — Train:", X_train_sel.shape)

# ============================================================
#   NORMALIZATION  (fit on TRAIN only)
# ============================================================

scaler = StandardScaler()
scaler.fit(X_train_sel)                      # <-- train only

X_train = scaler.transform(X_train_sel)
X_test  = scaler.transform(X_test_sel)

# Save preprocessors for live inference later
joblib.dump(selector, "selector.pkl")
joblib.dump(scaler,   "scaler.pkl")
print("Selector and scaler saved.")

# ============================================================
#   CLASS WEIGHTS
# ============================================================

classes      = np.unique(y_train)
class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))
print("\nClass Weights:", class_weight_dict)

# ============================================================
# MODULE 2 — AUTOENCODER  (trained on BENIGN traffic only)
# ============================================================
# The autoencoder learns to reconstruct "normal" traffic.
# High reconstruction error on a sample → likely an attack.
# It must never see attack samples during training.
# ============================================================

print("\n" + "="*50)
print("MODULE 2 — Training Autoencoder on BENIGN data only")
print("="*50)

X_train_benign = X_train[y_train == 0]   # benign rows only
input_dim      = X_train.shape[1]        # 40

# Encoder: compress  40 → 32 → 16 → 8
ae_input  = Input(shape=(input_dim,))
enc       = Dense(32, activation="relu")(ae_input)
enc       = Dense(16, activation="relu")(enc)
bottleneck = Dense(8, activation="relu")(enc)

# Decoder: reconstruct  8 → 16 → 32 → 40
dec = Dense(16, activation="relu")(bottleneck)
dec = Dense(32, activation="relu")(dec)
ae_output = Dense(input_dim, activation="linear")(dec)

autoencoder = Model(ae_input, ae_output)
autoencoder.compile(optimizer="adam", loss="mse")

ae_early_stop = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

autoencoder.fit(
    X_train_benign, X_train_benign,
    epochs=30,
    batch_size=256,
    validation_split=0.1,
    callbacks=[ae_early_stop],
    verbose=1
)

autoencoder.save("autoencoder.h5")
print("Autoencoder saved.")

# --- Reconstruction error threshold ---
# Using 99th-percentile → only top 1% of benign errors flagged (fewer false positives)
recon_train    = autoencoder.predict(X_train_benign, batch_size=512)
recon_errors_train = np.mean((X_train_benign - recon_train) ** 2, axis=1)
ae_threshold   = np.percentile(recon_errors_train, 99)  # Option 1: 95 → 99
print(f"\nAutoencoder anomaly threshold (99th pct): {ae_threshold:.6f}")

# ============================================================
# MODULE 1 — CNN → BiLSTM → ATTENTION  (Classifier)
# ============================================================
# Takes the 40 scaled features reshaped as a 1-D sequence.
# No autoencoder encoding — direct raw features.
# ============================================================

print("\n" + "="*50)
print("MODULE 1 — Training CNN → BiLSTM → Attention Classifier")
print("="*50)

# Reshape: (samples, 40) → (samples, 40, 1)  for Conv1D
X_train_3d = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_3d  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

inp    = Input(shape=(input_dim, 1))

# CNN block — extract local patterns
c1     = Conv1D(64,  3, activation="relu", padding="same")(inp)
p1     = MaxPooling1D(2)(c1)
c2     = Conv1D(128, 3, activation="relu", padding="same")(p1)
p2     = MaxPooling1D(2)(c2)

# BiLSTM — capture bidirectional sequential patterns
lstm   = Bidirectional(LSTM(64, return_sequences=True))(p2)

# Self-attention
attn   = Attention()([lstm, lstm])
gap    = GlobalAveragePooling1D()(attn)

drop   = Dropout(0.4)(gap)
out    = Dense(1, activation="sigmoid")(drop)

classifier = Model(inputs=inp, outputs=out)
classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

classifier.summary()

clf_early_stop = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
)

history = classifier.fit(
    X_train_3d, y_train,
    epochs=10,
    batch_size=256,          # increased from 128 → fewer steps per epoch
    validation_split=0.2,
    callbacks=[clf_early_stop, reduce_lr],
    class_weight=class_weight_dict,
    verbose=1
)

classifier.save("classifier.h5")
print("Classifier saved.")

# ============================================================
# COMBINED OUTPUT LOGIC  (Option 1 + Option 2)
# ============================================================
# Option 1 : ae_threshold raised to 99th pct (conservative)
# Option 2 : Soft fusion — classifier gets 70% vote, AE gets 30%
#            No hard override, so AE never fully overrules classifier
# ============================================================

print("\n" + "="*50)
print("COMBINED PREDICTION — Soft Fusion (Option 1 + 2)")
print("="*50)

# Step 1: Autoencoder reconstruction errors on test set
recon_test        = autoencoder.predict(X_test, batch_size=512)
recon_errors_test = np.mean((X_test - recon_test) ** 2, axis=1)

# Step 2: Classifier probabilities on test set
clf_probs = classifier.predict(X_test_3d, batch_size=256).ravel()

# Step 3: Hybrid Decision Logic
# Classifier has full control
clf_threshold = 0.40
y_pred_combined = (clf_probs >= clf_threshold).astype(int)

# AE only triggers for extremely abnormal traffic
ae_high_threshold = ae_threshold * 3.0

clf_confidence = np.maximum(clf_probs, 1 - clf_probs)

override = (
    (clf_confidence < 0.60) &
    (recon_errors_test > ae_high_threshold)
)

y_pred_combined[override] = 1

#print(f"\nAE high anomaly scores (>0.5) : {(ae_scores > 0.5).sum()} samples")
print(f"\nAE high anomaly samples : {(recon_errors_test > ae_high_threshold).sum()}")
print(f"Classifier-only flags         : {(clf_probs > 0.5).sum()} samples as attacks")
print(f"Combined final flags          : {y_pred_combined.sum()} samples as attacks")

# ============================================================
# EVALUATION
# ============================================================

print("\n--- Classifier-Only Results ---")
y_pred_clf_only = (clf_probs > 0.5).astype(int)
print(confusion_matrix(y_test, y_pred_clf_only))
print(classification_report(y_test, y_pred_clf_only,
                             target_names=["Benign", "Attack"]))

print("\n--- Combined Model Results ---")
print(confusion_matrix(y_test, y_pred_combined))
print(classification_report(y_test, y_pred_combined,
                             target_names=["Benign", "Attack"]))

# ============================================================
# PLOTS
# ============================================================

# 1. Training Accuracy & Loss
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
axes[0].set_title("Classifier — Accuracy")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"],     label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Val Loss")
axes[1].set_title("Classifier — Loss")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()

# 2. ROC Curve — Classifier
fpr_clf, tpr_clf, _ = roc_curve(y_test, clf_probs)
auc_clf = auc(fpr_clf, tpr_clf)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr_clf,
    tpr_clf,
    label=f"Classifier (AUC={auc_clf:.4f})"
)

plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — CNN + BiLSTM + Attention")
plt.legend()

plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()

print(f"AUC — Classifier: {auc_clf:.4f}")
# 3. Autoencoder Reconstruction Error Distribution
plt.figure(figsize=(8, 4))
plt.hist(recon_errors_test[y_test == 0], bins=80, alpha=0.6,
         label="Benign",  color="steelblue", density=True)
plt.hist(recon_errors_test[y_test == 1], bins=80, alpha=0.6,
         label="Attack",  color="tomato",    density=True)
plt.axvline(ae_threshold, color="black", linestyle="--",
            label=f"Threshold = {ae_threshold:.4f}")
plt.xlabel("Reconstruction Error (MSE)")
plt.ylabel("Density")
plt.title("Autoencoder — Reconstruction Error Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("recon_error_dist.png", dpi=150)
plt.show()

# 4. Confusion Matrix — Combined
cm = confusion_matrix(y_test, y_pred_combined)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks([0, 1], ["Benign", "Attack"])
plt.yticks([0, 1], ["Benign", "Attack"])
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("Confusion Matrix — Combined Model")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# ============================================================
# LIVE INFERENCE HELPER
# ============================================================
# Use this function to classify new, unseen traffic samples.

def predict_live(raw_features: np.ndarray,
                 threshold: float = ae_threshold) -> np.ndarray:
    """
    raw_features : 2-D numpy array, shape (n_samples, original_feature_count)
    Returns      : 1-D array of predictions — 0 = Benign, 1 = Attack
    """
    # Preprocess
    feat    = selector.transform(raw_features)
    feat    = scaler.transform(feat)
    feat_3d = feat.reshape(feat.shape[0], feat.shape[1], 1)

    # Autoencoder check
    recon   = autoencoder.predict(feat, batch_size=256)
    errors  = np.mean((feat - recon) ** 2, axis=1)
    flags   = (errors > threshold).astype(int)

    # Classifier
    probs   = classifier.predict(feat_3d, batch_size=256).ravel()

    # Hybrid decision
    clf_threshold = 0.50
    pred = (probs >= clf_threshold).astype(int)

    ae_high_threshold = threshold * 3.0
    clf_confidence = np.maximum(probs, 1 - probs)

    override = (
        (clf_confidence < 0.60) &
        (errors > ae_high_threshold)
    )

    pred[override] = 1

    return pred
