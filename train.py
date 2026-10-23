

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# 1. CONFIG
# ----------------------------
DATASET_DIR = "Dataset"                  # root folder containing High/Low/Moderate + csv
CSV_PATH = os.path.join(DATASET_DIR, "pollution_labels.csv")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 15          # training only the new classifier head
EPOCHS_FINETUNE = 15      # fine-tuning the base model
NUM_CLASSES = 3
CLASS_NAMES = ["High", "Low", "Moderate"]   # alphabetical, matches flow_from_dataframe sort
SEED = 42

# ----------------------------
# 2. LOAD & PREPARE DATA
# ----------------------------
df = pd.read_csv(CSV_PATH)

# Build full relative path: Dataset/<pollution_level>/<image_name>
df["filepath"] = df.apply(
    lambda row: os.path.join(DATASET_DIR, row["pollution_level"], row["image_name"]),
    axis=1
)

# Keep only rows whose image actually exists on disk (avoids crashes on missing files)
df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)
print(f"Total usable images: {len(df)}")
print(df["pollution_level"].value_counts())

# Use the CSV's own split column if it distinguishes train/test; otherwise make our own
if df["split"].nunique() > 1:
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] != "train"].reset_index(drop=True)
else:
    train_df, test_df = train_test_split(
        df, test_size=0.15, stratify=df["pollution_level"], random_state=SEED
    )

# Split a validation set out of the training data
train_df, val_df = train_test_split(
    train_df, test_size=0.15, stratify=train_df["pollution_level"], random_state=SEED
)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ----------------------------
# 3. DATA GENERATORS (with augmentation on train only)
# ----------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
)

val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_dataframe(
    train_df, x_col="filepath", y_col="pollution_level",
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", classes=CLASS_NAMES, shuffle=True, seed=SEED
)

val_gen = val_test_datagen.flow_from_dataframe(
    val_df, x_col="filepath", y_col="pollution_level",
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", classes=CLASS_NAMES, shuffle=False
)

test_gen = val_test_datagen.flow_from_dataframe(
    test_df, x_col="filepath", y_col="pollution_level",
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", classes=CLASS_NAMES, shuffle=False
)

# ----------------------------
# 4. BUILD MODEL (transfer learning)
# ----------------------------
base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False  # freeze for stage 1

inputs = Input(shape=(*IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
]

# ----------------------------
# 5. STAGE 1: TRAIN CLASSIFIER HEAD
# ----------------------------
print("\n=== Stage 1: Training classifier head ===")
history_head = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks,
)

# ----------------------------
# 6. STAGE 2: FINE-TUNE BASE MODEL
# ----------------------------
print("\n=== Stage 2: Fine-tuning base model ===")
base_model.trainable = True
# Freeze early layers, fine-tune only the later ones
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # much lower LR for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINETUNE,
    callbacks=callbacks,
)

# ----------------------------
# 7. EVALUATE ON TEST SET
# ----------------------------
print("\n=== Evaluating on test set ===")
test_loss, test_acc = model.evaluate(test_gen)
print(f"Test Accuracy: {test_acc*100:.2f}%")

y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# ----------------------------
# 8. SAVE FINAL MODEL
# ----------------------------
model.save("pollution_classifier_final.keras")
print("\nModel saved as pollution_classifier_final.keras")