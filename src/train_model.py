"""
train_model.py - Fire Detection Model Training with Fixes
- Dual-band input (Band 7 SWIR + Band 10 Thermal)
- Proper normalization (min-max scaling)
- TinyFireNet architecture (optimized for 32x32 thermal patches)
- Debug inspection to catch data issues early
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import os
from pathlib import Path

# Get script directory and construct paths relative to it
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, 'data', 'landsat_raw')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

def load_data(data_dir=None, test_split=0.2):
    """
    Load synthetic Landsat data from npy files.
    LOADS BOTH BANDS: Band 7 (SWIR) and Band 10 (Thermal)
    Applies proper min-max normalization.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    labels_path = os.path.join(data_dir, 'labels.csv')
    print(f"Loading labels from: {labels_path}")
    
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    labels_df = pd.read_csv(labels_path)
    X = []
    y = []
    
    print(f"Loading {len(labels_df)} samples from {data_dir}...")
    print("  Loading Band 7 (SWIR) and Band 10 (Thermal)...")
    
    for idx, row in labels_df.iterrows():
        sample_id = row['sample_id']
        label = row['fire_label']
        
        swir_path = os.path.join(data_dir, 'band7_swir', f'{sample_id}.npy')
        thermal_path = os.path.join(data_dir, 'band10_thermal', f'{sample_id}.npy')
        
        if os.path.exists(swir_path) and os.path.exists(thermal_path):
            swir = np.load(swir_path).astype(np.float32)
            thermal = np.load(thermal_path).astype(np.float32)
            dual_band = np.stack([swir, thermal], axis=-1)
            X.append(dual_band)
            y.append(label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"Loaded {len(X)} samples with shape {X.shape}")
    
    # ===== CRITICAL FIX: PROPER MIN-MAX NORMALIZATION =====
    band_dims = X.shape[-1]
    print(f"\n--- NORMALIZATION STATS (BEFORE) ---")
    for b in range(band_dims):
        band_name = "Band 7 (SWIR)" if b == 0 else "Band 10 (Thermal)"
        print(f"{band_name}: Min={X[:,:,:,b].min():.1f}, Max={X[:,:,:,b].max():.1f}, Mean={X[:,:,:,b].mean():.1f}")
    
    X_normalized = np.zeros_like(X)
    for b in range(band_dims):
        band_min = X[:, :, :, b].min()
        band_max = X[:, :, :, b].max()
        band_range = band_max - band_min + 1e-10
        X_normalized[:, :, :, b] = (X[:, :, :, b] - band_min) / band_range
    
    X = X_normalized
    
    print(f"\n--- NORMALIZATION STATS (AFTER) ---")
    for b in range(band_dims):
        band_name = "Band 7 (SWIR)" if b == 0 else "Band 10 (Thermal)"
        print(f"{band_name}: Min={X[:,:,:,b].min():.4f}, Max={X[:,:,:,b].max():.4f}, Mean={X[:,:,:,b].mean():.4f}")
    
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nTrain: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"Label distribution - Train: {np.bincount(y_train.astype(int))}")
    print(f"Label distribution - Test:  {np.bincount(y_test.astype(int))}")
    
    return X_train, y_train, X_test, y_test, X

def create_tiny_fire_net(input_shape=(32, 32, 2)):
    """TinyFireNet: Lightweight CNN for 32x32 dual-band thermal patches."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

print("="*70)
print("FIRE DETECTION MODEL TRAINING - WITH FIXES")
print("="*70)
print()

X_train, y_train, X_test, y_test, X_full = load_data()

print("\n" + "="*70)
print("--- DATA SANITY CHECK ---")
print("="*70)
print(f"Max Value in Dataset (Band 7): {X_full[:,:,:,0].max():.4f}")
print(f"Max Value in Dataset (Band 10): {X_full[:,:,:,1].max():.4f}")

model = create_tiny_fire_net(input_shape=(32, 32, 2))

print("\n" + "="*70)
print("TRAINING MODEL ON GPU...")
print("="*70)
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}\n")

history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test),
                    epochs=20, 
                    batch_size=32)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n{'='*70}")
print(f"FINAL TEST ACCURACY: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"{'='*70}\n")

# Save the full model
os.makedirs(MODELS_DIR, exist_ok=True)
model_path = os.path.join(MODELS_DIR, 'thermal_model.h5')
model.save(model_path)
print(f"Saved full model to {model_path}")

print("\n" + "="*70)
print("QUANTIZING MODEL FOR EMBEDDED DEPLOYMENT...")
print("="*70)

def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(100):
        yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

tflite_path = os.path.join(MODELS_DIR, 'fire_model_quant.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"\nQuantization complete!")
print(f"  Model size: {len(tflite_model) / 1024:.2f} KB")
print(f"  Saved to: {tflite_path}")
print(f"\n{'='*70}")
print("READY FOR SATELLITE INTEGRATION")
print(f"{'='*70}\n")
