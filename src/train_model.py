# train_model.py for User
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
    """Load synthetic Landsat data from npy files"""
    if data_dir is None:
        data_dir = DATA_DIR
    
    # Load labels
    labels_path = os.path.join(data_dir, 'labels.csv')
    print(f"Loading labels from: {labels_path}")
    
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    labels_df = pd.read_csv(labels_path)
    
    X = []
    y = []
    
    print(f"Loading {len(labels_df)} samples from {data_dir}...")
    
    for idx, row in labels_df.iterrows():
        sample_id = row['sample_id']
        label = row['fire_label']
        
        # Load thermal band (Band 10)
        thermal_path = os.path.join(data_dir, 'band10_thermal', f'{sample_id}.npy')
        
        if os.path.exists(thermal_path):
            thermal = np.load(thermal_path)
            # Normalize from uint16 (0-65535) to float32 (0.0-1.0)
            thermal_norm = thermal.astype(np.float32) / 65535.0
            # Add channel dimension
            thermal_norm = np.expand_dims(thermal_norm, axis=-1)
            
            X.append(thermal_norm)
            y.append(label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"Loaded {len(X)} samples with shape {X.shape}")
    
    # Split into train/test
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    return X_train, y_train, X_test, y_test

def create_mobilenet_model(input_shape=(32, 32, 1)):
    # Input shape: 32x32, 1 Channel (Thermal only)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=None # Training from scratch on thermal data
    )
    
    base_model.trainable = True
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid') # Binary Classification (Fire/No Fire)
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Load real synthetic data
X_train, y_train, X_test, y_test = load_data()

# Train
model = create_mobilenet_model(input_shape=(32, 32, 1))
print("Training Model on GPU...")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test),
                    epochs=10, 
                    batch_size=32)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Save the heavy model
os.makedirs(MODELS_DIR, exist_ok=True)
model_path = os.path.join(MODELS_DIR, 'thermal_model.h5')
model.save(model_path)
print(f"Saved full model to {model_path}")

# QUANTIZATION (The Critical Step)
print("\nQuantizing model for embedded deployment...")

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

print(f"Success: Quantized Model Saved as '{tflite_path}'")
print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
