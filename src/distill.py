"""
Model Distillation: Train lightweight linear student from TinyFireNet teacher.
Reduces model size from 285 KB → ~48 bytes (6 parameters).

Process:
1. Load trained TinyFireNet teacher model (quantized TFLite)
2. Load physics features extracted from training data
3. Get teacher predictions (soft labels)
4. Train linear regression student on features + soft labels
5. Save lightweight onboard model
"""

import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path
import sys
import os

def load_teacher_model(model_path='models/fire_model_quant.tflite'):
    """Load quantized TinyFireNet teacher model."""
    print(f"\nLoading teacher model: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  Input shape:  {input_details[0]['shape']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    
    return interpreter, input_details, output_details

def load_dual_band_data(data_dir='src/data/landsat_raw'):
    """Load and normalize dual-band data for teacher inference."""
    import pandas as pd
    
    labels_df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
    X_data = []
    
    print(f"\nLoading {len(labels_df)} dual-band samples for teacher inference...")
    
    for idx, row in labels_df.iterrows():
        sample_id = row['sample_id']
        band7_file = os.path.join(data_dir, 'band7_swir', f'{sample_id}.npy')
        band10_file = os.path.join(data_dir, 'band10_thermal', f'{sample_id}.npy')
        
        band7_dn = np.load(band7_file)
        band10_dn = np.load(band10_file)
        
        # Normalize (same as training)
        band7_norm = (band7_dn - 9090) / (65535 - 9090)
        band10_norm = (band10_dn - 20596) / (65535 - 20596)
        
        # Clip to [0, 1]
        band7_norm = np.clip(band7_norm, 0, 1)
        band10_norm = np.clip(band10_norm, 0, 1)
        
        # Stack channels
        patch = np.stack([band7_norm, band10_norm], axis=-1)
        X_data.append(patch)
        
        if (idx + 1) % 1000 == 0:
            print(f"  Loaded {idx + 1}/{len(labels_df)} samples")
    
    return np.array(X_data)

def get_teacher_predictions(interpreter, input_details, output_details, X_data):
    """
    Get soft labels from teacher model.
    
    Args:
        X_data: (N, 32, 32, 2) normalized dual-band data
    
    Returns:
        predictions: (N,) teacher confidence scores
    """
    predictions = []
    
    # Check if model expects INT8 input (quantized)
    input_dtype = input_details[0]['dtype']
    is_quantized = (input_dtype == np.int8 or input_dtype == np.uint8)
    
    if is_quantized:
        # Get quantization parameters
        input_scale = input_details[0].get('quantization_parameters', {}).get('scales', [1.0])[0]
        input_zero_point = input_details[0].get('quantization_parameters', {}).get('zero_points', [0])[0]
        print(f"Model is quantized (INT8): scale={input_scale}, zero_point={input_zero_point}")
    
    print("\nGenerating teacher predictions (soft labels)...")
    for i in range(len(X_data)):
        # Prepare input
        input_data = X_data[i:i+1]
        
        if is_quantized:
            # Quantize input to INT8
            input_data = input_data / input_scale + input_zero_point
            input_data = np.clip(input_data, -128, 127).astype(np.int8)
        else:
            input_data = input_data.astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Dequantize output if needed
        output_dtype = output_details[0]['dtype']
        if output_dtype == np.int8 or output_dtype == np.uint8:
            output_scale = output_details[0].get('quantization_parameters', {}).get('scales', [1.0])[0]
            output_zero_point = output_details[0].get('quantization_parameters', {}).get('zero_points', [0])[0]
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        
        predictions.append(output[0][0])
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(X_data)} samples")
    
    return np.array(predictions)

def train_student_model(features, teacher_preds, y_true):
    """
    Train lightweight linear student model.
    
    Args:
        features: (N, 5) physics features
        teacher_preds: (N,) soft labels from teacher
        y_true: (N,) ground truth labels
    
    Returns:
        student_model: Trained logistic regression
    """
    # Split data
    X_train, X_test, y_train_soft, y_test_soft, y_train_true, y_test_true = train_test_split(
        features, teacher_preds, y_true, test_size=0.2, random_state=42, stratify=y_true
    )
    
    print("\n" + "=" * 70)
    print("TRAINING LINEAR STUDENT MODEL")
    print("=" * 70)
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")
    
    # Train on soft labels from teacher (thresholded)
    student = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    y_train_binary = (y_train_soft > 0.5).astype(int)
    student.fit(X_train, y_train_binary)
    
    # Evaluate on test set
    y_test_binary = (y_test_soft > 0.5).astype(int)
    train_acc = student.score(X_train, y_train_binary)
    test_acc = student.score(X_test, y_test_binary)
    
    # Also check agreement with ground truth
    test_pred = student.predict(X_test)
    ground_truth_acc = accuracy_score(y_test_true, test_pred)
    
    print(f"\n{'Metric':<30} {'Value':<15}")
    print("-" * 70)
    print(f"{'Training Accuracy':<30} {train_acc * 100:>13.2f}%")
    print(f"{'Test Accuracy (vs Teacher)':<30} {test_acc * 100:>13.2f}%")
    print(f"{'Test Accuracy (vs Ground)':<30} {ground_truth_acc * 100:>13.2f}%")
    
    # Show model parameters
    print(f"\n{'Model Parameters':<30} {'Value':<40}")
    print("-" * 70)
    feature_names = ['Thermal Mean', 'Thermal Max', 'Thermal Std', 'SWIR Max', 'SWIR/Thermal']
    for i, name in enumerate(feature_names):
        print(f"{name:<30} {student.coef_[0][i]:>13.6f}")
    print(f"{'Bias':<30} {student.intercept_[0]:>13.6f}")
    
    print("\nTotal parameters: 6 (5 weights + 1 bias)")
    
    return student

def save_onboard_model(student, output_path='models/onboard_model.npy'):
    """Save student model as lightweight .npy file."""
    # Pack weights and bias
    weights = student.coef_[0]
    bias = student.intercept_[0]
    
    onboard_params = np.concatenate([weights, [bias]])
    
    Path(output_path).parent.mkdir(exist_ok=True)
    np.save(output_path, onboard_params)
    
    file_size = Path(output_path).stat().st_size
    teacher_size = 291792  # bytes (from TFLite model)
    compression_ratio = teacher_size / file_size
    
    print("\n" + "=" * 70)
    print("MODEL COMPRESSION RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'Value':<40}")
    print("-" * 70)
    print(f"{'Teacher Model Size':<30} {teacher_size:>13,} bytes (285 KB)")
    print(f"{'Student Model Size':<30} {file_size:>13,} bytes")
    print(f"{'Compression Ratio':<30} {compression_ratio:>13.0f}×")
    print(f"{'Space Saved':<30} {(teacher_size - file_size) / 1024:>13.1f} KB")
    
    print(f"\n✅ Saved onboard model: {output_path}")
    print("=" * 70)

def main():
    print("=" * 70)
    print("MODEL DISTILLATION PIPELINE")
    print("TinyFireNet (285 KB) → Linear Student (48 bytes)")
    print("=" * 70)
    
    # Check if features exist
    if not Path('src/data/features.npy').exists():
        print("\n❌ Error: Features not found!")
        print("   Run: python src/extract_features.py")
        return
    
    # Load features
    print("\n[1/5] Loading physics features...")
    features = np.load('src/data/features.npy')
    y_true = np.load('src/data/feature_labels.npy')
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape:   {y_true.shape}")
    
    # Load teacher model
    print("\n[2/5] Loading TinyFireNet teacher model...")
    interpreter, input_details, output_details = load_teacher_model()
    
    # Load dual-band data
    print("\n[3/5] Loading dual-band data...")
    X_data = load_dual_band_data()
    print(f"  Data shape: {X_data.shape}")
    
    # Get teacher predictions
    print("\n[4/5] Getting teacher predictions...")
    teacher_preds = get_teacher_predictions(interpreter, input_details, output_details, X_data)
    print(f"  Predictions shape: {teacher_preds.shape}")
    print(f"  Prediction range:  [{teacher_preds.min():.4f}, {teacher_preds.max():.4f}]")
    
    # Train student
    print("\n[5/5] Training student model...")
    student = train_student_model(features, teacher_preds, y_true)
    
    # Save onboard model
    save_onboard_model(student)
    
    print("\n✅ DISTILLATION COMPLETE")
    print("\nNext steps:")
    print("  1. Run: python src/benchmark.py (compare teacher vs student)")
    print("  2. Deploy onboard_model.npy to satellite")

if __name__ == '__main__':
    main()
