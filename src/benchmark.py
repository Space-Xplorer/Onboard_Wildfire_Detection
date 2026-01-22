"""
Benchmark comparison: TinyFireNet (285 KB) vs. Linear Student (48 bytes).

Compares:
- Model size
- Inference time (ms/sample)
- Accuracy
- Precision, Recall, F1
- Memory footprint
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from extract_features import extract_5_features
from profiler import measure_model_size

def sigmoid(x):
    """Sigmoid activation for linear student."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def load_test_data():
    """Load test dataset."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    labels_df = pd.read_csv('src/data/landsat_raw/labels.csv')
    
    # Split to get test indices (same split as distillation)
    train_idx, test_idx = train_test_split(
        range(len(labels_df)), test_size=0.2, random_state=42, 
        stratify=labels_df['fire_label'].values
    )
    
    test_df = labels_df.iloc[test_idx].reset_index(drop=True)
    
    # Load test samples
    X_teacher = []
    X_student = []
    y_true = []
    
    print("=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)
    print(f"\nLoading {len(test_df)} test samples...")
    
    for idx, row in test_df.iterrows():
        sample_id = row['sample_id']
        band7_file = os.path.join('src/data/landsat_raw/band7_swir', f'{sample_id}.npy')
        band10_file = os.path.join('src/data/landsat_raw/band10_thermal', f'{sample_id}.npy')
        
        band7_dn = np.load(band7_file)
        band10_dn = np.load(band10_file)
        
        # Teacher input: normalized dual-band
        band7_norm = (band7_dn - 9090) / (65535 - 9090)
        band10_norm = (band10_dn - 20596) / (65535 - 20596)
        band7_norm = np.clip(band7_norm, 0, 1)
        band10_norm = np.clip(band10_norm, 0, 1)
        patch = np.stack([band7_norm, band10_norm], axis=-1)
        X_teacher.append(patch)
        
        # Student input: 5 physics features
        features = extract_5_features(band7_dn, band10_dn)
        X_student.append(features)
        
        y_true.append(row['fire_label'])
        
        if (idx + 1) % 500 == 0:
            print(f"  Loaded {idx + 1}/{len(test_df)} samples")
    
    X_teacher = np.array(X_teacher)
    X_student = np.array(X_student)
    y_true = np.array(y_true)
    
    print(f"\nTest set loaded:")
    print(f"  Teacher input: {X_teacher.shape}")
    print(f"  Student input: {X_student.shape}")
    print(f"  Labels:        {y_true.shape}")
    print(f"  Class distribution: Fire={np.sum(y_true)}, Background={len(y_true) - np.sum(y_true)}")
    
    return X_teacher, X_student, y_true

def benchmark_teacher(X_test):
    """Benchmark TinyFireNet teacher model."""
    print("\n" + "=" * 70)
    print("BENCHMARKING TEACHER MODEL (TinyFireNet)")
    print("=" * 70)
    
    model_path = 'models/fire_model_quant.tflite'
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Check if model expects INT8 input
    input_dtype = input_details[0]['dtype']
    is_quantized = (input_dtype == np.int8 or input_dtype == np.uint8)
    
    if is_quantized:
        input_scale = input_details[0].get('quantization_parameters', {}).get('scales', [1.0])[0]
        input_zero_point = input_details[0].get('quantization_parameters', {}).get('zero_points', [0])[0]
    
    predictions = []
    start_time = time.perf_counter()
    
    print(f"\nRunning inference on {len(X_test)} samples...")
    for i in range(len(X_test)):
        input_data = X_test[i:i+1]
        
        if is_quantized:
            # Quantize to INT8
            input_data = input_data / input_scale + input_zero_point
            input_data = np.clip(input_data, -128, 127).astype(np.int8)
        else:
            input_data = input_data.astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Dequantize output if needed
        output_dtype = output_details[0]['dtype']
        if output_dtype == np.int8 or output_dtype == np.uint8:
            output_scale = output_details[0].get('quantization_parameters', {}).get('scales', [1.0])[0]
            output_zero_point = output_details[0].get('quantization_parameters', {}).get('zero_points', [0])[0]
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        
        predictions.append(output[0][0])
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(X_test)} samples")
    
    end_time = time.perf_counter()
    
    predictions = np.array(predictions)
    total_time_ms = (end_time - start_time) * 1000
    avg_time_ms = total_time_ms / len(X_test)
    model_size = measure_model_size(model_path)
    
    print(f"\n{'Metric':<25} {'Value':<20}")
    print("-" * 70)
    print(f"{'Model Size':<25} {model_size:<20}")
    print(f"{'Total Inference Time':<25} {total_time_ms:<20.2f} ms")
    print(f"{'Avg Inference Time':<25} {avg_time_ms:<20.3f} ms/sample")
    print(f"{'Throughput':<25} {1000 / avg_time_ms:<20.1f} samples/sec")
    
    return predictions, avg_time_ms

def benchmark_student(X_test):
    """Benchmark linear student model."""
    print("\n" + "=" * 70)
    print("BENCHMARKING STUDENT MODEL (Linear)")
    print("=" * 70)
    
    model_path = 'models/onboard_model.npy'
    if not Path(model_path).exists():
        print(f"\n❌ Error: {model_path} not found!")
        print("   Run: python src/distill.py")
        return None, None
    
    params = np.load(model_path)
    weights = params[:5]
    bias = params[5]
    
    predictions = []
    start_time = time.perf_counter()
    
    print(f"\nRunning inference on {len(X_test)} samples...")
    for i in range(len(X_test)):
        logit = np.dot(X_test[i], weights) + bias
        pred = sigmoid(logit)
        predictions.append(pred)
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(X_test)} samples")
    
    end_time = time.perf_counter()
    
    predictions = np.array(predictions)
    total_time_ms = (end_time - start_time) * 1000
    avg_time_ms = total_time_ms / len(X_test)
    model_size = measure_model_size(model_path)
    
    print(f"\n{'Metric':<25} {'Value':<20}")
    print("-" * 70)
    print(f"{'Model Size':<25} {model_size:<20}")
    print(f"{'Total Inference Time':<25} {total_time_ms:<20.2f} ms")
    print(f"{'Avg Inference Time':<25} {avg_time_ms:<20.3f} ms/sample")
    print(f"{'Throughput':<25} {1000 / avg_time_ms:<20.1f} samples/sec")
    
    return predictions, avg_time_ms

def print_classification_metrics(y_true, y_pred, model_name, threshold=0.5):
    """Print comprehensive classification metrics."""
    y_pred_binary = (y_pred > threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred_binary)
    prec = precision_score(y_true, y_pred_binary, zero_division=0)
    rec = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    print(f"\n{'Metric':<25} {'Value':<20}")
    print("-" * 70)
    print(f"{'Accuracy':<25} {acc * 100:<20.2f}%")
    print(f"{'Precision':<25} {prec * 100:<20.2f}%")
    print(f"{'Recall':<25} {rec * 100:<20.2f}%")
    print(f"{'F1 Score':<25} {f1 * 100:<20.2f}%")
    
    print(f"\n{'Confusion Matrix':<25}")
    print("-" * 70)
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")
    
    return acc, prec, rec, f1

def print_comparison_table(teacher_metrics, student_metrics):
    """Print side-by-side comparison."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    teacher_size = 291792  # bytes
    student_size = Path('models/onboard_model.npy').stat().st_size
    
    print(f"\n{'Metric':<30} {'Teacher (CNN)':<20} {'Student (Linear)':<20}")
    print("-" * 70)
    print(f"{'Model Size':<30} {'285 KB':<20} {f'{student_size} bytes':<20}")
    print(f"{'Inference Time (ms)':<30} {teacher_metrics['time']:<20.3f} {student_metrics['time']:<20.3f}")
    print(f"{'Accuracy':<30} {teacher_metrics['acc'] * 100:<20.2f}% {student_metrics['acc'] * 100:<20.2f}%")
    print(f"{'Precision':<30} {teacher_metrics['prec'] * 100:<20.2f}% {student_metrics['prec'] * 100:<20.2f}%")
    print(f"{'Recall':<30} {teacher_metrics['rec'] * 100:<20.2f}% {student_metrics['rec'] * 100:<20.2f}%")
    print(f"{'F1 Score':<30} {teacher_metrics['f1'] * 100:<20.2f}% {student_metrics['f1'] * 100:<20.2f}%")
    
    speedup = teacher_metrics['time'] / student_metrics['time']
    compression = teacher_size / student_size
    acc_drop = abs(teacher_metrics['acc'] - student_metrics['acc']) * 100
    
    print("\n" + "-" * 70)
    print(f"{'IMPROVEMENT FACTORS':<30}")
    print("-" * 70)
    print(f"{'Speedup:':<30} {speedup:<20.1f}×")
    print(f"{'Compression Ratio:':<30} {compression:<20.0f}×")
    print(f"{'Accuracy Drop:':<30} {acc_drop:<20.2f}%")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    
    if acc_drop < 2.0 and speedup > 5:
        print("✅ Student model is EXCELLENT for deployment!")
        print(f"   - Minimal accuracy loss ({acc_drop:.2f}%)")
        print(f"   - {speedup:.0f}× faster inference")
        print(f"   - {compression:.0f}× smaller model size")
    elif acc_drop < 5.0:
        print("✅ Student model is GOOD for deployment")
        print(f"   - Acceptable accuracy loss ({acc_drop:.2f}%)")
        print(f"   - Significant resource savings")
    else:
        print("⚠️  Student model may need improvement")
        print(f"   - Accuracy drop is significant ({acc_drop:.2f}%)")
        print("   - Consider adding more features or using polynomial features")
    
    print("=" * 70)

def main():
    print("=" * 70)
    print("MODEL BENCHMARK: TEACHER vs STUDENT")
    print("=" * 70)
    
    # Load test data
    X_teacher_test, X_student_test, y_true = load_test_data()
    
    # Benchmark teacher
    teacher_preds, teacher_time = benchmark_teacher(X_teacher_test)
    teacher_acc, teacher_prec, teacher_rec, teacher_f1 = print_classification_metrics(
        y_true, teacher_preds, "Teacher (TinyFireNet)"
    )
    
    # Benchmark student
    student_preds, student_time = benchmark_student(X_student_test)
    if student_preds is None:
        return
    
    student_acc, student_prec, student_rec, student_f1 = print_classification_metrics(
        y_true, student_preds, "Student (Linear)"
    )
    
    # Comparison
    teacher_metrics = {
        'time': teacher_time,
        'acc': teacher_acc,
        'prec': teacher_prec,
        'rec': teacher_rec,
        'f1': teacher_f1
    }
    
    student_metrics = {
        'time': student_time,
        'acc': student_acc,
        'prec': student_prec,
        'rec': student_rec,
        'f1': student_f1
    }
    
    print_comparison_table(teacher_metrics, student_metrics)

if __name__ == '__main__':
    main()
