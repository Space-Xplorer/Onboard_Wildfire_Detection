# main_demo.py (Run this for the presentation)
"""
main_demo.py - Wildfire Detection Flight Software Demo
Demonstrates complete 3-stage pipeline using the full teacher model (thermal_model.h5).
"""

import numpy as np
import tensorflow as tf
import os
import sys

# Add src to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'src'))

from flight_logic import dn_to_kelvin, check_trigger
from telemetry_encoder import CCSDSFireAlertPacket

# ===== CONFIGURATION =====
MODEL_PATH = 'models/thermal_model.h5'
DATA_DIR = 'src/data/landsat_raw'
SAT_ID = 1

def load_teacher_model():
    """Load the full teacher model (285 KB CNN)."""
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("[ERROR] Please train the model first:")
        print("[ERROR]   python src/train_model.py")
        sys.exit(1)
    
    print(f"[INIT] Loading teacher model: {MODEL_PATH}")
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except ValueError as e:
        if 'batch_shape' in str(e):
            print(f"[ERROR] Model incompatibility with current Keras version")
            print(f"[ERROR] Please retrain the model with current TensorFlow:")
            print(f"[ERROR]   python src/train_model.py")
            sys.exit(1)
        else:
            raise
    
    print(f"[INIT] Model loaded successfully")
    print(f"       Input shape: {model.input_shape}")
    print(f"       Output shape: {model.output_shape}")
    return model

def load_random_sample():
    """Load random sample from synthetic dataset."""
    import pandas as pd
    
    labels_path = os.path.join(DATA_DIR, 'labels.csv')
    df = pd.read_csv(labels_path)
    
    sample_idx = np.random.randint(0, len(df))
    row = df.iloc[sample_idx]
    sample_id = row['sample_id']
    label = row['fire_label']
    
    # Load dual-band data
    band7_path = os.path.join(DATA_DIR, 'band7_swir', f'{sample_id}.npy')
    band10_path = os.path.join(DATA_DIR, 'band10_thermal', f'{sample_id}.npy')
    
    band7_dn = np.load(band7_path).astype(np.float32)
    band10_dn = np.load(band10_path).astype(np.float32)
    
    return band7_dn, band10_dn, label, sample_id

def preprocess_for_teacher(band7_dn, band10_dn):
    """Preprocess dual-band data for teacher model (same as training)."""
    # Normalize each band using min-max scaling
    band7_norm = band7_dn.astype(np.float32)
    band10_norm = band10_dn.astype(np.float32)
    
    band7_min, band7_max = band7_norm.min(), band7_norm.max()
    band10_min, band10_max = band10_norm.min(), band10_norm.max()
    
    band7_range = band7_max - band7_min + 1e-10
    band10_range = band10_max - band10_min + 1e-10
    
    band7_norm = (band7_norm - band7_min) / band7_range
    band10_norm = (band10_norm - band10_min) / band10_range
    
    # Stack into dual-channel (32, 32, 2)
    dual_band = np.stack([band7_norm, band10_norm], axis=-1)
    
    # Add batch dimension: (1, 32, 32, 2)
    return np.expand_dims(dual_band, axis=0)

def main():
    """Run wildfire detection demo."""
    print("=" * 70)
    print("WILDFIRE DETECTION FLIGHT SOFTWARE DEMO")
    print("Using Full Teacher Model (thermal_model.h5)")
    print("=" * 70)
    print()
    
    # Initialize
    model = load_teacher_model()
    encoder = CCSDSFireAlertPacket(sat_id=SAT_ID)
    
    # Simulated satellite parameters
    SAT_LATITUDE = 35.6895
    SAT_LONGITUDE = -120.4068
    
    # Statistics tracking
    stats = {
        'total_frames': 0,
        'physics_triggers': 0,
        'ai_detections': 0,
        'packets_downlinked': 0,
        'ground_truth_fires': 0,
        'true_positives': 0,
        'false_positives': 0,
        'true_negatives': 0,
        'false_negatives': 0
    }
    
    # Process N samples
    N_FRAMES = 12
    
    for frame_idx in range(N_FRAMES):
        print(f"\n{'=' * 70}")
        print(f"[SAT] Acquisition #{frame_idx + 1}/{N_FRAMES}")
        print(f"{'=' * 70}")
        
        # 1. Load sample data (simulates sensor acquisition)
        band7_dn, band10_dn, ground_truth, sample_id = load_random_sample()
        print(f"[SAT] Sample ID: {sample_id}")
        
        ground_truth_label = 'FIRE' if ground_truth == 1 else 'NO FIRE'
        print(f"[SAT] Ground Truth: {ground_truth_label}")
        
        stats['total_frames'] += 1
        if ground_truth == 1:
            stats['ground_truth_fires'] += 1
        
        # 2. Stage 1: Physics Trigger
        print(f"\n[SAT] Stage 1: Physics Trigger")
        trigger = check_trigger(band7_dn.flatten(), band10_dn.flatten())
        
        # Check if any pixel triggered
        trigger_fired = np.any(trigger)
        
        if trigger_fired:
            stats['physics_triggers'] += 1
            max_temp = dn_to_kelvin(band10_dn.max())
            print(f"[PHYSICS] ✓ Trigger activated (T={max_temp:.1f}K)")
            
            # 3. Stage 2: AI Inference with Teacher Model
            print(f"\n[SAT] Stage 2: AI Inference (Teacher Model)")
            
            # Preprocess data
            input_data = preprocess_for_teacher(band7_dn, band10_dn)
            
            # Run inference
            prediction = model.predict(input_data, verbose=0)
            confidence = float(prediction[0][0]) * 100
            fire_class = 1 if confidence > 50 else 0
            
            print(f"[AI] Inference: {'FIRE' if fire_class == 1 else 'NO FIRE'} (Confidence: {confidence:.1f}%)")
            
            if fire_class == 1:
                stats['ai_detections'] += 1
            
            # Track accuracy
            if ground_truth == 1 and fire_class == 1:
                stats['true_positives'] += 1
            elif ground_truth == 0 and fire_class == 1:
                stats['false_positives'] += 1
            elif ground_truth == 0 and fire_class == 0:
                stats['true_negatives'] += 1
            
            # 4. Stage 3: Telemetry Downlink
            if confidence > 85:
                stats['packets_downlinked'] += 1
                print(f"\n[SAT] Stage 3: Telemetry Downlink")
                print(f"[COMMS] ✓ Downlinking CCSDS packet")
                
                packet = encoder.pack_alert(
                    latitude=SAT_LATITUDE,
                    longitude=SAT_LONGITUDE,
                    temperature_k=max_temp,
                    confidence_pct=int(confidence)
                )
                
                packet_hex = packet.hex().upper()
                print(f"[COMMS]   Hex: {packet_hex}")
                print(f"[COMMS]   Size: {len(packet)} bytes")
            else:
                print(f"\n[SAT] Confidence below threshold (85%). No downlink.")
        else:
            print(f"[PHYSICS] ✗ No trigger (cool scene)")
            if ground_truth == 0:
                stats['true_negatives'] += 1
            else:
                stats['false_negatives'] += 1
    
    # Print statistics
    print(f"\n{'=' * 70}")
    print(f"MISSION STATISTICS")
    print(f"{'=' * 70}")
    print(f"Total Frames Processed:     {stats['total_frames']}")
    print(f"Physics Triggers Fired:     {stats['physics_triggers']} ({100*stats['physics_triggers']/stats['total_frames']:.1f}%)")
    print(f"AI Detections:              {stats['ai_detections']}")
    print(f"Packets Downlinked:         {stats['packets_downlinked']}")
    print(f"\n{'Ground Truth Statistics':<30} {'':<20}")
    print(f"  Total Fires:              {stats['ground_truth_fires']}")
    print(f"  Total Background:         {stats['total_frames'] - stats['ground_truth_fires']}")
    print(f"\n{'Accuracy Metrics':<30} {'':<20}")
    
    if stats['true_positives'] + stats['false_negatives'] > 0:
        recall = 100 * stats['true_positives'] / (stats['true_positives'] + stats['false_negatives'])
        print(f"  Recall (Fire Detection):  {recall:.1f}%")
    
    if stats['true_positives'] + stats['false_positives'] > 0:
        precision = 100 * stats['true_positives'] / (stats['true_positives'] + stats['false_positives'])
        print(f"  Precision:                {precision:.1f}%")
    
    total_correct = stats['true_positives'] + stats['true_negatives']
    accuracy = 100 * total_correct / stats['total_frames']
    print(f"  Overall Accuracy:         {accuracy:.1f}%")
    print(f"\n  True Positives:           {stats['true_positives']}")
    print(f"  True Negatives:           {stats['true_negatives']}")
    print(f"  False Positives:          {stats['false_positives']}")
    print(f"  False Negatives:          {stats['false_negatives']}")
    
    print(f"\n{'=' * 70}")
    print(f"DEMO COMPLETE")
    print(f"{'=' * 70}")

if __name__ == '__main__':
    main()
