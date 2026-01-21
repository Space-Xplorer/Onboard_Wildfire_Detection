"""
main_flight_computer.py - Flight Software Orchestrator
Integrates physics trigger → AI inference → telemetry downlink.
Designed to run on legacy Earth observation satellites (LEON3/ARM).
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import tensorflow as tf
from flight_logic import dn_to_kelvin, check_trigger, FIRE_TEMP_THRESHOLD_K
from telemetry_encoder import CCSDSFireAlertPacket

# ===== FLIGHT SOFTWARE CONFIGURATION =====

# Inference thresholds
AI_CONFIDENCE_THRESHOLD = 85  # Only downlink if AI confidence > 85%
BATCH_ACQUISITION_SIZE = 1   # Process N patches per acquisition cycle

# Simulated satellite parameters
SAT_ID = 1
SAT_LATITUDE = 35.6895   # Placeholder (would come from orbit propagator)
SAT_LONGITUDE = -120.4068
SAT_ALTITUDE_KM = 705    # Landsat-8 altitude

# Paths
DATA_DIR = os.path.join(SCRIPT_DIR, 'data', 'landsat_raw')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'fire_model_quant.tflite')


class FlightComputer:
    """
    Main flight software orchestrator.
    Manages acquisition, physics trigger, AI inference, and telemetry.
    """
    
    def __init__(self, sat_id=SAT_ID, verbose=True):
        """
        Initialize flight computer.
        
        Args:
            sat_id (int): Satellite ID for telemetry packets
            verbose (bool): Print debug logs
        """
        self.sat_id = sat_id
        self.verbose = verbose
        self.ccsds_encoder = CCSDSFireAlertPacket(sat_id=sat_id)
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        # Statistics
        self.frames_processed = 0
        self.triggers_fired = 0
        self.ai_detections = 0
        self.packets_downlinked = 0
        
        self._load_model()
    
    def _load_model(self):
        """Load quantized TFLite model into memory."""
        if not os.path.exists(MODEL_PATH):
            print(f"[WARNING] Model not found: {MODEL_PATH}")
            print(f"[INFO] Run 'python src/train_model.py' to train and quantize the model first.")
            self.interpreter = None
            return
        
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        if self.verbose:
            print(f"[SAT] TFLite model loaded ({os.path.getsize(MODEL_PATH)} bytes)")
            print(f"[SAT] Input shape: {self.input_details[0]['shape']}")
            print(f"[SAT] Output shape: {self.output_details[0]['shape']}\n")
    
    def load_sample_data(self, sample_idx=None):
        """
        Load synthetic sample from disk (simulates sensor acquisition).
        
        Args:
            sample_idx (int): Sample index to load. If None, random sample.
        
        Returns:
            tuple: (band7_dn, band10_dn, label, sample_id)
        """
        labels_path = os.path.join(DATA_DIR, 'labels.csv')
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        df = pd.read_csv(labels_path)
        
        if sample_idx is None:
            sample_idx = np.random.randint(0, len(df))
        
        row = df.iloc[sample_idx]
        sample_id = row['sample_id']
        label = row['fire_label']
        
        # Load Band 7 (SWIR)
        band7_path = os.path.join(DATA_DIR, 'band7_swir', f'{sample_id}.npy')
        band7_dn = np.load(band7_path).astype(np.float32)
        
        # Load Band 10 (Thermal)
        band10_path = os.path.join(DATA_DIR, 'band10_thermal', f'{sample_id}.npy')
        band10_dn = np.load(band10_path).astype(np.float32)
        
        return band7_dn, band10_dn, label, sample_id
    
    def stage_1_physics_trigger(self, band7_dn, band10_dn):
        """
        Stage 1: Physics-based trigger.
        Lightweight radiometric check using Planck's Law.
        
        Returns:
            bool: True if anomaly detected (wake AI)
        """
        trigger_array = check_trigger(band7_dn, band10_dn, verbose=False)
        
        # Convert array result to scalar (True if ANY pixel triggers)
        trigger_active = bool(np.any(trigger_array)) if isinstance(trigger_array, np.ndarray) else bool(trigger_array)
        
        if trigger_active:
            tir_temp = dn_to_kelvin(band10_dn)
            if self.verbose:
                print(f"[PHYSICS] TRIGGER ACTIVATED (T={tir_temp.mean():.1f}K)")
        
        return trigger_active
    
    def stage_2_ai_inference(self, band7_dn, band10_dn):
        """
        Stage 2: AI inference on dual-band thermal patch.
        Runs quantized INT8 TinyFireNet model.
        
        Args:
            band7_dn (np.ndarray): SWIR patch (32x32, uint16 or float32)
            band10_dn (np.ndarray): Thermal patch (32x32, uint16 or float32)
        
        Returns:
            tuple: (fire_class, confidence)
                - fire_class (int): 0=No Fire, 1=Fire
                - confidence (float): 0-100%
        """
        if self.interpreter is None:
            if self.verbose:
                print(f"[AI] MODEL NOT LOADED. Defaulting to 0% confidence.")
            return 0, 0.0
        
        # Normalize BOTH bands the same way as training
        band7_normalized = band7_dn.astype(np.float32)
        band10_normalized = band10_dn.astype(np.float32)
        
        # Min-max normalization (matching train_model.py)
        band7_min, band7_max = band7_normalized.min(), band7_normalized.max()
        band10_min, band10_max = band10_normalized.min(), band10_normalized.max()
        
        band7_range = band7_max - band7_min + 1e-10
        band10_range = band10_max - band10_min + 1e-10
        
        band7_normalized = (band7_normalized - band7_min) / band7_range
        band10_normalized = (band10_normalized - band10_min) / band10_range
        
        # Stack into dual-channel (32, 32, 2)
        dual_band = np.stack([band7_normalized, band10_normalized], axis=-1)
        
        # Add batch dimension: (1, 32, 32, 2)
        input_data = dual_band.reshape(1, 32, 32, 2).astype(
            self.input_details[0]['dtype']
        )
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Parse output (single sigmoid output for binary classification)
        fire_prob = float(output.flat[0])
        fire_prob = np.clip(fire_prob, 0.0, 1.0)
        
        confidence = fire_prob * 100.0
        fire_class = 1 if fire_prob > 0.5 else 0
        
        if self.verbose:
            class_name = "FIRE" if fire_class else "NO_FIRE"
            print(f"[AI] {class_name} (Confidence: {confidence:.1f}%)")
        
        return fire_class, confidence
    
    def stage_3_telemetry_downlink(self, fire_class, confidence, 
                                   sample_id, latitude=SAT_LATITUDE, 
                                   longitude=SAT_LONGITUDE):
        """
        Stage 3: Generate CCSDS telemetry packet and downlink.
        
        Returns:
            str: Hex packet (26 bytes)
        """
        if confidence < AI_CONFIDENCE_THRESHOLD:
            if self.verbose:
                print(f"[COMMS] CONFIDENCE BELOW THRESHOLD: {confidence:.1f}% < {AI_CONFIDENCE_THRESHOLD}%")
            return None
        
        if fire_class == 0:
            if self.verbose:
                print(f"[COMMS] NO FIRE DETECTED")
            return None
        
        # Generate alert packet
        packet_hex = self.ccsds_encoder.hex_packet(
            latitude=latitude,
            longitude=longitude,
            temperature_k=dn_to_kelvin(np.ones((32, 32)) * 45000).mean(),  # Placeholder
            confidence_pct=int(confidence),
            status_flags=0x00
        )
        
        if self.verbose:
            print(f"[COMMS] PACKET DOWNLINK")
            print(f"[COMMS]   Hex: {packet_hex}")
            print(f"[COMMS]   Packet size: 26 bytes")
        
        self.packets_downlinked += 1
        return packet_hex
    
    def process_frame(self, band7_dn=None, band10_dn=None, sample_id=None):
        """
        Process single acquisition frame through 3-stage pipeline.
        
        Args:
            band7_dn, band10_dn: Sensor data (if None, load random sample)
        
        Returns:
            dict: Processing results
        """
        print(f"\n{'='*60}")
        print(f"[SAT] Acquisition #{self.frames_processed + 1}")
        print(f"{'='*60}")
        
        # Load data if not provided
        if band7_dn is None or band10_dn is None:
            band7_dn, band10_dn, label, sample_id = self.load_sample_data()
        
        self.frames_processed += 1
        
        results = {
            'frame_id': self.frames_processed,
            'sample_id': sample_id,
            'physics_trigger': False,
            'ai_detection': False,
            'confidence': 0.0,
            'packet_hex': None,
            'error': None
        }
        
        try:
            # ===== STAGE 1: Physics Trigger =====
            print("\n[SAT] Stage 1: Physics Trigger")
            trigger = self.stage_1_physics_trigger(band7_dn, band10_dn)
            results['physics_trigger'] = trigger
            
            if not trigger:
                if self.verbose:
                    print(f"[PHYSICS] TRIGGER INACTIVE")
                return results
            
            self.triggers_fired += 1
            
            # ===== STAGE 2: AI Inference =====
            print("\n[SAT] Stage 2: AI Inference")
            fire_class, confidence = self.stage_2_ai_inference(band7_dn, band10_dn)
            results['ai_detection'] = (fire_class == 1)
            results['confidence'] = confidence
            
            if fire_class == 0:
                return results
            
            self.ai_detections += 1
            
            # ===== STAGE 3: Telemetry Downlink =====
            print("\n[SAT] Stage 3: Telemetry Downlink")
            packet_hex = self.stage_3_telemetry_downlink(fire_class, confidence, sample_id)
            results['packet_hex'] = packet_hex
        
        except Exception as e:
            results['error'] = str(e)
            if self.verbose:
                print(f"[ERROR] {e}")
        
        return results
    
    def print_statistics(self):
        """Print flight statistics."""
        print(f"\n{'='*60}")
        print(f"[STATS] Flight Computer Summary")
        print(f"{'='*60}")
        print(f"  Frames processed:     {self.frames_processed}")
        print(f"  Physics triggers:     {self.triggers_fired}")
        print(f"  AI detections:        {self.ai_detections}")
        print(f"  Packets downlinked:   {self.packets_downlinked}")
        print(f"  Trigger rate:         {100*self.triggers_fired/max(1,self.frames_processed):.1f}%")
        print(f"  Detection rate:       {100*self.ai_detections/max(1,self.triggers_fired):.1f}%")
        print(f"{'='*60}\n")


# ===== DEMO EXECUTION =====

def main():
    """Main flight software demo."""
    print(f"\n{'='*60}")
    print(f"WILDFIRE DETECTION ONBOARD SATELLITE")
    print(f"Legacy-Compatible Flight Software Prototype")
    print(f"RTX 4060 / Synthetic Landsat-8 Data")
    print(f"{'='*60}\n")
    
    # Initialize flight computer
    flight_computer = FlightComputer(sat_id=SAT_ID, verbose=True)
    
    # Process N random samples
    N_FRAMES = 10  # Run 10 acquisition cycles
    
    for i in range(N_FRAMES):
        results = flight_computer.process_frame()
        
        if results['error']:
            print(f"[ERROR] {results['error']}")
            break
    
    # Print summary
    flight_computer.print_statistics()


if __name__ == "__main__":
    main()
