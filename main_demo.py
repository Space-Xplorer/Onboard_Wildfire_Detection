# main_demo.py (Run this for the presentation)
import numpy as np
import tensorflow as tf
from src import flight_software as fs # Member 3's code

# 1. Simulate Satellite Acquisition
print("[SAT] Acquiring Scene...")
# (In real life, load Member 2's Tiff here. For demo, we mock a hot array)
raw_mir_band = np.random.randint(0, 65535, (256, 256))
raw_tir_band = np.random.randint(20000, 35000, (256, 256)) # Warmish DNs

# 2. Run Stage 1 (Physics Trigger)
is_fire, temp_map = fs.run_trigger(raw_mir_band, raw_tir_band)

if is_fire:
    # 3. Preprocess for AI (Normalize Kelvin to 0-1)
    # We clip to expected fire range (280K - 400K)
    print("[AI] Trigger Active. Preprocessing ROI...")
    roi = (temp_map[0:64, 0:64] - 280) / (400 - 280)
    roi = np.clip(roi, 0, 1).astype(np.float32)
    roi = np.expand_dims(roi, axis=-1) # Add channel
    roi = np.expand_dims(roi, axis=0)  # Add batch

    # 4. Run Stage 2 (Quantized Inference)
    # Load TFLite
    interpreter = tf.lite.Interpreter(model_path="models/fire_model_quant.tflite")
    interpreter.allocate_tensors()
    
    # (Note: For INT8, you'd need to cast input to int8 here, keeping float for simplicity of demo script)
    # ... Inference logic ...
    confidence = 0.92 # Mock result for demo flow
    
    print(f"[AI] Inference Complete. Confidence: {confidence*100:.1f}%")
    
    # 5. Run Stage 3 (Packet Gen)
    if confidence > 0.85:
        packet_hex = fs.generate_packet(37.77, -122.41, confidence, 500)
        print(f"[COMMS] ALERT DOWNLINKED: 0x{packet_hex}")
        print(f"[COMMS] Size: {len(bytes.fromhex(packet_hex))} Bytes")
else:
    print("[SAT] No anomalies. Sleeping.")
