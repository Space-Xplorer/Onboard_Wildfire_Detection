"""
data_create.py - Synthetic Landsat-8 Dataset Generator
Creates realistic fire detection training data using radiometric physics.
"""

import os
import numpy as np
import csv

# ===== CONFIGURATION =====
NUM_SAMPLES = 10000
PATCH_SIZE = 32

# ===== LANDSAT-8 RADIOMETRIC CONSTANTS (From Spectral Analysis) =====
# Band 7 (SWIR 2, 2.11 μm)
BAND7_REFL_SCALE = 0.0000275

# Band 10 (Thermal, 10.6 μm) - Planck Inversion Parameters
BAND10_ML = 3.3420e-4      # Multiplicative rescaling
BAND10_AL = 0.1            # Additive rescaling
BAND10_K1 = 774.8853       # W/(m² sr μm)
BAND10_K2 = 1321.0789      # Kelvin
BAND10_DN_MIN = 0
BAND10_DN_MAX = 65535

# ===== HELPER FUNCTIONS =====

def get_swir_dn(reflectance):
    """
    Convert apparent reflectance to Band 7 (SWIR) digital number.
    Uses Landsat-8 radiometric calibration.
    
    Args:
        reflectance (float): Apparent reflectance (0-2.5, can exceed 1.0 for fires)
    
    Returns:
        int: Digital number (0-65535, clipped)
    """
    dn = (reflectance + 0.2) / BAND7_REFL_SCALE
    return int(np.clip(dn, 0, 65535))


def get_thermal_dn(temp_kelvin):
    """
    Convert brightness temperature to Band 10 (Thermal) digital number.
    Uses Landsat-8 thermal calibration and Planck inversion.
    
    Args:
        temp_kelvin (float): Brightness temperature in Kelvin
    
    Returns:
        int: Digital number (0-65535)
    """
    # Planck inversion: L = K1 / (exp(K2/T) - 1)
    radiance = BAND10_K1 / (np.exp(BAND10_K2 / max(temp_kelvin, 100)) - 1.0)
    
    # Radiance to DN: DN = (L - AL) / ML
    dn = (radiance - BAND10_AL) / BAND10_ML
    return int(np.clip(dn, BAND10_DN_MIN, BAND10_DN_MAX))


# ===== DATASET GENERATION =====

# Get script directory and construct paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(SCRIPT_DIR, "data", "landsat_raw")

# Create directories if they don't exist
os.makedirs(os.path.join(base_dir, "band7_swir"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "band10_thermal"), exist_ok=True)

rows = []

print(f"Generating {NUM_SAMPLES} synthetic samples based on Landsat 8 Physics...")

for i in range(NUM_SAMPLES):
    r = np.random.rand()
    
    # Initialize empty patches
    band7 = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint16)
    band10 = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint16)
    
    if r < 0.5:
        # --- CLASS 0: BACKGROUND (No Fire) ---
        # Soil/Vegetation is dark in SWIR (Reflectance 0.05 - 0.25)
        # Temp is ambient (285K - 310K / 12°C - 37°C)
        label = 0
        
        # Vectorized generation for speed
        refl_map = np.random.uniform(0.05, 0.25, (PATCH_SIZE, PATCH_SIZE))
        temp_map = np.random.uniform(285, 310, (PATCH_SIZE, PATCH_SIZE))
        
        # Convert to DN
        for x in range(PATCH_SIZE):
            for y in range(PATCH_SIZE):
                band7[x, y] = get_swir_dn(refl_map[x, y])
                band10[x, y] = get_thermal_dn(temp_map[x, y])

    elif r < 0.8:
        # --- CLASS 1: EARLY/SMOLDERING FIRE ---
        # Use Gaussian blob: fires are connected hotspots, not scattered pixels
        # Background temp = 300K, then add Gaussian blob
        label = 1
        
        # Start with background temperature
        temp_map = np.random.normal(300, 5, (PATCH_SIZE, PATCH_SIZE))
        refl_map = np.random.normal(0.15, 0.08, (PATCH_SIZE, PATCH_SIZE))
        
        # Add Gaussian blob for fire hotspot
        x0, y0 = np.random.randint(5, PATCH_SIZE-5, 2)
        sigma = np.random.uniform(2, 4)
        x, y = np.meshgrid(np.arange(PATCH_SIZE), np.arange(PATCH_SIZE))
        d = np.sqrt((x - x0)**2 + (y - y0)**2)
        blob = np.exp(-(d**2) / (2.0 * sigma**2))
        
        # Add heat from blob (smoldering: +60K from background)
        temp_map += blob * 60
        # Add reflectance from blob (smoldering: +0.35)
        refl_map += blob * 0.35
        
        # Clip to physics limits
        temp_map = np.clip(temp_map, 280, 600)
        refl_map = np.clip(refl_map, 0.05, 1.0)
        
        for x in range(PATCH_SIZE):
            for y in range(PATCH_SIZE):
                band7[x, y] = get_swir_dn(refl_map[x, y])
                band10[x, y] = get_thermal_dn(temp_map[x, y])

    else:
        # --- CLASS 1: ACTIVE FLAMING FIRE ---
        # Intense heat with bright Gaussian blob
        label = 1
        
        # Start with warm background
        temp_map = np.random.normal(300, 5, (PATCH_SIZE, PATCH_SIZE))
        refl_map = np.random.normal(0.15, 0.08, (PATCH_SIZE, PATCH_SIZE))
        
        # Add Gaussian blob for intense fire
        x0, y0 = np.random.randint(5, PATCH_SIZE-5, 2)
        sigma = np.random.uniform(2, 5)
        x, y = np.meshgrid(np.arange(PATCH_SIZE), np.arange(PATCH_SIZE))
        d = np.sqrt((x - x0)**2 + (y - y0)**2)
        blob = np.exp(-(d**2) / (2.0 * sigma**2))
        
        # Add intense heat from blob (active fire: +500K from background)
        temp_map += blob * 500
        # Add bright reflectance from blob (active fire: +2.0)
        refl_map += blob * 2.0
        
        # Clip to physics limits
        temp_map = np.clip(temp_map, 280, 1000)
        refl_map = np.clip(refl_map, 0.05, 2.5)
        
        for x in range(PATCH_SIZE):
            for y in range(PATCH_SIZE):
                band7[x, y] = get_swir_dn(refl_map[x, y])
                band10[x, y] = get_thermal_dn(temp_map[x, y])

    # Save files
    sid = f"L8_{i:05d}"
    np.save(os.path.join(base_dir, "band7_swir", f"{sid}.npy"), band7)
    np.save(os.path.join(base_dir, "band10_thermal", f"{sid}.npy"), band10)
    
    rows.append([sid, label])

# Save CSV
with open(os.path.join(base_dir, "labels.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sample_id", "fire_label"])
    writer.writerows(rows)

print(f"Dataset generation complete. Saved to {base_dir}")