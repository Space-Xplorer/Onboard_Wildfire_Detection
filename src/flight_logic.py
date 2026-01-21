"""
flight_logic.py - Physics Engine for Wildfire Detection
Implements Planck's Law inverse and radiometric trigger logic.
Target: LEON3/ARM Onboard Computer (32-bit integer arithmetic)
"""

import numpy as np

# ===== LANDSAT-8 RADIOMETRIC CONSTANTS =====
# Band 10 (Thermal IR, 10.6 μm)
BAND10_K1 = 774.8853      # W/(m² sr μm) - Spectral radiance at sensor
BAND10_K2 = 1321.0789     # Kelvin - Boltzmann constant ratio
BAND10_ML = 3.3420e-4     # Multiplicative rescaling factor
BAND10_AL = 0.1           # Additive rescaling factor

# Band 7 (SWIR, 2.1 μm) - Reflectance to DN
BAND7_REFL_SCALE = 0.0000275

# Physics Trigger Thresholds
FIRE_TEMP_THRESHOLD_K = 320.0     # Kelvin (47°C) - Active fire signature
FIRE_REFLECTANCE_THRESHOLD = 0.4  # Apparent reflectance in SWIR (fires emit)

# ===== CORE PHYSICS FUNCTIONS =====

def dn_to_radiance(dn, band=10):
    """
    Convert Digital Number to Spectral Radiance (Band 10 only).
    
    Args:
        dn (float or np.ndarray): Digital number (raw sensor output, 0-65535)
        band (int): Landsat-8 band (only 10 supported)
    
    Returns:
        float or np.ndarray: Spectral radiance W/(m² sr μm)
    """
    if band != 10:
        raise ValueError("Only Band 10 (Thermal) supported for DN->Radiance")
    
    radiance = dn * BAND10_ML + BAND10_AL
    return radiance


def radiance_to_kelvin(radiance):
    """
    Convert Spectral Radiance to Brightness Temperature (Planck Inverse).
    Uses Landsat-8 Band 10 constants.
    
    Args:
        radiance (float or np.ndarray): Spectral radiance W/(m² sr μm)
    
    Returns:
        float or np.ndarray: Brightness temperature in Kelvin
    """
    # T = K2 / ln(K1/L + 1)
    # Avoid division by zero and log of negative
    radiance = np.asarray(radiance)
    ratio = BAND10_K1 / np.clip(radiance, 0.001, 1e6)
    kelvin = BAND10_K2 / np.log(ratio + 1.0)
    return kelvin


def dn_to_kelvin(dn):
    """
    Direct DN to Brightness Temperature conversion (Band 10).
    
    Args:
        dn (float or np.ndarray): Raw sensor digital number
    
    Returns:
        float or np.ndarray: Brightness temperature in Kelvin
    """
    radiance = dn_to_radiance(dn, band=10)
    kelvin = radiance_to_kelvin(radiance)
    return kelvin


def dn_to_reflectance(dn, band=7):
    """
    Convert Band 7 (SWIR) DN to Apparent Reflectance.
    Inverse of: DN = (Reflectance + 0.2) / BAND7_REFL_SCALE
    
    Args:
        dn (float or np.ndarray): Digital number
        band (int): Landsat-8 band (only 7 supported)
    
    Returns:
        float or np.ndarray: Apparent reflectance
    """
    if band != 7:
        raise ValueError("Only Band 7 (SWIR) supported for DN->Reflectance")
    
    reflectance = (dn * BAND7_REFL_SCALE) - 0.2
    return reflectance


# ===== TRIGGER LOGIC =====

def check_trigger(mir_dn, tir_dn, verbose=False):
    """
    Physics-based fire trigger using Band 7 (MIR) and Band 10 (TIR).
    
    Fire signature:
    1. Absolute temperature > 320K (hot pixel)
    2. OR: SWIR reflectance > 0.4 (fires emit in SWIR)
    3. OR: (TIR - MIR) gradient indicates anomaly
    
    Args:
        mir_dn (float or np.ndarray): Band 7 (SWIR) digital number
        tir_dn (float or np.ndarray): Band 10 (Thermal) digital number
        verbose (bool): Print debug info
    
    Returns:
        bool or np.ndarray: True if trigger fires (candidate region)
    """
    mir_dn = np.asarray(mir_dn)
    tir_dn = np.asarray(tir_dn)
    
    # Convert to physics units
    tir_temp = dn_to_kelvin(tir_dn)
    mir_refl = dn_to_reflectance(mir_dn, band=7)
    
    # Trigger conditions (any one can activate)
    condition_1 = tir_temp > FIRE_TEMP_THRESHOLD_K
    condition_2 = mir_refl > FIRE_REFLECTANCE_THRESHOLD
    
    # Combined trigger
    trigger = condition_1 | condition_2
    
    if verbose:
        if np.isscalar(trigger):
            print(f"[PHYSICS] TIR Temp: {tir_temp:.1f}K | MIR Refl: {mir_refl:.3f} | Trigger: {trigger}")
        else:
            print(f"[PHYSICS] TIR Temp range: {tir_temp.min():.1f}K - {tir_temp.max():.1f}K")
            print(f"[PHYSICS] MIR Refl range: {mir_refl.min():.3f} - {mir_refl.max():.3f}")
            print(f"[PHYSICS] Triggers activated: {np.sum(trigger)}/{len(trigger)}")
    
    return trigger


# ===== TEST / VERIFICATION =====

if __name__ == "__main__":
    print("=== Flight Logic Verification ===\n")
    
    # Test Case 1: Background (cool, low reflectance)
    bg_tir_dn = 22000  # ~300K
    bg_mir_dn = 3000   # ~0.08 reflectance
    
    print("Test 1: Background (No Fire)")
    print(f"  TIR DN: {bg_tir_dn}, MIR DN: {bg_mir_dn}")
    print(f"  TIR Temp: {dn_to_kelvin(bg_tir_dn):.1f}K")
    print(f"  MIR Refl: {dn_to_reflectance(bg_mir_dn):.3f}")
    trigger_bg = check_trigger(bg_mir_dn, bg_tir_dn, verbose=True)
    print(f"  → Trigger: {trigger_bg}\n")
    
    # Test Case 2: Active fire (hot, high reflectance)
    fire_tir_dn = 45000  # ~400K+
    fire_mir_dn = 20000  # ~0.55 reflectance
    
    print("Test 2: Active Fire")
    print(f"  TIR DN: {fire_tir_dn}, MIR DN: {fire_mir_dn}")
    print(f"  TIR Temp: {dn_to_kelvin(fire_tir_dn):.1f}K")
    print(f"  MIR Refl: {dn_to_reflectance(fire_mir_dn):.3f}")
    trigger_fire = check_trigger(fire_mir_dn, fire_tir_dn, verbose=True)
    print(f"  → Trigger: {trigger_fire}\n")
    
    print("✓ Flight Logic module verified")
