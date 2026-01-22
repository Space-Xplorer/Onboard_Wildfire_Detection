"""
Extract physics-aware features from Landsat-8 dual-band data.
Used for training lightweight linear student model.

Physics Features (5):
1. Thermal Mean: Average brightness temperature
2. Thermal Max: Peak thermal signature (hotspot detection)
3. Thermal Std: Temperature variability
4. SWIR Max: Peak SWIR reflectance (fire response)
5. SWIR/Thermal Ratio: Spectral contrast metric
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

def extract_5_features(band7_dn, band10_dn):
    """
    Extract 5 physics-aware features from dual-band patch.
    
    Args:
        band7_dn: (32, 32) SWIR Band 7 DN values
        band10_dn: (32, 32) Thermal Band 10 DN values
    
    Returns:
        features: (5,) array [thermal_mean, thermal_max, thermal_std, swir_max, ratio]
    """
    # Normalize to [0, 1] using same ranges as training
    band7_norm = (band7_dn - 9090) / (65535 - 9090)
    band10_norm = (band10_dn - 20596) / (65535 - 20596)
    
    # Clip to valid range
    band7_norm = np.clip(band7_norm, 0, 1)
    band10_norm = np.clip(band10_norm, 0, 1)
    
    thermal_mean = np.mean(band10_norm)
    thermal_max = np.max(band10_norm)
    thermal_std = np.std(band10_norm)
    swir_max = np.max(band7_norm)
    
    # Avoid division by zero
    ratio = swir_max / (thermal_max + 1e-8)
    
    return np.array([thermal_mean, thermal_max, thermal_std, swir_max, ratio])

def extract_all_features(data_dir='src/data/landsat_raw'):
    """Extract features from all samples in dataset."""
    labels_df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
    
    features_list = []
    labels_list = []
    
    print("=" * 70)
    print("PHYSICS FEATURE EXTRACTION")
    print("=" * 70)
    print(f"\nExtracting 5 physics features from {len(labels_df)} samples...")
    
    for idx, row in labels_df.iterrows():
        sample_id = row['sample_id']
        band7_file = os.path.join(data_dir, 'band7_swir', f'{sample_id}.npy')
        band10_file = os.path.join(data_dir, 'band10_thermal', f'{sample_id}.npy')
        
        band7_dn = np.load(band7_file)
        band10_dn = np.load(band10_file)
        
        features = extract_5_features(band7_dn, band10_dn)
        features_list.append(features)
        labels_list.append(row['fire_label'])
        
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(labels_df)} samples")
    
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)
    
    # Save features
    output_dir = Path('src/data')
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / 'features.npy', features_array)
    np.save(output_dir / 'feature_labels.npy', labels_array)
    
    print(f"\n{'Feature':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 70)
    feature_names = ['Thermal Mean', 'Thermal Max', 'Thermal Std', 'SWIR Max', 'SWIR/Thermal']
    for i, name in enumerate(feature_names):
        print(f"{name:<20} {features_array[:, i].mean():>11.4f}  {features_array[:, i].std():>11.4f}  {features_array[:, i].min():>11.4f}  {features_array[:, i].max():>11.4f}")
    
    print("\n✅ Saved features:")
    print(f"   src/data/features.npy ({features_array.shape})")
    print(f"   src/data/feature_labels.npy ({labels_array.shape})")
    print("=" * 70)
    
    return features_array, labels_array

if __name__ == '__main__':
    extract_all_features()
