"""
MASTER SCRIPT - RUN ALL FEATURE ENGINEERING STEPS
==================================================
This script runs all feature engineering steps in sequence:
1. Data Inspection
2. Data Cleaning
3. Feature Engineering
4. NPZ Dataset Creation

"""

import subprocess
import sys
import time
from pathlib import Path

print("="*80)
print("ATP TENNIS MATCH PREDICTION - COMPLETE FEATURE ENGINEERING PIPELINE")
print("="*80)

scripts = [
    ("01_data_inspection.py", "Data Inspection & EDA"),
    ("02_data_cleaning.py", "Data Cleaning"),
    ("03_feature_engineering.py", "Feature Engineering"),
    ("04_create_npz_dataset.py", "NPZ Dataset Creation")
]

total_start = time.time()

for i, (script, description) in enumerate(scripts, 1):
    print(f"\n{'='*80}")
    print(f"STEP {i}/{len(scripts)}: {description}")
    print(f"Running: {script}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ {description} completed in {elapsed:.1f} seconds")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script}")
        print(f"Error: {e}")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

total_elapsed = time.time() - total_start

print("\n" + "="*80)
print("✅ ALL STEPS COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\nTotal time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
print("\n📁 Generated files:")
print("   - atp_matches_cleaned.csv")
print("   - atp_matches_featured.csv")
print("   - atp_featured_dataset.npz")
print("   - Various analysis reports and visualizations")
print("\n🎯 Your feature-engineered dataset is ready for training!")
print("   Load it with: data = np.load('atp_featured_dataset.npz')")
