import numpy as np
import pandas as pd
from pathlib import Path

def convert_npz_to_csv(npz_file_path, output_csv_path=None):
    """
    Convert an NPZ file containing ATP match data to CSV format.
    
    Args:
        npz_file_path: Path to the .npz file
        output_csv_path: Path for the output CSV file (optional)
    """
    # Load the NPZ file
    print(f"Loading NPZ file: {npz_file_path}")
    data = np.load(npz_file_path, allow_pickle=True)
    
    # Display available arrays in the NPZ file
    print(f"\n📦 Available arrays in NPZ file:")
    for key in data.files:
        print(f"   - {key}: shape {data[key].shape}, dtype {data[key].dtype}")
    
    # Extract train, validation, and test data (they are in transposed format)
    X_train = data['X_train'].T  # Transpose to get (samples, features)
    Y_train = data['Y_train'].T
    X_val = data['X_val'].T
    Y_val = data['Y_val'].T
    X_test = data['X_test'].T
    Y_test = data['Y_test'].T
    
    # Combine all splits
    X = np.vstack([X_train, X_val, X_test])
    y = np.vstack([Y_train, Y_val, Y_test]).flatten()
    
    # Create split indicator
    split_labels = (['train'] * len(X_train) + 
                   ['validation'] * len(X_val) + 
                   ['test'] * len(X_test))
    
    feature_names = data['feature_names']  # Feature column names
    
    print(f"\n✓ Loaded data:")
    print(f"   - Features (X): {X.shape}")
    print(f"   - Labels (y): {y.shape}")
    print(f"   - Feature names: {len(feature_names)} features")
    
    # Create a DataFrame with features
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add the winner column and split indicator
    df['winner'] = y
    df['split'] = split_labels
    
    # Determine output file path
    if output_csv_path is None:
        output_csv_path = npz_file_path.replace('.npz', '.csv')
    
    # Save to CSV
    print(f"\n💾 Saving to CSV: {output_csv_path}")
    df.to_csv(output_csv_path, index=False)
    
    print(f"✓ Successfully converted!")
    print(f"   - Total rows: {len(df)}")
    print(f"   - Total columns: {len(df.columns)}")
    print(f"   - Train samples: {(df['split'] == 'train').sum()}")
    print(f"   - Validation samples: {(df['split'] == 'validation').sum()}")
    print(f"   - Test samples: {(df['split'] == 'test').sum()}")
    
    # Check surface features
    surface_cols = [col for col in df.columns if 'surface' in col]
    if surface_cols:
        print(f"\n📊 Surface feature values (should be 0 or 1):")
        for col in surface_cols:
            unique_vals = df[col].unique()
            print(f"   {col}: {sorted(unique_vals)}")
    
    print(f"\n📊 First few rows:")
    print(df.head())
    
    print(f"\n📊 Column names:")
    print(df.columns.tolist())
    
    return df

def main():
    # Set the path to the NPZ file
    npz_file = Path(__file__).parent / 'atp_featured_dataset.npz'
    
    if not npz_file.exists():
        print(f"❌ Error: File not found: {npz_file}")
        return
    
    # Convert to CSV
    output_csv = Path(__file__).parent / 'atp_featured_dataset.csv'
    df = convert_npz_to_csv(str(npz_file), str(output_csv))
    
    print(f"\n✅ Conversion complete!")
    print(f"   CSV file: {output_csv}")

if __name__ == '__main__':
    main()
