import pandas as pd
import numpy as np
import os

# --- SETTINGS ---
path = '../datas/'  # Make sure the path is correct
output_file = "atp_basic_dataset.npz"

# 1. Create File Lists
training_years = [str(y) for y in range(1968, 2022)]
validation_years = [str(y) for y in range(2022, 2024)]
test_file = path + "atp_matches_2024.csv"

training_files = [path + f"atp_matches_{y}.csv" for y in training_years]
validation_files = [path + f"atp_matches_{y}.csv" for y in validation_years]

# 2. Raw Columns to Use
RAW_FEATURES = [
    'winner_id', 'loser_id', 'winner_rank_points', 'loser_rank_points', 
    'winner_age', 'loser_age', 'winner_ht', 'loser_ht',
    'surface', 'winner_hand', 'loser_hand'
]

# 3. Helper Functions
def load_and_concatenate(file_list):
    li = []
    for filename in file_list:
        try:
            df_temp = pd.read_csv(filename, index_col=None, header=0, usecols=RAW_FEATURES)
            li.append(df_temp)
            print(f"   ✓ Loaded: {filename} ({len(df_temp)} rows)")
        except Exception as e:
            print(f"   ✗ Skipped: {filename} -> {e}")
            continue
    if li:
        return pd.concat(li, axis=0, ignore_index=True)
    return pd.DataFrame()

def create_anonymous_dataframe(df):
    # Drop rows with missing values only in critical columns
    # (Other missing values will be filled later with fillna)
    CRITICAL_COLUMNS = ['winner_id', 'loser_id', 'winner_rank_points', 'loser_rank_points',
                        'winner_age', 'loser_age', 'winner_ht', 'loser_ht',
                        'winner_hand', 'loser_hand', 'surface']
    df_clean = df[RAW_FEATURES].copy().dropna(subset=CRITICAL_COLUMNS)
    
    # Randomize p1/p2
    df_clean['p1_id'] = np.where(np.random.rand(len(df_clean)) > 0.5, df_clean['winner_id'], df_clean['loser_id'])
    # p2 is the person who is not p1
    df_clean['p2_id'] = np.where(df_clean['p1_id'] == df_clean['winner_id'], df_clean['loser_id'], df_clean['winner_id'])

    # Assign statistics
    df_clean['p1_rank_points'] = np.where(df_clean['p1_id'] == df_clean['winner_id'], df_clean['winner_rank_points'], df_clean['loser_rank_points'])
    df_clean['p2_rank_points'] = np.where(df_clean['p1_id'] == df_clean['winner_id'], df_clean['loser_rank_points'], df_clean['winner_rank_points'])
    
    df_clean['p1_age'] = np.where(df_clean['p1_id'] == df_clean['winner_id'], df_clean['winner_age'], df_clean['loser_age'])
    df_clean['p2_age'] = np.where(df_clean['p1_id'] == df_clean['winner_id'], df_clean['loser_age'], df_clean['winner_age'])
    
    df_clean['p1_ht'] = np.where(df_clean['p1_id'] == df_clean['winner_id'], df_clean['winner_ht'], df_clean['loser_ht'])
    df_clean['p2_ht'] = np.where(df_clean['p1_id'] == df_clean['winner_id'], df_clean['loser_ht'], df_clean['winner_ht'])
    
    df_clean['p1_hand'] = np.where(df_clean['p1_id'] == df_clean['winner_id'], df_clean['winner_hand'], df_clean['loser_hand'])
    df_clean['p2_hand'] = np.where(df_clean['p1_id'] == df_clean['winner_id'], df_clean['loser_hand'], df_clean['winner_hand'])

    # Y Label: 1 if p1 won
    Y_df = (df_clean['winner_id'] == df_clean['p1_id']).astype(int)
    
    features_to_process = [
        'p1_rank_points', 'p2_rank_points', 'p1_age', 'p2_age', 
        'p1_ht', 'p2_ht', 'p1_hand', 'p2_hand', 'surface'
    ]
    return df_clean[features_to_process], Y_df

# --- PROCESSING FLOW ---
print("1. Loading raw data files...")
df_train_full = load_and_concatenate(training_files)
df_val_full = load_and_concatenate(validation_files)
try:
    df_test = pd.read_csv(test_file, usecols=RAW_FEATURES)
except:
    df_test = pd.DataFrame()

print(f"\n2. Anonymizing data...")
print(f"   Raw data counts: Train={len(df_train_full)}, Val={len(df_val_full)}, Test={len(df_test)}")
X_train_df, Y_train_df = create_anonymous_dataframe(df_train_full)
X_val_df, Y_val_df = create_anonymous_dataframe(df_val_full)
X_test_df, Y_test_df = create_anonymous_dataframe(df_test)
print(f"   After dropna: Train={len(X_train_df)}, Val={len(X_val_df)}, Test={len(X_test_df)}")
print(f"   Total kept: {len(X_train_df) + len(X_val_df) + len(X_test_df)}")

# Convert labels to Numpy
Y_train = Y_train_df.values.reshape(1, -1)
Y_val = Y_val_df.values.reshape(1, -1)
Y_test = Y_test_df.values.reshape(1, -1)

print("3. Performing feature engineering (One-Hot Encoding)...")
X_train_df['dataset'] = 'train'
X_val_df['dataset'] = 'val'
X_test_df['dataset'] = 'test'

df_combined = pd.concat([X_train_df, X_val_df, X_test_df], axis=0, ignore_index=True)

# Hand processing
df_combined['p1_hand'] = df_combined['p1_hand'].replace('U', 'R').fillna('R')
df_combined['p2_hand'] = df_combined['p2_hand'].replace('U', 'R').fillna('R')
df_combined['p1_is_L'] = (df_combined['p1_hand'] == 'L').astype(int)
df_combined['p2_is_L'] = (df_combined['p2_hand'] == 'L').astype(int)

# Surface processing
df_combined['surface'] = df_combined['surface'].replace('Carpet', 'Hard').fillna('U')
df_combined = pd.get_dummies(df_combined, columns=['surface'], prefix='surface')

# Feature List
base_features = ['p1_rank_points', 'p2_rank_points', 'p1_age', 'p2_age', 'p1_ht', 'p2_ht']
hand_features = ['p1_is_L', 'p2_is_L']
surface_features = [col for col in df_combined.columns if col.startswith('surface_') and col != 'surface_U']
FINAL_FEATURES = base_features + hand_features + surface_features

# Fill missing values
df_combined[base_features] = df_combined[base_features].fillna(df_combined[base_features].mean())

# Separate data back
df_train_final = df_combined[df_combined['dataset'] == 'train'][FINAL_FEATURES]
df_val_final = df_combined[df_combined['dataset'] == 'val'][FINAL_FEATURES]
df_test_final = df_combined[df_combined['dataset'] == 'test'][FINAL_FEATURES]

print("4. Normalizing and converting to Numpy...")
X_train_np = df_train_final.T.values.astype(np.float64)
X_val_np = df_val_final.T.values.astype(np.float64)
X_test_np = df_test_final.T.values.astype(np.float64)

# Calculate and store normalization factor (We need to save this too!)
X_max_per_feature = np.max(X_train_np, axis=1, keepdims=True)
X_max_per_feature[X_max_per_feature == 0] = 1.0

X_train = X_train_np / X_max_per_feature
X_val = X_val_np / X_max_per_feature
X_test = X_test_np / X_max_per_feature

print(f"5. Saving data to {output_file}...")
# np.savez_compressed: Saves data in compressed .npz format
np.savez_compressed(
    output_file,
    X_train=X_train,
    Y_train=Y_train,
    X_val=X_val,
    Y_val=Y_val,
    X_test=X_test,
    Y_test=Y_test,
    X_max=X_max_per_feature, # Save normalization coefficients (needed for future predictions)
    feature_names=np.array(FINAL_FEATURES)  # Save feature names
)

print("Process completed! You can use this .npz file in your model.")