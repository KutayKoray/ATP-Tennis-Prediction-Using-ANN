import numpy as np
import pandas as pd
import glob
import os
from collections import defaultdict, deque
import gc
from warnings import filterwarnings

print("--- 'Premium Yakıt' Üretim Script'i Başlatıldı (Elo, H2H, Momentum) ---")
print("--- UYARI: Bu işlem RAM kullanımı düşük ancak ÇOK YAVAŞ olacaktır (30-60+ dk). ---")
filterwarnings('ignore')

# ===============================================
# A) VERİ YÜKLEME VE TEMEL HAZIRLIK
# ===============================================

# --- 1. Dosya Yolları ---
matches_file = '../datas/all_atp_matches_1968_2024.csv'
players_file = '../datas/atp_players.csv'
output_npz_file = 'processed_data_V2_ELO.npz' # Nihai dosya adı

# --- 2. Verileri Yükleme ---
print(f"Adım 1/7: Ana maç dosyası ({matches_file}) yükleniyor...")
try:
    df_all = pd.read_csv(matches_file, low_memory=False)
except FileNotFoundError:
    print(f"HATA: {matches_file} bulunamadı.")
    exit()
except Exception as e:
    print(f"HATA: {matches_file} yüklenirken sorun oluştu: {e}")
    exit()

print(f"Adım 2/7: Oyuncu dosyası ({players_file}) yükleniyor ve eksik veriler dolduruluyor...")
try:
    # 'ht' değil, 'height' sütununu kullan
    df_players = pd.read_csv(players_file, usecols=['player_id', 'hand', 'height']) 
    df_players['player_id'] = pd.to_numeric(df_players['player_id'], errors='coerce')
    df_players = df_players.dropna(subset=['player_id'])
    df_players = df_players.drop_duplicates(subset=['player_id'])
    
    # Winner
    df_all = pd.merge(df_all, df_players.rename(columns={'hand': 'w_hand_fill', 'height': 'w_ht_fill'}), 
                      left_on='winner_id', right_on='player_id', how='left')
    df_all['winner_hand'].fillna(df_all['w_hand_fill'], inplace=True)
    df_all['winner_ht'].fillna(df_all['w_ht_fill'], inplace=True) # 'ht' -> 'height'
    # Loser
    df_all = pd.merge(df_all, df_players.rename(columns={'hand': 'l_hand_fill', 'height': 'l_ht_fill'}), 
                      left_on='loser_id', right_on='player_id', how='left')
    df_all['loser_hand'].fillna(df_all['l_hand_fill'], inplace=True)
    df_all['loser_ht'].fillna(df_all['l_ht_fill'], inplace=True) # 'ht' -> 'height'
    
    df_all = df_all.drop(columns=['player_id_x', 'w_hand_fill', 'w_ht_fill', 'player_id_y', 'l_hand_fill', 'l_ht_fill'])
    
except FileNotFoundError:
    print(f"UYARI: {players_file} bulunamadı. Boy (ht) ve El (hand) verileri eksik kalabilir.")
except Exception as e:
    print(f"UYARI: {players_file} işlenirken sorun oluştu: {e}")


# --- 3. Veriyi Sıralama (Kritik!) ---
df_all['tourney_date'] = pd.to_datetime(df_all['tourney_date'], format='%Y%m%d')
df_all.sort_values(by=['tourney_date', 'match_num'], inplace=True)
df_all.reset_index(drop=True, inplace=True)
df_all['match_id'] = df_all.index.astype(np.int32)

# --- 4. 'dataset_marker' Sütununu Oluşturma (Train/Val/Test Ayırımı) ---
print("   ...'dataset_marker' (Train/Val/Test) sütunu oluşturuluyor...")
df_all['tourney_year'] = df_all['tourney_date'].dt.year
# Sizin istediğiniz ayar (Val=2023)
df_all['dataset_marker'] = 'train' # Varsayılan: 1968-2022
df_all.loc[df_all['tourney_year'] == 2023, 'dataset_marker'] = 'val'
df_all.loc[df_all['tourney_year'] >= 2024, 'dataset_marker'] = 'test' # 2024 ve sonrası

# --- 5. Gerekli Sütunları Temizleme ve Optimize Etme ---
df_all['surface'] = df_all['surface'].astype('category')
df_all['winner_id'] = pd.to_numeric(df_all['winner_id'], errors='coerce').fillna(0).astype(np.int32)
df_all['loser_id'] = pd.to_numeric(df_all['loser_id'], errors='coerce').fillna(0).astype(np.int32)
df_all['winner_rank'] = pd.to_numeric(df_all['winner_rank'], errors='coerce')
df_all['loser_rank'] = pd.to_numeric(df_all['loser_rank'], errors='coerce')
df_all['minutes'] = pd.to_numeric(df_all['minutes'], errors='coerce').fillna(0)
# Servis istatistiklerini hesaplamak için
df_all['w_ace_rate'] = (df_all['w_ace'] / df_all['w_svpt']).fillna(0).astype(np.float32)
df_all['w_1stWon_rate'] = (df_all['w_1stWon'] / df_all['w_1stIn']).fillna(0).astype(np.float32)
df_all['l_ace_rate'] = (df_all['l_ace'] / df_all['l_svpt']).fillna(0).astype(np.float32)
df_all['l_1stWon_rate'] = (df_all['l_1stWon'] / df_all['l_1stIn']).fillna(0).astype(np.float32)


print(f"Adım 3/7: Satır satır özellik mühendisliği başlatılıyor ({len(df_all)} maç işlenecek)...")

# ===============================================
# B) SATIR SATIR ÖZELLİK HESAPLAMA (ELO, H2H, MOMENTUM)
# ===============================================

# --- 1. Elo K-Faktörü Fonksiyonu (Sizin Gönderdiğiniz Resimden) ---
def get_elo_k_factor(matches_played):
    """ Dinamik K-Faktörü: Oyuncu acemiyken (az maç) hızlı, ustalaştıkça yavaş öğrenir. """
    return 250 / ((matches_played + 5) ** 0.4)

# --- 2. Hafıza (Bellek) Sözlüklerini Başlat ---
elo_ratings = defaultdict(lambda: 1500) # Tüm oyuncular 1500 Elo ile başlar
player_stats = defaultdict(lambda: {
    'matches_played': 0,
    'wins': 0,
    'surface_matches': defaultdict(int),
    'surface_wins': defaultdict(int),
    'last_10_matches_won': deque(maxlen=10), # Momentum (Galibiyet)
    'last_10_matches_ace_rate': deque(maxlen=10), # Momentum (Servis)
    'last_7_days_minutes': deque() # Yorgunluk (Zaman damgası, dakika)
})
# H2H (anahtar: (düşük_id, yüksek_id))
h2h_stats = defaultdict(lambda: {'p1_wins': 0, 'total_matches': 0})

# --- 3. Sonuçları (Özellikleri) Kaydetmek İçin Listeler Oluştur ---
N = len(df_all)
# Elo
w_elo_pre_match, l_elo_pre_match = np.full(N, 1500.0), np.full(N, 1500.0)
# Kariyer
w_career_win_rate, l_career_win_rate = np.full(N, 0.5), np.full(N, 0.5)
w_career_ace_rate, l_career_ace_rate = np.full(N, 0.05), np.full(N, 0.05)
# Zemin
w_surface_win_rate, l_surface_win_rate = np.full(N, 0.5), np.full(N, 0.5)
# Momentum (Form)
w_momentum_win, l_momentum_win = np.full(N, 0.5), np.full(N, 0.5)
w_momentum_ace, l_momentum_ace = np.full(N, 0.05), np.full(N, 0.05)
# Yorgunluk (Fatigue)
w_fatigue_7d_mins, l_fatigue_7d_mins = np.full(N, 0.0), np.full(N, 0.0)
# H2H (Head-to-Head)
w_h2h_win_rate, l_h2h_win_rate = np.full(N, 0.5), np.full(N, 0.5)
h2h_matches_played = np.full(N, 0)

# --- 4. Ana Döngü (Satır Satır İşleme) ---
# .itertuples() en hızlı döngü yöntemidir
for row in df_all.itertuples(index=True):
    
    i = row.Index # Satır numarasını (index'i) al
    
    if i % 10000 == 0 and i > 0:
        print(f"   ...{i} maç işlendi...")
        
    # --- A) Maç Verilerini Al ---
    w_id = row.winner_id
    l_id = row.loser_id
    surface = row.surface
    current_date = row.tourney_date
    match_minutes = row.minutes
    
    w_stats = player_stats[w_id]
    l_stats = player_stats[l_id]
    
    # --- B) ÖZELLİKLERİ ÇEK (MAÇ ÖNCESİ) ---
    
    # 1. Elo
    w_elo_pre_match[i] = elo_ratings[w_id]
    l_elo_pre_match[i] = elo_ratings[l_id]

    # 2. Kariyer
    w_matches_played = w_stats['matches_played']
    l_matches_played = l_stats['matches_played']
    
    if w_matches_played > 0:
        w_career_win_rate[i] = w_stats['wins'] / w_matches_played
        w_career_ace_rate[i] = np.mean(w_stats['last_10_matches_ace_rate']) if w_stats['last_10_matches_ace_rate'] else 0.05
    if l_matches_played > 0:
        l_career_win_rate[i] = l_stats['wins'] / l_matches_played
        l_career_ace_rate[i] = np.mean(l_stats['last_10_matches_ace_rate']) if l_stats['last_10_matches_ace_rate'] else 0.05

    # 3. Zemin
    if w_stats['surface_matches'][surface] > 0:
        w_surface_win_rate[i] = w_stats['surface_wins'][surface] / w_stats['surface_matches'][surface]
    if l_stats['surface_matches'][surface] > 0:
        l_surface_win_rate[i] = l_stats['surface_wins'][surface] / l_stats['surface_matches'][surface]
        
    # 4. Momentum (Form)
    if len(w_stats['last_10_matches_won']) > 0:
        w_momentum_win[i] = np.mean(w_stats['last_10_matches_won'])
    if len(l_stats['last_10_matches_won']) > 0:
        l_momentum_win[i] = np.mean(l_stats['last_10_matches_won'])

    # 5. Yorgunluk
    w_total_mins = 0
    for date, mins in list(w_stats['last_7_days_minutes']):
        if (current_date - date).days <= 7:
            w_total_mins += mins
        else:
            w_stats['last_7_days_minutes'].popleft() 
    w_fatigue_7d_mins[i] = w_total_mins

    l_total_mins = 0
    for date, mins in list(l_stats['last_7_days_minutes']):
        if (current_date - date).days <= 7:
            l_total_mins += mins
        else:
            l_stats['last_7_days_minutes'].popleft()
    l_fatigue_7d_mins[i] = l_total_mins

    # 6. H2H
    h2h_key = tuple(sorted((w_id, l_id)))
    h2h_data = h2h_stats[h2h_key]
    p1_in_key = h2h_key[0] # Sözlükte 'p1' olarak saklanan oyuncu
    
    h2h_matches_played[i] = h2h_data['total_matches']
    if h2h_data['total_matches'] > 0:
        p1_win_rate_in_key = h2h_data['p1_wins'] / h2h_data['total_matches']
        w_h2h_win_rate[i] = p1_win_rate_in_key if w_id == p1_in_key else (1.0 - p1_win_rate_in_key)
        l_h2h_win_rate[i] = 1.0 - w_h2h_win_rate[i]

    # --- C) HAFIZAYI GÜNCELLE (MAÇ SONRASI) ---
    
    # 1. Elo
    E_w = 1 / (1 + 10**((l_elo_pre_match[i] - w_elo_pre_match[i]) / 400))
    E_l = 1 - E_w
    # Yeni K-Faktörü (dinamik)
    K_w = get_elo_k_factor(w_matches_played)
    K_l = get_elo_k_factor(l_matches_played)
    
    R_w_new = w_elo_pre_match[i] + K_w * (1 - E_w) # S=1 (Kazandı)
    R_l_new = l_elo_pre_match[i] + K_l * (0 - E_l) # S=0 (Kaybetti)
    
    elo_ratings[w_id] = R_w_new
    elo_ratings[l_id] = R_l_new

    # 2. Kazanan
    w_stats['matches_played'] += 1
    w_stats['wins'] += 1
    w_stats['surface_matches'][surface] += 1
    w_stats['surface_wins'][surface] += 1
    w_stats['last_10_matches_won'].append(1)
    w_stats['last_10_matches_ace_rate'].append(row.w_ace_rate)
    if match_minutes > 0: w_stats['last_7_days_minutes'].append((current_date, match_minutes))
    
    # 3. Kaybeden
    l_stats['matches_played'] += 1
    l_stats['surface_matches'][surface] += 1
    l_stats['last_10_matches_won'].append(0)
    l_stats['last_10_matches_ace_rate'].append(row.l_ace_rate)
    if match_minutes > 0: l_stats['last_7_days_minutes'].append((current_date, match_minutes))
    
    # 4. H2H
    h2h_data['total_matches'] += 1
    if w_id == p1_in_key: # Eğer kazanan, sözlükteki 'p1' ise
        h2h_data['p1_wins'] += 1

print(f"   ...{N} maç işlendi. Döngü tamamlandı.")

# --- D) Hesaplanan Özellikleri Ana DataFrame'e Ekle ---
print("Adım 4/7: Hesaplanan özellikler ana tabloya ekleniyor...")
df_all['w_elo_pre_match'] = w_elo_pre_match
df_all['l_elo_pre_match'] = l_elo_pre_match
df_all['w_career_win_rate'] = w_career_win_rate
df_all['l_career_win_rate'] = l_career_win_rate
df_all['w_career_ace_rate'] = w_career_ace_rate
df_all['l_career_ace_rate'] = l_career_ace_rate
df_all['w_surface_win_rate'] = w_surface_win_rate
df_all['l_surface_win_rate'] = l_surface_win_rate
df_all['w_momentum_win'] = w_momentum_win
df_all['l_momentum_win'] = l_momentum_win
df_all['w_momentum_ace'] = w_momentum_ace
df_all['l_momentum_ace'] = l_momentum_ace
df_all['w_fatigue_7d_mins'] = w_fatigue_7d_mins
df_all['l_fatigue_7d_mins'] = l_fatigue_7d_mins
df_all['w_h2h_win_rate'] = w_h2h_win_rate
df_all['l_h2h_win_rate'] = l_h2h_win_rate
df_all['h2h_matches_played'] = h2h_matches_played

del elo_ratings, player_stats, h2h_stats
gc.collect()

print("Adım 5/7: Veri anonimleştiriliyor (p1/p2 atanıyor)...")
# ===============================================
# C) ANONİMLEŞTİRME VE SON HAZIRLIK
# ===============================================

df_final = df_all.copy()
df_final['dataset_marker'] = df_all['dataset_marker'] 

N = len(df_final)
df_final['p1_id'] = np.where(np.random.rand(N) > 0.5, df_final['winner_id'], df_final['loser_id'])
df_final['p2_id'] = np.where(df_final['p1_id'] == df_final['winner_id'], df_final['loser_id'], df_final['winner_id'])
df_final['Y'] = (df_final['p1_id'] == df_final['winner_id']).astype(np.int8)

print("   ... p1/p2 özellikleri atanıyor ...")
features_to_map = [
    'rank', 'age', 'ht', 'hand', 
    'elo_pre_match', 'career_win_rate', 'career_ace_rate', 
    'surface_win_rate', 'momentum_win', 'momentum_ace', 
    'fatigue_7d_mins'
]

for col in features_to_map:
    # p1'e ata
    df_final[f'p1_{col}'] = np.where(df_final['p1_id'] == df_final['winner_id'], 
                                    df_final.get(f'w_{col}'), 
                                    df_final.get(f'l_{col}'))
    # p2'ye ata
    df_final[f'p2_{col}'] = np.where(df_final['p2_id'] == df_final['winner_id'], 
                                    df_final.get(f'w_{col}'), 
                                    df_final.get(f'l_{col}'))

# H2H (p1'in p2'ye karşı oranı)
df_final['p1_h2h_rate'] = np.where(df_final['p1_id'] == df_final['winner_id'], df_final['w_h2h_win_rate'], df_final['l_h2h_win_rate'])
df_final['p2_h2h_rate'] = 1.0 - df_final['p1_h2h_rate']
df_final['h2h_matches_played'] = df_all['h2h_matches_played']

# Fark (Difference) Özellikleri
df_final['p1_rank_diff'] = df_final['p1_rank'] - df_final['p2_rank']
df_final['p1_elo_diff'] = df_final['p1_elo_pre_match'] - df_final['p2_elo_pre_match']

print("Adım 6/7: Son özellik mühendisliği (One-Hot) ve Doldurma...")
df_final['p1_is_L'] = (df_final['p1_hand'] == 'L').astype(np.int8)
df_final['p2_is_L'] = (df_final['p2_hand'] == 'L').astype(np.int8)
df_final['surface'] = df_final['surface'].astype(str).replace('Carpet', 'Hard').fillna('U')
df_final = pd.get_dummies(df_final, columns=['surface'], prefix='surface', dtype=np.int8)

# --- NİHAİ ÖZELLİK LİSTESİ (FINAL_FEATURES) ---
base_features = ['p1_age', 'p1_ht', 'p1_rank', 'p1_is_L', 'p1_rank_diff',
                 'p2_age', 'p2_ht', 'p2_rank', 'p2_is_L']
elo_features = ['p1_elo_pre_match', 'p2_elo_pre_match', 'p1_elo_diff']
stat_features = [
    'p1_career_win_rate', 'p1_surface_win_rate', 'p1_momentum_win', 'p1_h2h_rate', 'p1_career_ace_rate', 'p1_momentum_ace', 'p1_fatigue_7d_mins',
    'p2_career_win_rate', 'p2_surface_win_rate', 'p2_momentum_win', 'p2_h2h_rate', 'p2_career_ace_rate', 'p2_momentum_ace', 'p2_fatigue_7d_mins'
]
meta_features = ['h2h_matches_played']
surface_features = [col for col in df_final.columns if col.startswith('surface_')]
FINAL_FEATURES = base_features + elo_features + stat_features + meta_features + surface_features

# Eksik Veri Doldurma (Imputation)
# (NaN'ler sadece ilk maçlarda vs. oluşmalı)
# 'rate' veya 'momentum' içerenleri 0.5 (nötr) ile doldur
rate_cols = [col for col in FINAL_FEATURES if 'rate' in col or 'h2h' in col or 'momentum' in col]
df_final[rate_cols] = df_final[rate_cols].fillna(0.5) 
# Kalanlar (age, ht, rank, fatigue, vb.)
other_cols = [col for col in FINAL_FEATURES if col not in rate_cols]
df_final[other_cols] = df_final[other_cols].fillna(0) # Nötr (0)

# Rank için 500 (en kötü) gibi bir ceza
df_final['p1_rank'].replace(0, 500, inplace=True)
df_final['p2_rank'].replace(0, 500, inplace=True)
df_final['p1_rank_diff'].replace(0, 500, inplace=True) # Eğer p2 0 (NaN) ise fark da NaN olur
df_final['p1_ht'].replace(0, 180, inplace=True) # Ortalama boy tahmini
df_final['p2_ht'].replace(0, 180, inplace=True)

# --- Veri Setlerini Ayır, Normalize et ve Numpy'a Çevir ---
df_train_processed = df_final[df_final['dataset_marker'] == 'train']
df_val_processed = df_final[df_final['dataset_marker'] == 'val']
df_test_processed = df_final[df_final['dataset_marker'] == 'test']

Y_train = df_train_processed['Y'].values.reshape(1, -1)
Y_val = df_val_processed['Y'].values.reshape(1, -1)
Y_test = df_test_processed['Y'].values.reshape(1, -1)

X_train_df = df_train_processed[FINAL_FEATURES]
X_val_df = df_val_processed[FINAL_FEATURES]
X_test_df = df_test_processed[FINAL_FEATURES]

# Normalleştirme (StandardScaler en iyisidir, ama uyumluluk için MaxAbsScaler)
print("   ... Veri normalize ediliyor (MaxAbsScaler) ...")
X_train_np = X_train_df.T.values.astype(np.float64)
# MaxAbsScaler: Her özelliği [-1, 1] arasına sıkıştırır
X_max_per_feature = np.max(np.abs(X_train_np), axis=1, keepdims=True)
X_max_per_feature[X_max_per_feature == 0] = 1.0

X_train = X_train_np / X_max_per_feature
X_val = X_val_df.T.values.astype(np.float64) / X_max_per_feature
X_test = X_test_df.T.values.astype(np.float64) / X_max_per_feature

# 'Aşama 2' setlerini de oluştur
X_final_train = np.concatenate((X_train, X_val), axis=1)
Y_final_train = np.concatenate((Y_train, Y_val), axis=1)

print(f"Adım 7/7: Tüm veri setleri '{output_npz_file}' dosyasına kaydediliyor...")
np.savez_compressed(
    output_npz_file,
    X_train=X_train,
    Y_train=Y_train,
    X_val=X_val,
    Y_val=Y_val,
    X_test=X_test,
    Y_test=Y_test,
    X_final_train=X_final_train,
    Y_final_train=Y_final_train,
    FINAL_FEATURES=np.array(FINAL_FEATURES)
)

print("\n--- İŞLEM TAMAMLANDI ---")
print(f"Tüm veriler (Elo, H2H, Momentum dahil) '{output_npz_file}' dosyasına kaydedildi.")
print(f"Yeni Özellik Sayısı (N_X): {X_train.shape[0]}")
print(f"Yeni Özellikler ({len(FINAL_FEATURES)} adet):")
print(FINAL_FEATURES)