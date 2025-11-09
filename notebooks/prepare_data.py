import numpy as np
import pandas as pd
import glob
from warnings import filterwarnings
from collections import defaultdict, deque
import gc

print("--- 'Premium Yakıt' Üretim Script'i Başlatıldı (H2H ve Momentum ile) ---")
print("--- UYARI: Bu işlem RAM kullanımı düşük ancak ÇOK YAVAŞ olacaktır (30+ dk). ---")
filterwarnings('ignore')

# ===============================================
# A) VERİ YÜKLEME VE HAZIRLAMA (ÖN İŞLEME)
# ===============================================

path = '../datas/' 

# 1. Dosya Listelerini Oluşturma (Sizin ayarınız: Val=2023)
training_years = [str(y) for y in range(1968, 2023)] 
validation_years = [str(y) for y in range(2023, 2024)]
test_file = path + "atp_matches_2024.csv" 

training_files = [path + f"atp_matches_{y}.csv" for y in training_years]
validation_files = [path + f"atp_matches_{y}.csv" for y in validation_years]

# 3. KULLANILACAK HAM SÜTUNLAR
RAW_FEATURES = [
    'tourney_id', 'tourney_name', 'surface', 'tourney_level', 'tourney_date', 'match_num',
    'winner_id', 'winner_hand', 'winner_ht', 'winner_age', 
    'loser_id', 'loser_hand', 'loser_ht', 'loser_age',
    'winner_rank_points', 'loser_rank_points',
    'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
    'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced'
]

# 2. Veri Yükleme Fonksiyonu
def load_and_concatenate(file_list, file_type='train'):
    # (Bu fonksiyon değişmedi)
    li = []
    for filename in file_list:
        try:
            df_temp = pd.read_csv(filename, index_col=None, header=0, usecols=lambda c: c in RAW_FEATURES or c.startswith('atp_matches'))
            df_temp['dataset_marker'] = file_type
            li.append(df_temp)
        except Exception as e:
            continue
    if li:
        return pd.concat(li, axis=0, ignore_index=True)
    return pd.DataFrame() 

# 5. YENİ İŞ AKIŞI: VERİ HAZIRLAMA PİPELINE'I
print("Adım 1/6: Tüm veri setleri yükleniyor (1968-2024)...")
df_train = load_and_concatenate(training_files, 'train')
df_val = load_and_concatenate(validation_files, 'val')
try:
    df_test = pd.read_csv(test_file, usecols=lambda c: c in RAW_FEATURES)
    df_test['dataset_marker'] = 'test'
except:
    df_test = pd.DataFrame()

df_all = pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True)
df_all['tourney_date'] = pd.to_datetime(df_all['tourney_date'], format='%Y%m%d')
df_all.sort_values(by=['tourney_date', 'match_num'], inplace=True)
df_all.reset_index(drop=True, inplace=True)
df_all['match_id'] = df_all.index.astype(np.int32)

# --- RAM Optimizasyonu (Erken) ---
df_all['surface'] = df_all['surface'].astype('category')
df_all['winner_id'] = df_all['winner_id'].astype(np.int32)
df_all['loser_id'] = df_all['loser_id'].astype(np.int32)

print(f"Adım 2/6: Satır satır özellik mühendisliği başlatılıyor ({len(df_all)} maç işlenecek)...")

# --- YENİ YÖNTEM: Satır Satır İstatistik Hesaplama ---

# 1. Hafıza (Bellek) Sözlüklerini Başlat
#    Bu sözlükler, her oyuncunun o ana kadarki istatistiklerini tutar.
player_stats = defaultdict(lambda: {
    'matches_played': 0,
    'wins': 0,
    'ace_total': 0,
    'svpt_total': 0,
    'surface_matches': defaultdict(int),
    'surface_wins': defaultdict(int),
    'last_10_matches': deque(maxlen=10) # Momentum için (son 10 maçın sonucu)
})

# H2H (Head-to-Head) istatistiklerini tutmak için ayrı bir sözlük
# Anahtar: (player1_id, player2_id) -> Değer: {'p1_wins': 2, 'total_matches': 3}
h2h_stats = defaultdict(lambda: {'p1_wins': 0, 'total_matches': 0})

# 2. Sonuçları (Özellikleri) Kaydetmek İçin Listeler Oluştur
#    Bu listeler, her maçın *o anki* (maçtan önceki) istatistiklerini tutar.
N = len(df_all)
w_career_win_rate = np.full(N, 0.5) # (0.5 = Nötr başlangıç)
l_career_win_rate = np.full(N, 0.5)
w_surface_win_rate = np.full(N, 0.5)
l_surface_win_rate = np.full(N, 0.5)
w_momentum_10 = np.full(N, 0.5)
l_momentum_10 = np.full(N, 0.5)
w_h2h_rate = np.full(N, 0.5)
l_h2h_rate = np.full(N, 0.5)
w_career_ace_rate = np.full(N, 0.05) # (0.05 = Nötr ace oranı)
l_career_ace_rate = np.full(N, 0.05)

# 3. Ana Döngü (Satır Satır İşleme)
# (Bu döngü yavaş olacak)
for i, row in df_all.iterrows():
    
    if i % 10000 == 0 and i > 0:
        print(f"   ...{i} maç işlendi...")
        
    w_id = row['winner_id']
    l_id = row['loser_id']
    surface = row['surface']
    
    # --- A) VERİLERİ ÇEK (Sızıntı Önleme) ---
    # Modelin göreceği veriler, bu maç *oynanmadan önceki* verilerdir.
    
    # 1. Kazanan (Winner) için istatistikleri çek
    w_stats = player_stats[w_id]
    w_matches = w_stats['matches_played']
    w_surf_matches = w_stats['surface_matches'][surface]
    
    if w_matches > 0:
        w_career_win_rate[i] = w_stats['wins'] / w_matches
        w_career_ace_rate[i] = w_stats['ace_total'] / w_stats['svpt_total'] if w_stats['svpt_total'] > 0 else 0.05
    if w_surf_matches > 0:
        w_surface_win_rate[i] = w_stats['surface_wins'][surface] / w_surf_matches
    if len(w_stats['last_10_matches']) > 0:
        w_momentum_10[i] = np.mean(w_stats['last_10_matches'])

    # 2. Kaybeden (Loser) için istatistikleri çek
    l_stats = player_stats[l_id]
    l_matches = l_stats['matches_played']
    l_surf_matches = l_stats['surface_matches'][surface]
    
    if l_matches > 0:
        l_career_win_rate[i] = l_stats['wins'] / l_matches
        l_career_ace_rate[i] = l_stats['ace_total'] / l_stats['svpt_total'] if l_stats['svpt_total'] > 0 else 0.05
    if l_surf_matches > 0:
        l_surface_win_rate[i] = l_stats['surface_wins'][surface] / l_surf_matches
    if len(l_stats['last_10_matches']) > 0:
        l_momentum_10[i] = np.mean(l_stats['last_10_matches'])
        
    # 3. H2H istatistiklerini çek
    # (w_id, l_id) sırası önemlidir
    h2h_key = tuple(sorted((w_id, l_id))) # Anahtarı her zaman (düşük_id, yüksek_id) yap
    h2h_data = h2h_stats[h2h_key]
    
    if h2h_data['total_matches'] > 0:
        # H2H oranını *her iki oyuncunun bakış açısıyla* hesapla
        if w_id < l_id: # w_id, anahtardaki p1 ise
            w_h2h_rate[i] = h2h_data['p1_wins'] / h2h_data['total_matches']
            l_h2h_rate[i] = 1.0 - w_h2h_rate[i]
        else: # w_id, anahtardaki p2 ise (l_id = p1)
            l_h2h_rate[i] = h2h_data['p1_wins'] / h2h_data['total_matches']
            w_h2h_rate[i] = 1.0 - l_h2h_rate[i]

    # --- B) HAFIZAYI GÜNCELLE (Bu maçın sonuçlarını ekle) ---
    # (Bu kısım model tarafından asla görülmez, sadece bir sonraki maç için kullanılır)
    
    # 1. Kazanan (Winner) istatistiklerini güncelle
    w_stats['matches_played'] += 1
    w_stats['wins'] += 1
    w_stats['ace_total'] += row['w_ace']
    w_stats['svpt_total'] += row['w_svpt']
    w_stats['surface_matches'][surface] += 1
    w_stats['surface_wins'][surface] += 1
    w_stats['last_10_matches'].append(1) # 1 = kazandı
    
    # 2. Kaybeden (Loser) istatistiklerini güncelle
    l_stats['matches_played'] += 1
    l_stats['ace_total'] += row['l_ace']
    l_stats['svpt_total'] += row['l_svpt']
    l_stats['surface_matches'][surface] += 1
    l_stats['last_10_matches'].append(0) # 0 = kaybetti
    
    # 3. H2H istatistiklerini güncelle
    h2h_data['total_matches'] += 1
    if w_id < l_id: # w_id, anahtardaki p1 ise
        h2h_data['p1_wins'] += 1
    
print(f"   ...{N} maç işlendi. Döngü tamamlandı.")

# --- C) Hesaplanan Özellikleri Ana DataFrame'e Ekle ---
print("Adım 3/6: Hesaplanan özellikler ana tabloya ekleniyor...")

df_all['w_career_win_rate'] = w_career_win_rate
df_all['l_career_win_rate'] = l_career_win_rate
df_all['w_surface_win_rate'] = w_surface_win_rate
df_all['l_surface_win_rate'] = l_surface_win_rate
df_all['w_momentum_10'] = w_momentum_10
df_all['l_momentum_10'] = l_momentum_10
df_all['w_h2h_rate'] = w_h2h_rate
df_all['l_h2h_rate'] = l_h2h_rate
df_all['w_career_ace_rate'] = w_career_ace_rate
df_all['l_career_ace_rate'] = l_career_ace_rate

# RAM Boşaltma
del player_stats, h2h_stats
gc.collect()

print("Adım 4/6: Veri anonimleştiriliyor (p1/p2 atanıyor)...")
# --- Anonimleştirme ---
# (Bu, artık 'Melt'e gerek duymadan doğrudan df_all üzerinde yapılır)

df_final = df_all.copy()
df_final['p1_id'] = np.where(np.random.rand(N) > 0.5, df_final['winner_id'], df_final['loser_id'])
df_final['p2_id'] = np.where(df_final['p1_id'] == df_final['winner_id'], df_final['loser_id'], df_final['winner_id'])
df_final['Y'] = (df_final['p1_id'] == df_final['winner_id']).astype(np.int8)

# --- Özellikleri p1/p2'ye ata ---
# p1 özellikleri
df_final['p1_rank_points'] = np.where(df_final['p1_id'] == df_final['winner_id'], df_final['winner_rank_points'], df_final['loser_rank_points'])
df_final['p1_age'] = np.where(df_final['p1_id'] == df_final['winner_id'], df_final['winner_age'], df_final['loser_age'])
df_final['p1_ht'] = np.where(df_final['p1_id'] == df_final['winner_id'], df_final['winner_ht'], df_final['loser_ht'])
df_final['p1_hand'] = np.where(df_final['p1_id'] == df_final['winner_id'], df_final['winner_hand'], df_final['loser_hand'])
df_final['p1_career_win_rate'] = np.where(df_final['p1_id'] == df_final['winner_id'], df_final['w_career_win_rate'], df_final['l_career_win_rate'])
df_final['p1_surface_win_rate'] = np.where(df_final['p1_id'] == df_final['winner_id'], df_final['w_surface_win_rate'], df_final['l_surface_win_rate'])
df_final['p1_momentum_10'] = np.where(df_final['p1_id'] == df_final['winner_id'], df_final['w_momentum_10'], df_final['l_momentum_10'])
df_final['p1_h2h_rate'] = np.where(df_final['p1_id'] == df_final['winner_id'], df_final['w_h2h_rate'], df_final['l_h2h_rate'])
df_final['p1_career_ace_rate'] = np.where(df_final['p1_id'] == df_final['winner_id'], df_final['w_career_ace_rate'], df_final['l_career_ace_rate'])

# p2 özellikleri
df_final['p2_rank_points'] = np.where(df_final['p2_id'] == df_final['winner_id'], df_final['winner_rank_points'], df_final['loser_rank_points'])
df_final['p2_age'] = np.where(df_final['p2_id'] == df_final['winner_id'], df_final['winner_age'], df_final['loser_age'])
df_final['p2_ht'] = np.where(df_final['p2_id'] == df_final['winner_id'], df_final['winner_ht'], df_final['loser_ht'])
df_final['p2_hand'] = np.where(df_final['p2_id'] == df_final['winner_id'], df_final['winner_hand'], df_final['loser_hand'])
df_final['p2_career_win_rate'] = np.where(df_final['p2_id'] == df_final['winner_id'], df_final['w_career_win_rate'], df_final['l_career_win_rate'])
df_final['p2_surface_win_rate'] = np.where(df_final['p2_id'] == df_final['winner_id'], df_final['w_surface_win_rate'], df_final['l_surface_win_rate'])
df_final['p2_momentum_10'] = np.where(df_final['p2_id'] == df_final['winner_id'], df_final['w_momentum_10'], df_final['l_momentum_10'])
df_final['p2_h2h_rate'] = np.where(df_final['p2_id'] == df_final['winner_id'], df_final['w_h2h_rate'], df_final['l_h2h_rate'])
df_final['p2_career_ace_rate'] = np.where(df_final['p2_id'] == df_final['winner_id'], df_final['w_career_ace_rate'], df_final['l_career_ace_rate'])

print("Adım 5/6: Son özellik mühendisliği yapılıyor (One-Hot)...")
# --- Son Hazırlık ---
df_final['p1_is_L'] = (df_final['p1_hand'] == 'L').astype(np.int8)
df_final['p2_is_L'] = (df_final['p2_hand'] == 'L').astype(np.int8)
df_final['surface'] = df_final['surface'].astype(str).replace('Carpet', 'Hard').fillna('U') # Kategori'den str'ye
df_final = pd.get_dummies(df_final, columns=['surface'], prefix='surface', dtype=np.int8)

# --- NİHAİ ÖZELLİK LİSTESİ (FINAL_FEATURES) ---
base_features = ['p1_age', 'p1_ht', 'p1_rank_points', 'p1_is_L', 
                 'p2_age', 'p2_ht', 'p2_rank_points', 'p2_is_L']
# YENİ EKLENEN ÖZELLİKLER
stat_features = [
    'p1_career_win_rate', 'p1_surface_win_rate', 'p1_momentum_10', 'p1_h2h_rate', 'p1_career_ace_rate',
    'p2_career_win_rate', 'p2_surface_win_rate', 'p2_momentum_10', 'p2_h2h_rate', 'p2_career_ace_rate'
]
surface_features = [col for col in df_final.columns if col.startswith('surface_')]
FINAL_FEATURES = base_features + stat_features + surface_features

# Eksik Veri Doldurma (Imputation)
# (NaN'ler sadece ilk maçlarda vs. oluşmalı)
rate_cols = [col for col in FINAL_FEATURES if 'rate' in col or 'h2h' in col or 'momentum' in col]
df_final[rate_cols] = df_final[rate_cols].fillna(0.5) # Nötr

other_num_cols = [col for col in FINAL_FEATURES if col not in rate_cols and not col.startswith('surface_')]
# Eğitim setinin ortalamasını bul (dataset_marker ile)
train_means = df_final[df_final['dataset_marker'] == 'train'][other_num_cols].mean()
df_final[other_num_cols] = df_final[other_num_cols].fillna(train_means)
df_final[other_num_cols] = df_final[other_num_cols].fillna(0) # Güvenlik önlemi

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

# Normalleştirme (MaxAbsScaler)
X_train_np = X_train_df.T.values.astype(np.float64)
X_max_per_feature = np.max(np.abs(X_train_np), axis=1, keepdims=True)
X_max_per_feature[X_max_per_feature == 0] = 1.0

X_train = X_train_np / X_max_per_feature
X_val = X_val_df.T.values.astype(np.float64) / X_max_per_feature
X_test = X_test_df.T.values.astype(np.float64) / X_max_per_feature

# 'Aşama 2' setlerini de oluştur
X_final_train = np.concatenate((X_train, X_val), axis=1)
Y_final_train = np.concatenate((Y_train, Y_val), axis=1)

print("Adım 6/6: Tüm veri setleri 'processed_data_H2H_Momentum.npz' dosyasına kaydediliyor...")
# --- Tüm NumPy dizilerini tek bir sıkıştırılmış dosyaya kaydet ---
np.savez_compressed(
    'processed_data_H2H_Momentum.npz', # YENİ DOSYA ADI
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
print("Tüm veriler (H2H ve Momentum dahil) 'processed_data_H2H_Momentum.npz' dosyasına kaydedildi.")
print(f"Yeni Özellik Sayısı (N_X): {X_train.shape[0]}")
print(f"Yeni Özellikler: {FINAL_FEATURES}")