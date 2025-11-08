import numpy as np
import pandas as pd
import glob

# ===============================================
# A) VERİ YÜKLEME VE HAZIRLAMA (YENİ ÖZELLİKLERLE)
# ===============================================

path = '../datas/' # Lütfen yerel path'inizin doğru olduğundan emin olun!

# 1. Dosya Listelerini Oluşturma (Zamana Göre Ayrım)
training_years = [str(y) for y in range(1968, 2022)] # 1968-2021
validation_years = [str(y) for y in range(2022, 2024)] # 2022-2023
test_file = path + "atp_matches_2024.csv" 

training_files = [path + f"atp_matches_{y}.csv" for y in training_years]
validation_files = [path + f"atp_matches_{y}.csv" for y in validation_years]

# 2. Veri Yükleme Fonksiyonu
def load_and_concatenate(file_list):
    li = []
    for filename in file_list:
        try:
            # ÖNEMLİ: Sadece gerekli sütunları yükleyerek hafızayı verimli kullan
            # (winner_name, score vb. gereksiz sütunları baştan atar)
            df_temp = pd.read_csv(filename, index_col=None, header=0, usecols=RAW_FEATURES)
            li.append(df_temp)
        except Exception as e:
            # print(f"Uyarı: {filename} yüklenemedi veya sütunlar eksik. Hata: {e}")
            continue
    if li:
        return pd.concat(li, axis=0, ignore_index=True)
    return pd.DataFrame() 

# 3. KULLANILACAK HAM SÜTUNLAR (GÜNCELLENDİ)
# Modelin ihtiyaç duyduğu tüm ham veriler
RAW_FEATURES = [
    'winner_id', 'loser_id', 
    'winner_rank_points', 'loser_rank_points', 
    'winner_age', 'loser_age', 
    'winner_ht', 'loser_ht',
    'surface',          # YENİ EKLENDİ (Maç zemini)
    'winner_hand',      # YENİ EKLENDİ (Oyuncu eli)
    'loser_hand'        # YENİ EKLENDİ (Oyuncu eli)
]

# 4. MODELİN GÖRECEĞİ NİHAİ ÖZELLİKLER (Dinamikleşti)
# Bu liste artık global değil, veri işleme sırasında (6.5 bölümü) oluşturulacak.


# 5. KRİTİK FONKSİYON: ANONİMLEŞTİRME (Yeniden Düzenlendi)
def create_anonymous_dataframe(df):
    """
    Bu fonksiyon, ham veri setini alır, eksik verileri atar, 
    p1/p2 ataması yapar ve yarı işlenmiş bir DataFrame döndürür.
    One-hot encoding bu aşamada YAPILMAZ.
    """
    
    # 1. Sadece gerekli ham sütunları tut ve EKSİK VERİLERİ AT
    # (Eğer 'surface', 'ht' veya 'hand' bilgisi yoksa o maçı kullanamayız)
    df_clean = df[RAW_FEATURES].copy().dropna(subset=RAW_FEATURES)
    
    # 2. p1 ve p2'yi rastgele ata (Ezberlemeyi Önler)
    df_clean['p1_id'] = np.where(np.random.rand(len(df_clean)) > 0.5, df_clean['winner_id'], df_clean['loser_id'])
    df_clean['p2_id'] = np.where(df_clean['p1_id'] == df_clean['winner_id'], df_clean['loser_id'], df_clean['winner_id'])

    # 3. Anonim İstatistikleri Oluşturma (p1/p2)
    
    # Rank Points
    df_clean['p1_rank_points'] = np.where(df_clean['p1_id'] == df_clean['winner_id'], df_clean['winner_rank_points'], df_clean['loser_rank_points'])
    df_clean['p2_rank_points'] = np.where(df_clean['p1_id'] == df_clean['winner_id'], df_clean['loser_rank_points'], df_clean['winner_rank_points'])
    
    # Age
    df_clean['p1_age'] = np.where(df_clean['p1_id'] == df_clean['winner_id'], df_clean['winner_age'], df_clean['loser_age'])
    df_clean['p2_age'] = np.where(df_clean['p1_id'] == df_clean['winner_id'], df_clean['loser_age'], df_clean['winner_age'])
    
    # Height (Boy)
    df_clean['p1_ht'] = np.where(df_clean['p1_id'] == df_clean['winner_id'], df_clean['winner_ht'], df_clean['loser_ht'])
    df_clean['p2_ht'] = np.where(df_clean['p1_id'] == df_clean['winner_id'], df_clean['loser_ht'], df_clean['winner_ht'])
    
    # Hand (El) - YENİ EKLENDİ (Henüz 1/0 değil, 'R'/'L'/'U' olarak)
    df_clean['p1_hand'] = np.where(df_clean['p1_id'] == df_clean['winner_id'], df_clean['winner_hand'], df_clean['loser_hand'])
    df_clean['p2_hand'] = np.where(df_clean['p1_id'] == df_clean['winner_id'], df_clean['loser_hand'], df_clean['winner_hand'])

    # Surface (Zemin) - Bu özellik maça aittir, p1/p2'ye değil, o yüzden olduğu gibi kalır.
    # 'surface' sütunu zaten df_clean içinde mevcut.

    # 4. Etiket (Y) oluşturma: p1_wins = 1 (p1 kazandıysa), 0 (p2 kazandıysa)
    Y_df = (df_clean['winner_id'] == df_clean['p1_id']).astype(int)
    
    # 5. Sonraki adıma (one-hot encoding) hazır özellikleri seç
    features_to_process = [
        'p1_rank_points', 'p2_rank_points', 
        'p1_age', 'p2_age', 
        'p1_ht', 'p2_ht',
        'p1_hand', 'p2_hand',   # Kategorik
        'surface'               # Kategorik
    ]
    
    X_df = df_clean[features_to_process]
    
    return X_df, Y_df


# 6. Veri Kümelerini Yükleme ve Anonimleştirme
print("Ham veri dosyaları yükleniyor...")
df_train_full = load_and_concatenate(training_files)
df_val_full = load_and_concatenate(validation_files)
try:
    df_test = pd.read_csv(test_file, usecols=RAW_FEATURES)
except:
    df_test = pd.DataFrame()

print("Veri anonimleştiriliyor...")
X_train_df, Y_train_df = create_anonymous_dataframe(df_train_full)
X_val_df, Y_val_df = create_anonymous_dataframe(df_val_full)
X_test_df, Y_test_df = create_anonymous_dataframe(df_test) # Düzeltme: df_test olmalı

# Y etiketlerini Numpy formatına çevir
Y_train = Y_train_df.values.reshape(1, -1)
Y_val = Y_val_df.values.reshape(1, -1)
Y_test = Y_test_df.values.reshape(1, -1)


# 6.5 (YENİ BÖLÜM) ÖZELLİK MÜHENDİSLİĞİ (ONE-HOT ENCODING VE HİZALAMA)
# Train, Val ve Test setlerindeki kategorilerin (örn. surface)
# tutarlı olmasını sağlamak için hepsini birleştir, işlemi yap, ayır.

print("Özellik mühendisliği (One-Hot Encoding) yapılıyor...")

# Veri setlerini tanıyabilmek için etiketle
X_train_df['dataset'] = 'train'
X_val_df['dataset'] = 'val'
X_test_df['dataset'] = 'test'

# Tüm X verilerini birleştir
df_combined = pd.concat([X_train_df, X_val_df, X_test_df], axis=0, ignore_index=True)

# --- Kategorik Özellik İşleme ---

# 1. Hand (El) İşleme
# 'U' (Bilinmeyen) ve NaNs (Eksik) olanları en yaygın olan 'R' (Sağ) ile doldur
df_combined['p1_hand'] = df_combined['p1_hand'].replace('U', 'R').fillna('R')
df_combined['p2_hand'] = df_combined['p2_hand'].replace('U', 'R').fillna('R')

# 'L' (Solak) olup olmadığını 1/0 olarak kodla
df_combined['p1_is_L'] = (df_combined['p1_hand'] == 'L').astype(int)
df_combined['p2_is_L'] = (df_combined['p2_hand'] == 'L').astype(int)

# 2. Surface (Zemin) İşleme
# 'Carpet' (Halı) çok nadirdir, 'Hard' (Sert) zemin gibi kabul edilebilir veya 'U' (Bilinmeyen)
df_combined['surface'] = df_combined['surface'].replace('Carpet', 'Hard').fillna('U')

# One-Hot Encoding: 'surface' sütununu 'surface_Hard', 'surface_Clay' vb. sütunlara ayır.
df_combined = pd.get_dummies(df_combined, columns=['surface'], prefix='surface')


# --- NİHAİ ÖZELLİK LİSTESİNİ OLUŞTURMA ---
# Sayısal (Base) Özellikler
base_features = [
    'p1_rank_points', 'p2_rank_points', 
    'p1_age', 'p2_age', 
    'p1_ht', 'p2_ht'
]

# Yeni oluşturulan özellikler
hand_features = ['p1_is_L', 'p2_is_L']
surface_features = [col for col in df_combined.columns if col.startswith('surface_') and col != 'surface_U']
# Not: surface_U'yu (Bilinmeyen zemin) atıyoruz, çünkü 3 zemin türü de 0 ise, 
# model zaten 'U' olduğunu anlayacaktır (dummy variable trap).

FINAL_FEATURES = base_features + hand_features + surface_features

# 3. Sayısal Verilerde Eksik Veri Doldurma (Imputation)
# (Çok nadir de olsa, rank_points vb. eksikse, ortalama ile doldur)
df_combined[base_features] = df_combined[base_features].fillna(df_combined[base_features].mean())

# --- Veri Setlerini Geri Ayırma ---
df_train_final = df_combined[df_combined['dataset'] == 'train'][FINAL_FEATURES]
df_val_final = df_combined[df_combined['dataset'] == 'val'][FINAL_FEATURES]
df_test_final = df_combined[df_combined['dataset'] == 'test'][FINAL_FEATURES]


# 7. Normalleştirme ve Numpy Dönüşümü
print("Veri normalize ediliyor...")

# Numpy formatına dönüştür (N_X, m)
X_train_np = df_train_final.T.values.astype(np.float64)
X_val_np = df_val_final.T.values.astype(np.float64)
X_test_np = df_test_final.T.values.astype(np.float64)

# Normalleştirme (Sadece Eğitim setinin max değerlerine göre)
X_max_per_feature = np.max(X_train_np, axis=1, keepdims=True)

# 0'a bölme hatasını engelle
X_max_per_feature[X_max_per_feature == 0] = 1.0

X_train = X_train_np / X_max_per_feature
X_val = X_val_np / X_max_per_feature
X_test = X_test_np / X_max_per_feature

# 8. Ana Eğitim Değişkenlerini Güncelleme
X = X_train
Y = Y_train

# Kontrol
print(f"--- Yeni Veri Yapısı (Anonimleştirilmiş ve Genişletilmiş) ---")
print(f"Özellik İsimleri: {FINAL_FEATURES}")
print(f"1. Eğitim Seti Boyutu: {X_train.shape[1]} örnek")
print(f"2. Doğruluk Seti Boyutu: {X_val.shape[1]} örnek")
print(f"3. Test Seti Boyutu: {X_test.shape[1]} örnek")

# ===============================================
# YAPILANDIRMA: HİPER-PARAMETRELER (AYARLANABİLİR KISIM)
# ===============================================
N_X = X.shape[0]        # Girdi Boyutu
N_H = 32                # Gizli Katman Nöron Sayısı (DENEME İÇİN DEĞİŞTİRİLEBİLİR!)
N_Y = Y.shape[0]        # Çıktı Boyutu
LEARNING_RATE = 0.05    # Öğrenme Hızı (DENEME İÇİN DEĞİŞTİRİLEBİLİR!)
EPOCHS = 5000           # Eğitim Döngüsü Sayısı

print(f"Girdi Özellik Sayısı (N_X): {N_X}")

# ===============================================
# A) YARDIMCI FONKSİYONLAR (HELPER FUNCTIONS)
# ===============================================

def sigmoid(Z):
    """Sigmoid aktivasyon fonksiyonunu hesaplar. Çıktıyı 0 ile 1 arasına sıkıştırır."""
    A = 1 / (1 + np.exp(-Z))
    return A

def sigmoid_backward(A):
    """Sigmoid'in türevini (Geriye yayılım için) hesaplar: A * (1 - A)"""
    # Bu türev, hatanın gizli katmandan geriye doğru nasıl yayılacağını belirler.
    dZ = A * (1 - A) 
    return dZ

# ===============================================
# B) MODEL SINIFI VE BAŞLATMA
# ===============================================

def initialize_parameters(n_x, n_h, n_y):
    """
    Ağırlık ve sapmaları "Xavier (Glorot)" yöntemiyle başlatır.
    Bu, tanh aktivasyonu için 'vanishing gradient' sorununu önler.
    """
    
    # Hatalı Yöntem (Gradyanları sıfıra çökertiyor):
    # W1 = np.random.randn(n_h, n_x) * 0.01 
    # W2 = np.random.randn(n_y, n_h) * 0.01
    
    # DOĞRU YÖNTEM (Xavier):
    # Varyansı, giren nöron sayısına bölerek koru.
    
    W1 = np.random.randn(n_h, n_x) * np.sqrt(1 / n_x)
    b1 = np.zeros((n_h, 1))
    
    W2 = np.random.randn(n_y, n_h) * np.sqrt(1 / n_h)
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters


def forward_propagation(X, parameters):
    """Girdiyi alır ve tahmini çıktıyı (A2) hesaplar."""
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # 1. Katman (Gizli Katman)
    # Z1 = W1 * X + b1
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1) 
    
    # 2. Katman (Çıkış Katmanı)
    # Z2 = W2 * A1 + b2
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2) # Tahminimiz (Y_hat)
    
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

# D) MALİYET FONKSİYONU
def compute_cost(A2, Y):
    """
    İkili Çapraz Entropi maliyetini hesaplar.
    
    A2: Ağın tahmini (Y_hat)
    Y: Gerçek Etiketler
    """
    m = Y.shape[1] # Örnek sayısı (sütun sayısı)
    
    # !!! DÜZELTME: np.log(0) hatasını önlemek için kırpma (clipping) !!!
    epsilon = 1e-8 # Çok küçük bir sayı
    A2_clipped = np.clip(A2, epsilon, 1 - epsilon)
    # -------------------------------------------------------------
    
    # Çapraz Entropi formülünün iki ana terimini hesaplayın:
    # 1. terim: Y * log(A2_clipped)
    log_probs = Y * np.log(A2_clipped)
    
    # 2. terim: (1 - Y) * log(1 - A2_clipped)
    log_probs_complement = (1 - Y) * np.log(1 - A2_clipped)
    
    # Toplam Maliyet (Cost) J = -(1/m) * (terim1 + terim2) matrisinin tüm elemanlarının toplamı
    cost = - (1 / m) * (np.sum(log_probs + log_probs_complement))
    
    # Maliyetin tek bir sayı (skaler) olarak döndürülmesi gerekir.
    cost = np.squeeze(cost) 
    
    return cost

# E) GERİYE YAYILIM (BACKWARD PROPAGATION)
def backward_propagation(parameters, cache, X, Y):
    
    # 1. Gerekli Değişkenleri Çekme
    m = X.shape[1] # Örnek sayısı
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # Parametreleri al
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    # 2. Çıkış Katmanı Gradyanları (Layer 2)
    
    # dZ2 = A2 - Y (Çapraz Entropi ve Sigmoid kombinasyonundan gelen sadeleşmiş formül)
    dZ2 = A2 - Y
    
    # dW2 = (1/m) * dZ2 . A1^T
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    
    # db2 = dZ2'nin örnekler üzerinden ortalaması
    db2 = np.mean(dZ2, axis=1, keepdims=True)
    
    # 3. Gizli Katman Gradyanları (Layer 1)
    
    # Hatanın geriye yayılımı: W2^T . dZ2
    # Bu, hatanın W2 üzerinden Layer 1'deki nöronlara dağıtılmasıdır.
    dZ1_weighted = np.dot(W2.T, dZ2)
    
    # tanh türevi: 1 - A1^2
    dZ1 = dZ1_weighted * (1 - np.power(A1, 2))
    
    # dW1 = (1/m) * dZ1 . X^T
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    
    # db1 = dZ1'in örnekler üzerinden ortalaması
    db1 = np.mean(dZ1, axis=1, keepdims=True)
    
    # Gradyanları bir sözlükte toplama
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    
    return grads

# F) PARAMETRE GÜNCELLEME (GRADIENT DESCENT)
def update_parameters(parameters, grads, learning_rate):
    
    # Parametreleri al
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Gradyanları al
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    # Gradyan İnişi Kuralını Uygulama
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    # Güncellenmiş parametreleri sözlüğe kaydetme
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    return parameters

# G) ANA EĞİTİM FONKSİYONU
def train_ann(X, Y, X_val, Y_val, n_h, learning_rate, num_epochs, print_cost=False): # [1] EKLEME: X_val, Y_val argümanları
    
    n_x = X.shape[0] # Girdi boyutu (2)
    n_y = Y.shape[0] # Çıktı boyutu (1)
    
    # 1. PARAMETRELERİ BAŞLATMA
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    costs = []
    validation_costs = [] # [2] EKLEME: Doğrulama maliyetlerini tutacak liste
    
    # 2. EĞİTİM DÖNGÜSÜ
    for i in range(num_epochs):
        
        # 1. İLERİ YAYILIM (Training Set üzerinde)
        A2, cache = forward_propagation(X, parameters)
        
        # 2. MALİYET HESAPLAMA (Training Cost)
        cost = compute_cost(A2, Y)
        
        # 3. GERİYE YAYILIM
        grads = backward_propagation(parameters, cache, X, Y)
        
        # 4. PARAMETRE GÜNCELLEME
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # [3] EKLEME: DOĞRULAMA MALİYETİ HESAPLAMA
        # Eğitimli parametreleri X_val üzerinde kullanıyoruz (sadece ileri yayılım)
        A2_val, _ = forward_propagation(X_val, parameters) 
        cost_val = compute_cost(A2_val, Y_val)
        
        # Maliyet takibi ve yazdırma
        if i % (EPOCHS / 10 ) == 0:
            costs.append(cost)
            validation_costs.append(cost_val) # [4] EKLEME: Doğrulama maliyetini listeye ekle
            
            if print_cost:
                print(f"Epoch {i} | EĞİTİM Maliyeti: {cost:.4f} | DOĞRULAMA Maliyeti: {cost_val:.4f}") # Düzenlendi
                
    # -------------------------------------------------------------
    # !!! YENİ EKLEME: DÖNGÜ BİTTİKTEN SONRA NİHAİ MALİYETİ EKLE !!!
    # -------------------------------------------------------------
    # Son maliyeti hesapla (Döngü num_epochs-1'de bitmiş olsa bile)
    A2_final, _ = forward_propagation(X, parameters)
    cost_final = compute_cost(A2_final, Y)
    A2_val_final, _ = forward_propagation(X_val, parameters) 
    cost_val_final = compute_cost(A2_val_final, Y_val)
    
    costs.append(cost_final)
    validation_costs.append(cost_val_final)
    
    if print_cost:
        # 10000. epoch'un maliyetini yazdır
        print(f"Epoch {num_epochs} | EĞİTİM Maliyeti: {cost_final:.4f} | DOĞRULAMA Maliyeti: {cost_val_final:.4f}")
    
    return parameters, costs, validation_costs 
                
    return parameters, costs, validation_costs # [5] DÜZENLEME: Doğrulama maliyetlerini de döndür
# H) EĞİTİMİ BAŞLATMA VE SONUÇLARI GÖRSELLEŞTİRME

print(f"--- Yapay Sinir Ağı Eğitimi Başlatılıyor ---")
print(f"Girdi Boyutu (N_X): {N_X} | Gizli Katman (N_H): {N_H} | Örnek Sayısı (m): {X.shape[1]}")
print(f"Öğrenme Hızı: {LEARNING_RATE} | Epoch Sayısı: {EPOCHS}\n")

# Modelin Eğitilmesi (!!! EĞİTİM BU SATIRDA BAŞLIYOR !!!)
optimized_parameters, costs, validation_costs = train_ann( # 3 değer döndürüyoruz
    X=X_train, 
    Y=Y_train, 
    X_val=X_val, # Doğrulama Seti
    Y_val=Y_val, # Doğrulama Etiketleri
    n_h=N_H, 
    learning_rate=LEARNING_RATE, 
    num_epochs=EPOCHS, 
    print_cost=True
)

print(f"\n--- Eğitim Tamamlandı ---")
print(f"Başlangıç EĞİTİM Maliyeti: {costs[0]:.4f}")
print(f"Nihai EĞİTİM Maliyeti (Epoch {EPOCHS}): {costs[-1]:.4f}")
print(f"Nihai DOĞRULAMA Maliyeti: {validation_costs[-1]:.4f}") # Doğrulama sonucunu da yazdırıyoruz

# Sonuçları Görselleştirme
epochs_list = np.arange(0, EPOCHS + 1, (EPOCHS / 10)) # X ekseni (0, 1000, 2000, ...) hazırlanır.

# Sonuçları Görselleştirme
import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))
# Eğitim Maliyeti
plt.plot(epochs_list, costs, label="Eğitim Maliyeti")
# Doğrulama Maliyeti
plt.plot(epochs_list, validation_costs, label="Doğrulama Maliyeti")

plt.title(f"Maliyet Azalımı (LR: {LEARNING_RATE}, N_H: {N_H})")
plt.xlabel("Epoch Sayısı")
plt.ylabel("Maliyet (J)")
plt.legend()
plt.grid(True)
plt.show()

# J) DOĞRULUK HESAPLAMA FONKSİYONU
def evaluate_accuracy(Y_prediction, Y_true):
    """
    Tahminleri (Y_prediction) gerçek etiketlerle (Y_true) karşılaştırarak
    modelin doğruluk skorunu hesaplar.
    """
    # NumPy'da iki matrisi karşılaştırmak için == kullanılır. 
    # Sonuç (True=1, False=0) matrisidir.
    correct_predictions = (Y_prediction == Y_true)
    
    # Ortalama almak, bize doğrudan doğru tahminlerin oranını (Doğruluğu) verir.
    accuracy = np.mean(correct_predictions) 
    
    return accuracy


# I) TAHMİN FONKSİYONU
def predict(parameters, X):
    """
    Eğitilmiş parametreleri kullanarak girdi X için tahmin yapar.
    
    Argümanlar:
    parameters -- Eğitilmiş W ve b parametreleri sözlüğü
    X -- Tahmin edilecek girdi matrisi (N, m)
    
    Döndürür:
    Y_prediction -- Girdi X için 0 veya 1 etiketlerinden oluşan NumPy vektörü
    """
    
    # Sadece İleri Yayılım yapıyoruz
    A2, cache = forward_propagation(X, parameters)
    
    # Karar Eşiği (Threshold): Olasılık 0.5'ten büyükse 1 (kazandı), değilse 0 (kaybetti).
    Y_prediction = (A2 > 0.5).astype(int)
    
    return Y_prediction

# K) NİHAİ PERFORMANS VE SONUÇ BLOĞU

# 1. Test Seti Tahmini (2024 verisi)
# optimized_parameters: train_ann'den gelen eğitilmiş W ve b değerleri
Y_pred_test = predict(optimized_parameters, X_test)

# 2. Doğruluk Hesaplama
# Modelin 2024 verisindeki nihai performansı
test_accuracy = evaluate_accuracy(Y_pred_test, Y_test)

print(f"\n--- Model Değerlendirmesi ---")
print(f"Eğitim Seti (1968-2021) Örnek Sayısı: {X.shape[1]}")
print(f"Doğrulama Seti (2022-2023) Örnek Sayısı: {X_val.shape[1]}")
print(f"Test Seti (2024) Örnek Sayısı: {X_test.shape[1]}")
print("-" * 35)
print(f"Nihai TEST Doğruluğu (2024): {test_accuracy*100:.2f}%")
print("-" * 35)