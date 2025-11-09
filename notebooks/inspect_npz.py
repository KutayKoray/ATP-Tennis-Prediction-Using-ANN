import numpy as np
import pandas as pd

# Pandas'ın tüm sütunları göstermesini sağla
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("--- 'processed_data_H2H_Momentum.npz' Yükleniyor... ---")

try:
    data = np.load('processed_data_H2H_Momentum.npz', allow_pickle=True)
    
    # 1. Dosyanın içinde hangi diziler var?
    print(f"Dosyanın içindeki kaydedilmiş diziler: {list(data.files)}")
    
    # 2. En önemlisi: Özellik isimleri neler?
    FINAL_FEATURES = data['FINAL_FEATURES'].tolist()
    print("\n--- Modelin Kullandığı Özellikler (FINAL_FEATURES) ---")
    print(FINAL_FEATURES)
    
    # 3. Verinin kendisini görelim
    # X_train verisini (N_X, m) formatından (m, N_X) formatına çevir (Transpose)
    # ve DataFrame'e yükle.
    X_train_data = data['X_train'].T 
    
    # Sütun isimlerini ata
    df_visual = pd.DataFrame(X_train_data, columns=FINAL_FEATURES)
    
    print("\n--- Veri Görselleştirmesi (İlk 5 Maç) ---")
    print(df_visual.head())
    
    print("\n--- Veri Tipleri ve Boyutu ---")
    df_visual.info()
    
    # Önemli Not: Bu DataFrame'deki değerler (örn: 0.004),
    # 'prepare_data.py' script'inin sonunda normalize edilmiş değerlerdir.
    
except FileNotFoundError:
    print("\n--- HATA: 'processed_data_H2H_Momentum.npz' dosyası bulunamadı! ---")