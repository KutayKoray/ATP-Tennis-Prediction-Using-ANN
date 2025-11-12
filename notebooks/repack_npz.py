import numpy as np
import pandas as pd
import os

# --- AYARLAR: Okunacak 3 CSV Dosyası ---
TRAIN_CSV = 'X_train_VERILERI_duzenlendi.csv' # Adım 1'de oluşturduğumuz yeni dosya
VAL_CSV = 'X_val_VERILERI.csv'             # 'inspect_npz.py'den gelen Orijinal dosya
TEST_CSV = 'X_test_VERILERI.csv'            # 'inspect_npz.py'den gelen Orijinal dosya

# --- ÇIKTI DOSYASI ---
OUTPUT_NPZ_FILE = 'processed_data_MANUAL_EDITED.npz'

print(f"--- Manuel Düzenlenmiş Veriler NPZ Dosyasına Paketleniyor ---")

try:
    # 1. Tüm CSV dosyalarını DataFrame olarak oku
    print(f"'{TRAIN_CSV}' okunuyor...")
    df_train = pd.read_csv(TRAIN_CSV)
    
    print(f"'{VAL_CSV}' okunuyor...")
    df_val = pd.read_csv(VAL_CSV)
    
    print(f"'{TEST_CSV}' okunuyor...")
    df_test = pd.read_csv(TEST_CSV)

    # 2. Özellik (Sütun) isimlerini al (DÜZELTME)
    
    # Train setinin sütun listesini al
    train_columns = df_train.columns.tolist()
    
    # DİNAMİK OLARAK BUL: Son sütunun etiket (Y) olduğunu varsay
    Y_LABEL_COLUMN_NAME = train_columns[-1] 
    
    # DİNAMİK OLARAK BUL: Geri kalan her şeyin özellik (X) olduğunu varsay
    FINAL_FEATURES = train_columns[:-1] 
    
    print(f"Etiket (Y) sütunu olarak bulundu: '{Y_LABEL_COLUMN_NAME}'")
    print(f"Toplam {len(FINAL_FEATURES)} özellik (feature) bulundu.")

    # 3. Y (Etiket) ve X (Özellik) olarak ayır
    
    # --- Train ---
    Y_train_df = df_train[Y_LABEL_COLUMN_NAME]
    X_train_df = df_train[FINAL_FEATURES] # Sadece özellikleri al
    
    # --- Val ---
    # Val ve Test setlerindeki etiket sütununun adının da son sütun olduğunu varsay
    Y_val_df = df_val[df_val.columns[-1]]
    X_val_df = df_val[FINAL_FEATURES]
    
    # --- Test ---
    Y_test_df = df_test[df_test.columns[-1]]
    X_test_df = df_test[FINAL_FEATURES]

    # 4. Modelin istediği formata (N_X, m) çevir (Transpoze et)
    X_train = X_train_df.T.values.astype(np.float64)
    Y_train = Y_train_df.values.reshape(1, -1)
    
    X_val = X_val_df.T.values.astype(np.float64)
    Y_val = Y_val_df.values.reshape(1, -1)
    
    X_test = X_test_df.T.values.astype(np.float64)
    Y_test = Y_test_df.values.reshape(1, -1)

    # 5. 'Aşama 2' setlerini (X_final_train) yeniden oluştur
    print("'Aşama 2' (Train+Val) setleri oluşturuluyor...")
    X_final_train = np.concatenate((X_train, X_val), axis=1)
    Y_final_train = np.concatenate((Y_train, Y_val), axis=1)

    # 6. Tüm dizileri tek bir .npz dosyasına kaydet
    print(f"Tüm veriler '{OUTPUT_NPZ_FILE}' dosyasına kaydediliyor...")
    np.savez_compressed(
        OUTPUT_NPZ_FILE,
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
    
    print("\n" + "="*50)
    print("--- İŞLEM BAŞARILI ---")
    print(f"Düzenlenmiş verileriniz başarıyla '{OUTPUT_NPZ_FILE}' dosyasına paketlendi.")
    print("Ana eğitim script'inizde bu .npz dosyasını okuyabilirsiniz.")
    print("="*50)

except FileNotFoundError as e:
    print(f"\n--- HATA: Dosya bulunamadı! ---")
    print(f"Dosya: {e.filename}")
    print("Lütfen Adım 1'i tamamladığınızdan ve 'X_val_VERILERI.csv' gibi dosyaların dizinde olduğundan emin olun.")
except KeyError as e:
    print(f"\n--- HATA: Sütun eşleşme sorunu! ---")
    print(f"Hata: {e}")
    print("Lütfen 'X_train_VERILERI_duzenlendi.csv', 'X_val_VERILERI.csv' ve 'X_test_VERILERI.csv' dosyalarının")
    print("sütun sayılarının ve sırasının eşleştiğinden emin olun (Etiket sütunu hariç).")
except Exception as e:
    print(f"\n--- Beklenmedik bir hata oluştu: {e} ---")