import numpy as np
import pandas as pd
import os

# --- Pandas Ayarları (Geniş Tabloyu Göstermek İçin) ---
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 1000)

# --- İNCELENECEK NPZ DOSYASI ---
NPZ_FILE = 'processed_data_V2_ELO_1990plus.npz'

# --- OLUŞTURULACAK YENİ CSV DOSYALARININ ADLARI ---
OUTPUT_TRAIN_CSV = 'X_train_VERILERI.csv'
OUTPUT_VAL_CSV = 'X_val_VERILERI.csv'
OUTPUT_TEST_CSV = 'X_test_VERILERI.csv'

print(f"--- '{NPZ_FILE}' Dosyası İnceleniyor ve CSV'ye Dönüştürülüyor... ---")

if not os.path.exists(NPZ_FILE):
    print(f"\n--- HATA: '{NPZ_FILE}' dosyası bulunamadı! ---")
    print("Lütfen önce 'prepare_data_V2_1990+.py' script'ini çalıştırdığınızdan emin olun.")
else:
    try:
        # 1. Kaydedilmiş NumPy dizilerini yükle
        data = np.load(NPZ_FILE, allow_pickle=True)
        
        # 2. Özellik isimlerini (Sütun Başlıklarını) al
        FINAL_FEATURES = data['FINAL_FEATURES'].tolist()
        print(f"\nToplam {len(FINAL_FEATURES)} adet özellik (sütun) bulundu.")
        
        # 3. Y (Etiket) Verilerini Yükle
        Y_train_data = data['Y_train'].T
        Y_val_data = data['Y_val'].T
        Y_test_data = data['Y_test'].T

        # 4. X (Özellik) Verilerini Yükle ve DataFrame'e Çevir
        
        # --- EĞİTİM (TRAIN) VERİSİ ---
        print(f"\n'{OUTPUT_TRAIN_CSV}' oluşturuluyor...")
        X_train_data = data['X_train'].T 
        df_train_visual = pd.DataFrame(X_train_data, columns=FINAL_FEATURES)
        df_train_visual['Y_LABEL (1=P1 Kazandı)'] = Y_train_data
        # CSV OLARAK KAYDET
        df_train_visual.to_csv(OUTPUT_TRAIN_CSV, index=False)
        print(f"   -> Başarılı! {len(df_train_visual)} satır kaydedildi.")

        # --- DOĞRULAMA (VALIDATION) VERİSİ ---
        print(f"\n'{OUTPUT_VAL_CSV}' oluşturuluyor...")
        X_val_data = data['X_val'].T 
        df_val_visual = pd.DataFrame(X_val_data, columns=FINAL_FEATURES)
        df_val_visual['Y_LABEL (1=P1 Kazandı)'] = Y_val_data
        # CSV OLARAK KAYDET
        df_val_visual.to_csv(OUTPUT_VAL_CSV, index=False)
        print(f"   -> Başarılı! {len(df_val_visual)} satır kaydedildi.")

        # --- TEST VERİSİ ---
        print(f"\n'{OUTPUT_TEST_CSV}' oluşturuluyor...")
        X_test_data = data['X_test'].T 
        df_test_visual = pd.DataFrame(X_test_data, columns=FINAL_FEATURES)
        df_test_visual['Y_LABEL (1=P1 Kazandı)'] = Y_test_data
        # CSV OLARAK KAYDET
        df_test_visual.to_csv(OUTPUT_TEST_CSV, index=False)
        print(f"   -> Başarılı! {len(df_test_visual)} satır kaydedildi.")

        print("\n" + "="*50)
        print("--- İŞLEM TAMAMLANDI ---")
        print("Modelin kullandığı normalize edilmiş veriler CSV dosyalarına başarıyla aktarıldı.")
        print("Lütfen Excel veya başka bir programla şu dosyaları inceleyin:")
        print(f"1. {OUTPUT_TRAIN_CSV}")
        print(f"2. {OUTPUT_VAL_CSV}")
        print(f"3. {OUTPUT_TEST_CSV}")
        print("="*50)

    except Exception as e:
        print(f"\n--- Dosya işlenirken beklenmedik bir hata oluştu: {e} ---")