import pandas as pd
import os

# --- AYARLAR ---
# Düzenlediğiniz Excel dosyasının adı
INPUT_EXCEL_FILE = 'X_train_VERILERI.xlsx' 

# Oluşturulacak yeni CSV dosyasının adı
OUTPUT_CSV_FILE = 'X_train_VERILERI_duzenlendi.csv'
# ----------------

print(f"'{INPUT_EXCEL_FILE}' okunuyor...")

if not os.path.exists(INPUT_EXCEL_FILE):
    print(f"HATA: '{INPUT_EXCEL_FILE}' dosyası bulunamadı.")
else:
    try:
        # 1. Excel dosyasını oku
        df_excel = pd.read_excel(INPUT_EXCEL_FILE)
        
        # 2. Yeni CSV dosyası olarak kaydet
        df_excel.to_csv(OUTPUT_CSV_FILE, index=False)
        
        print(f"--- BAŞARILI ---")
        print(f"Dosya başarıyla '{OUTPUT_CSV_FILE}' olarak kaydedildi.")
        
    except Exception as e:
        print(f"HATA: Dosya dönüştürülürken bir sorun oluştu: {e}")
        print("Lütfen 'pip install openpyxl' komutunu çalıştırdığınızdan emin olun.")