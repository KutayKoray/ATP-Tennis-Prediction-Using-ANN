import pandas as pd
import glob
import os

# 1. Dosyalarınızın bulunduğu dizin (PATH)
data_path = '../datas/'

# 2. Kaydedilecek olan birleştirilmiş dosyanın adı
output_file = os.path.join(data_path, 'all_atp_matches_1968_2024.csv')

# 3. 'atp_matches_' ile başlayan tüm CSV dosyalarını bulmak için glob deseni
#    (işletim sisteminden bağımsız çalışması için os.path.join kullanılır)
file_pattern = os.path.join(data_path, 'atp_matches_*.csv')

# 4. Desenle eşleşen tüm dosyaların bir listesini al
#    sorted() kullanarak kronolojik olarak doğru sırada (1968, 1969...) olmalarını garantileriz.
all_files = sorted(glob.glob(file_pattern))

if not all_files:
    print(f"HATA: '{data_path}' dizininde 'atp_matches_*.csv' formatında hiçbir dosya bulunamadı.")
else:
    print(f"Toplam {len(all_files)} adet CSV dosyası bulundu. Birleştirme işlemi başlıyor...")

    # 5. Tüm DataFrameleri geçici olarak saklamak için bir liste oluştur
    li = []

    # 6. Bulunan her bir dosyayı döngüye al
    for filename in all_files:
        try:
            # Dosyayı oku
            df_temp = pd.read_csv(filename, index_col=None, header=0, low_memory=False)
            
            # Okunan DataFrame'i listeye ekle
            li.append(df_temp)
            print(f"  -> {filename} okundu ve eklendi.")
        except Exception as e:
            print(f"  -> UYARI: {filename} okunurken bir sorun oluştu: {e}")

    # 7. Tüm DataFrameler okunduktan sonra, hepsini tek bir DataFrame'de birleştir
    if li:
        print("\nTüm dosyalar okundu. Tek bir DataFrame'de birleştiriliyor...")
        
        # axis=0: Dikey olarak (satırları alt alta) birleştir
        # ignore_index=True: Eski dosyalardaki index'leri at, yeni bir index (0, 1, 2...) oluştur
        combined_df = pd.concat(li, axis=0, ignore_index=True)

        # 8. Birleştirilmiş DataFrame'i yeni bir CSV dosyasına kaydet
        print(f"\nBirleştirme tamamlandı. Sonuçlar '{output_file}' dosyasına kaydediliyor...")
        try:
            # index=False: Pandas'ın DataFrame index'ini dosyaya yazmasını engeller
            combined_df.to_csv(output_file, index=False)
            
            print("\n--- İŞLEM BAŞARILI ---")
            print(f"Tüm veriler başarıyla '{output_file}' dosyasına kaydedildi.")
            print(f"Toplam {len(combined_df)} maç satırı oluşturuldu.")
        
        except Exception as e:
            print(f"\n--- HATA ---")
            print(f"Birleştirilmiş dosya kaydedilirken bir hata oluştu: {e}")
    else:
        print("Birleştirilecek hiçbir dosya bulunamadı veya okunamadı.")