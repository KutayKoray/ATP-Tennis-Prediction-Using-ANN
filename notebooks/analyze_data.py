import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- KONTROL: Seaborn ve Matplotlib kurulu mu? ---
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    PLOT_ENABLED = True
except ImportError:
    print("UYARI: 'seaborn' veya 'matplotlib' kütüphaneleri bulunamadı.")
    print("Grafikler oluşturulmayacak. Lütfen 'pip install seaborn matplotlib' ile kurun.")
    PLOT_ENABLED = False

print("--- Veri Analizi Script'i Başlatıldı ---")
FILE_NAME = '../datas/all_atp_matches_1968_2024.csv'

if not os.path.exists(FILE_NAME):
    print(f"HATA: '{FILE_NAME}' dosyası bulunamadı. Lütfen dosyanın doğru dizinde olduğundan emin olun.")
else:
    print(f"'{FILE_NAME}' yükleniyor. Bu işlem birkaç saniye sürebilir...")
    df = pd.read_csv(FILE_NAME, low_memory=False)

    # ===============================================
    # BÖLÜM 1: TEMİZLİK VE VERİ TİPLERİ
    # ===============================================
    print("\n--- BÖLÜM 1: Veri Sütunları ve Tipleri (Genel Bakış) ---")
    # .info() çıktısı çok yer kaplar, özetini yazdıralım
    print(f"Toplam Satır (Maç) Sayısı: {len(df)}")
    print(f"Toplam Sütun (Özellik) Sayısı: {len(df.columns)}")
    print("Veri tipleri ve eksik veri (NaN) sayıları (İlk 25 sütun):")
    print(df.info(max_cols=25))


    # ===============================================
    # BÖLÜM 2: EKSİK VERİ ANALİZİ (YILLARA GÖRE)
    # ===============================================
    print("\n--- BÖLÜM 2: Yıllara Göre Kritik Veri Eksikliği Analizi ---")
    print("Bu analiz, 'Dağılım Kayması' (Distribution Shift) sorununu ortaya çıkarır.")
    
    # Tarih sütununu işle
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
    df['year'] = df['tourney_date'].dt.year

    # Modelimiz için kullandığımız/kullanacağımız temel özellikler
    features_to_check = [
        'winner_rank',      # Sıralama (Puan yerine)
        'winner_rank_points', # Sıralama Puanı (Enflasyonlu)
        'winner_ht',        # Boy
        'winner_hand',      # El
        'surface',          # Zemin
        'minutes',          # Maç süresi (Yorgunluk için)
        'w_ace',            # Servis istatistikleri
        'l_ace'
    ]

    # Yıla göre grupla ve her özellik için eksik (NaN) oranını hesapla
    missing_data_by_year = df.groupby('year')[features_to_check].apply(lambda x: x.isnull().mean()) * 100

    print("\nKritik Özelliklerin Yıllara Göre Eksiklik Yüzdesi (% NaN):")
    print(missing_data_by_year.tail(10)) # Son 10 yılı göster

    if PLOT_ENABLED:
        try:
            plt.figure(figsize=(15, 8))
            # Isı haritası: Koyu renk = Veri Var, Açık renk = Veri Eksik
            sns.heatmap(missing_data_by_year.T, cmap='viridis', annot=False, fmt='.0f')
            plt.title("Kritik Özelliklerin Yıllara Göre Eksiklik Durumu (% NaN)")
            plt.xlabel("Yıl")
            plt.ylabel("Özellik")
            plt.tight_layout()
            plt.savefig('missing_data_heatmap.png')
            print("\nGRAFİK KAYDEDİLDİ: 'missing_data_heatmap.png'")
            print("   (Yorum: Açık renkler (Sarı), o yıl o verinin olmadığını gösterir.)")
        except Exception as e:
            print(f"UYARI: Isı haritası çizilirken hata oluştu: {e}")


    # ===============================================
    # BÖLÜM 3: AYKIRI DEĞER (OUTLIER) ANALİZİ
    # ===============================================
    print("\n--- BÖLÜM 3: Aykırı Değer (Outlier) Analizi (Sayısal) ---")
    
    # Boy (ht) ve Yaş (age) için .describe()
    numeric_features = ['winner_age', 'loser_age', 'winner_ht', 'loser_ht', 'minutes']
    print(df[numeric_features].describe())
    
    # 'winner_rank' ve 'winner_rank_points' için .describe()
    print(df[['winner_rank', 'loser_rank', 'winner_rank_points', 'loser_rank_points']].describe())

    if PLOT_ENABLED:
        try:
            # Yaş Dağılımı
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            sns.histplot(df['winner_age'].dropna(), bins=30, kde=True, color='blue', label='Kazanan')
            sns.histplot(df['loser_age'].dropna(), bins=30, kde=True, color='red', label='Kaybeden', alpha=0.6)
            plt.title("Yaş Dağılımı (Kazanan vs Kaybeden)")
            plt.legend()
            
            # Boy Dağılımı
            plt.subplot(1, 2, 2)
            # Boy verisi daha az, o yüzden sadece tüm oyuncuları (winner_ht) alalım
            sns.histplot(df['winner_ht'].dropna(), bins=30, kde=True, color='green')
            plt.title("Oyuncu Boy Dağılımı (cm)")
            plt.tight_layout()
            plt.savefig('age_height_distributions.png')
            print("\nGRAFİK KAYDEDİLDİ: 'age_height_distributions.png'")
            print("   (Yorum: Bu grafiklerde mantıksız (örn: 50cm) veya (250cm) gibi aykırı değerler var mı?)")

        except Exception as e:
            print(f"UYARI: Dağılım grafikleri çizilirken hata oluştu: {e}")

    # ===============================================
    # BÖLÜM 4: KATEGORİK VERİ BÜTÜNLÜĞÜ
    # ===============================================
    print("\n--- BÖLÜM 4: Kategorik Veri Bütünlüğü ---")
    
    print("\n'surface' (Zemin) Sütunu Benzersiz Değerleri:")
    print(df['surface'].value_counts(dropna=False))
    print("\n   (Yorum: 'Carpet' (Halı) artık kullanılmıyor ve 'None'/'NaN' değerleri var.)")
    
    print("\n'winner_hand' (El) Sütunu Benzersiz Değerleri:")
    print(df['winner_hand'].value_counts(dropna=False))
    print("\n   (Yorum: 'U' (Unknown/Bilinmeyen) ve 'NaN' (Eksik) değerler çoğunlukla 'R' (Sağ) ile doldurulabilir.)")


    # ===============================================
    # BÖLÜM 5: KAZANMA İLE İLİŞKİ (FEATURE IMPORTANCE)
    # ===============================================
    print("\n--- BÖLÜM 5: Özelliklerin Kazanma İle İlişkisi ---")

    # --- 1. Sıralama (Rank) ---
    # (Puan (rank_points) enflasyonlu olduğu için 'rank' daha güvenilirdir)
    df_ranks = df[['winner_rank', 'loser_rank']].dropna()
    higher_rank_won = (df_ranks['winner_rank'] < df_ranks['loser_rank']).mean()
    print(f"\nSıralaması Daha İyi Olan Oyuncu (düşük 'rank' sayısı) maçların %{higher_rank_won * 100:.2f}'sini kazandı.")
    print("   (Yorum: Bu, modelimizin öğrendiği en temel kuraldır. %50'den ne kadar yüksekse o kadar iyidir.)")
    
    # --- 2. Yaş (Age) ---
    df_age = df[['winner_age', 'loser_age']].dropna()
    younger_player_won = (df_age['winner_age'] < df_age['loser_age']).mean()
    print(f"\nDaha Genç Olan Oyuncu maçların %{younger_player_won * 100:.2f}'sini kazandı.")
    print("   (Yorum: Bu sonucun %50'ye yakın olması (örn: %51.84), yaşın tek başına güçlü bir gösterge olmadığını, tecrübenin de önemli olduğunu gösterir.)")

    # --- 3. Boy (Height) ---
    df_ht = df[['winner_ht', 'loser_ht']].dropna()
    taller_player_won = (df_ht['winner_ht'] > df_ht['loser_ht']).mean()
    print(f"\nDaha Uzun Boylu Olan Oyuncu maçların %{taller_player_won * 100:.2f}'sini kazandı.")
    print("   (Yorum: %50'den yüksek olması, boy avantajının teniste (örn: servis) gerçek bir faktör olduğunu gösterir.)")

    # --- 4. Zemin ve El Etkileşimi ---
    print("\nZeminlere Göre Kazanan El Oranları (R/L/U):")
    # (dropna=True ile bilinmeyen elleri hesaba katmayız)
    crosstab_hand_surface = pd.crosstab(df['surface'], df['winner_hand'], normalize='index') * 100
    print(crosstab_hand_surface.to_string(float_format="%.2f%%"))
    print("\n   (Yorum: Solakların ('L') tüm zeminlerde (özellikle 'Grass' - Çim) genel nüfusa göre (%10-11) daha başarılı olup olmadığını buradan görebiliriz.)")

    # --- 5. "Sürpriz" Grafiği (Rank Farkı) ---
    if PLOT_ENABLED:
        try:
            df_ranks['rank_diff'] = df_ranks['loser_rank'] - df_ranks['winner_rank']
            plt.figure(figsize=(10, 6))
            # Sadece favorilerin kazandığı (rank_diff > 0) ve sürprizlerin olduğu (rank_diff < 0) durumlara odaklan
            sns.histplot(df_ranks['rank_diff'].clip(-100, 300), bins=100, kde=True)
            plt.title("Sıralama Farkı Dağılımı (Kaybeden Rankı - Kazanan Rankı)")
            plt.xlabel("Sıralama Farkı (Pozitif = Favori Kazandı)")
            plt.ylabel("Maç Sayısı")
            plt.axvline(0, color='red', linestyle='--', label='Sürpriz Sınırı (0)')
            plt.legend()
            plt.savefig('rank_difference_histogram.png')
            print("\nGRAFİK KAYDEDİLDİ: 'rank_difference_histogram.png'")
            print("   (Yorum: Grafiğin 0'ın sağında yığılması beklenir. 0'ın solundaki tepecikler 'sürpriz' maçları gösterir.)")
        except Exception as e:
            print(f"UYARI: Sıralama farkı grafiği çizilirken hata oluştu: {e}")
            
    print("\n--- Analiz Tamamlandı ---")