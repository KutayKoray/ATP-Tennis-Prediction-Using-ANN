# ATP Tennis Match Prediction (ANN From Scratch)

Bu proje, ATP tenis maçı verilerini kullanarak maç kazananını tahmin etmeyi amaçlayan, NumPy ile sıfırdan (TensorFlow/Keras kullanmadan) yazılmış ileri beslemeli yapay sinir ağı (ANN) denemelerini içerir. Veri seti 1968–2024 yılları arasındaki ATP maçlarını kapsar. Eğitim/Doğrulama/Test ayrımı zamansal olarak yapılır.

## Proje Yapısı
- **datas/**: ATP maçları, oyuncular ve sıralama CSV’leri, ayrıca veri sözlüğü
- **notebooks/**:
  - `04_atp_eda_overview_v2.ipynb`: EDA, özet ve kapsam
  - `1HLayer_ANN.py`: Tek gizli katmanlı ANN (NumPy ile)
  - `2HLayer_ANN.py`: İki gizli katmanlı ANN (NumPy ile)
  - `1HLayer_ANN_Notebook*.ipynb`: Deneysel notebook’lar
- **requirements.txt**: Gerekli Python paketleri
- **MyNotes.txt**: Deney notları ve sonraki adımlar

## Veri ve Özellikler
- Kaynak CSV’ler `datas/` klasöründe (1968–2024 maçları, oyuncu ve ranking kayıtları)
- Kullanılan ham sütunlar (örnek): `winner_id`, `loser_id`, `winner_rank_points`, `loser_rank_points`, `winner_age`, `loser_age`, `winner_ht`, `loser_ht`, `surface`, `winner_hand`, `loser_hand`
- Anonimleştirme: `p1` ve `p2` rastgele atanır; etiket `p1_wins`
- Özellik mühendisliği (özet):
  - `p1/p2_rank_points`, `p1/p2_age`, `p1/p2_ht`
  - `p1/p2_hand` dönüşümü ve solak (`*_is_L`) bayrakları
  - `surface` için one-hot encoding; tüm setler arası kategori hizalaması

## Zaman Bazlı Veri Ayrımı
- **Eğitim**: 1968–2021
- **Doğrulama**: 2022–2023
- **Test**: 2024

## Kurulum
1. Python 3.9+ önerilir.
2. Sanal ortam oluşturup etkinleştirin (opsiyonel):
   - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
   - Windows (PowerShell): `python -m venv .venv; .venv\Scripts\Activate.ps1`
3. Bağımlılıklar:
   ```bash
   pip install -r requirements.txt
   ```

## Çalıştırma
- Komut satırından (veri yolu `../datas/` olacak şekilde ayarlı):
  ```bash
  python notebooks/1HLayer_ANN.py
  python notebooks/2HLayer_ANN.py
  ```
  Not: `notebooks/*.py` dosyaları `../datas/` yolunu varsayar. Proje yapınız farklıysa `path` değişkenini güncelleyin.

- Notebook’lar üzerinden:
  - `notebooks/04_atp_eda_overview_v2.ipynb` (EDA)
  - `notebooks/1HLayer_ANN_Notebook*.ipynb` (deneyler)

## Mevcut Durum ve Sonuçlar
- Doğrulama doğruluğu mevcut özellik ve mimariyle ~%63 civarında seyretmektedir (bkz. `MyNotes.txt`).
- Eğitim sırasında eğitim/doğrulama maliyeti yazdırılır ve basit maliyet grafikleri çizilir.

## Yol Haritası (MyNotes’a göre)
- Standardizasyon: `StandardScaler` benzeri 0-ortalama/1-std ölçekleme
- Düzenlileştirme: L2, Dropout
- Optimizasyon: Mini-batch, Adam, Early Stopping
- Özellik geliştirme: Ek oyuncu/zemin/forma metrikleri, sıralama-temelli ek sinyaller

## Gereksinimler
- Büyük CSV’ler nedeniyle bellek kullanımına dikkat edin. Gerekirse yalnızca gerekli sütunlarla yükleme yapılır (kodda `usecols`).

## Lisans
- Bu depo için lisans belirtilmemiştir.
