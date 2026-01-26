# ML Training and Prediction Guide

Bu doküman, ML model eğitimi ve prediction üretimi için gerekli scriptleri ve çalıştırma sırasını açıklar.

## 📋 Gerekli Scriptler

### 1. ML Model Eğitimi: `train_ml_models.py`

**Konum:** `backend/scripts/train_ml_models.py`

**Ne Yapar:**
- Tüm component modellerini (xMins, Attack, Defense) eğitir
- Veritabanından historical data yükler
- Eğitilmiş modelleri `.pkl` dosyasına kaydeder
- Model dosyası: `models/plengine_model_gw{gameweek}_{season}.pkl`

**Kullanım:**
```bash
# Docker içinde çalıştırma
docker compose exec backend python3 scripts/train_ml_models.py [gameweek] [season]

# Örnekler:
# Mevcut gameweek için eğit (otomatik gameweek tespit eder)
docker compose exec backend python3 scripts/train_ml_models.py

# Belirli bir gameweek'e kadar eğit
docker compose exec backend python3 scripts/train_ml_models.py 10

# Belirli gameweek ve season ile eğit
docker compose exec backend python3 scripts/train_ml_models.py 10 "2025-26"
```

**Gereksinimler:**
- Veritabanında en az 5 gameweek'lik historical data olmalı
- `PlayerGameweekStats` tablosunda yeterli veri bulunmalı

**Çıktı:**
- Model dosyası: `models/plengine_model_gw{gameweek}_{season}.pkl`
- Training logları konsola yazdırılır

---

### 2. Prediction Güncelleme: `update_predictions.py`

**Konum:** `backend/app/scripts/update_predictions.py`

**Ne Yapar:**
- Eğitilmiş modelleri yükler (lazy loading)
- Tüm oyuncular için prediction hesaplar
- Prediction'ları `Prediction` tablosuna kaydeder
- Injury/suspension durumlarını kontrol eder ve filtreler

**Kullanım:**
```bash
# Docker içinde çalıştırma
docker compose exec backend python3 app/scripts/update_predictions.py [gameweek]

# Örnekler:
# Mevcut gameweek için prediction güncelle
docker compose exec backend python3 app/scripts/update_predictions.py

# Belirli bir gameweek için prediction güncelle
docker compose exec backend python3 app/scripts/update_predictions.py 10
```

**Gereksinimler:**
- Modeller önceden eğitilmiş olmalı (train_ml_models.py çalıştırılmış olmalı)
- Model dosyası `models/` dizininde bulunmalı
- Veritabanında `Player` ve `PlayerGameweekStats` verileri olmalı

**Çıktı:**
- `Prediction` tablosunda güncellenmiş prediction'lar
- Her oyuncu için: xp, xg, xa, xmins, xcs, defcon_score, confidence_score

---

## 🔄 Çalıştırma Sırası

### İlk Kurulum (Boş Veritabanı)

1. **Veritabanını Doldur:**
   ```bash
   # Önce takımları, oyuncuları ve gameweek istatistiklerini yükle
   docker compose exec backend python3 scripts/populate_database.py
   ```

2. **ML Modellerini Eğit:**
   ```bash
   # En az 5 gameweek verisi olduktan sonra modelleri eğit
   docker compose exec backend python3 scripts/train_ml_models.py
   ```

3. **Prediction'ları Oluştur:**
   ```bash
   # Eğitilmiş modellerle prediction'ları hesapla
   docker compose exec backend python3 app/scripts/update_predictions.py
   ```

### Günlük/Periyodik Güncelleme

1. **Yeni Gameweek Verilerini Yükle:**
   ```bash
   # ETL servisi ile yeni gameweek verilerini çek
   docker compose exec backend python3 scripts/populate_database.py
   ```

2. **Modelleri Yeniden Eğit (Opsiyonel):**
   ```bash
   # Yeni verilerle modelleri yeniden eğit (haftalık önerilir)
   docker compose exec backend python3 scripts/train_ml_models.py
   ```

3. **Prediction'ları Güncelle:**
   ```bash
   # Her gameweek için prediction'ları güncelle
   docker compose exec backend python3 app/scripts/update_predictions.py
   ```

---

## ⚠️ Yaygın Hatalar ve Çözümleri

### Hata: "Model not loaded. Call load() first."

**Sebep:** Modeller henüz eğitilmemiş veya model dosyası bulunamıyor.

**Çözüm:**
1. Önce `train_ml_models.py` scriptini çalıştırın
2. Model dosyasının `models/` dizininde olduğundan emin olun
3. Model dosyası yolu doğru mu kontrol edin

### Hata: "Not enough gameweeks for training (need 5, have X)"

**Sebep:** Veritabanında yeterli historical data yok.

**Çözüm:**
1. Daha fazla gameweek verisi yükleyin
2. `populate_database.py` scriptini çalıştırın
3. En az 5 gameweek verisi olduğundan emin olun

### Hata: "No training data available!"

**Sebep:** Veritabanında `PlayerGameweekStats` verisi yok.

**Çözüm:**
1. FPL API'den veri çekin
2. `populate_database.py` veya ETL servisi ile verileri yükleyin

---

## 📊 Model Dosya Yapısı

Model dosyaları şu formatta kaydedilir:
```
models/plengine_model_gw{gameweek}_{season}.pkl
```

Örnek:
- `models/plengine_model_gw10_2025_26.pkl`
- `models/plengine_model_gw15_2025_26.pkl`

Model dosyası içeriği:
- `xmins_model`: XGBoost/RandomForest modeli
- `attack_xg_model`: LightGBM xG modeli
- `attack_xa_model`: LightGBM xA modeli
- `defense_model`: LightGBM/Poisson clean sheet modeli

---

## 🔍 Model Yükleme Mekanizması

PLEngine otomatik olarak en son model dosyasını bulur:
1. `backend/models/` dizininde
2. `/app/models/` dizininde (Docker)
3. `models/` dizininde (current working directory)

En son değiştirilme tarihine göre en güncel `.pkl` dosyası seçilir.

---

## 📝 Notlar

- **Training Süresi:** Modellerin eğitimi veri miktarına bağlı olarak 5-30 dakika sürebilir
- **Prediction Süresi:** Tüm oyuncular için prediction hesaplama ~2-5 dakika sürebilir
- **Model Boyutu:** Her model dosyası ~50-200 MB olabilir
- **RAM Kullanımı:** Modeller lazy loading ile yüklenir, kullanımdan sonra unload edilir (4GB RAM constraint için)

---

## 🚀 Otomatik Çalıştırma

FastAPI uygulaması başlatıldığında (`app/main.py`):
- Modeller otomatik yüklenir (startup event)
- Background task ile prediction'lar otomatik güncellenir
- Günlük ETL scheduler çalışır (saat 02:00)

Manuel çalıştırma gerekmez, ancak ilk kurulumda veya hata durumunda yukarıdaki scriptler manuel çalıştırılabilir.
