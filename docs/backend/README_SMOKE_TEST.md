# Smoke Test Script - Kullanım Kılavuzu

## Genel Bakış

`smoke_test.py` scripti, FPL Prediction Platform'un tüm core servislerini test eder:

1. **DB Connection**: Veritabanı bağlantısı testi
2. **ETL Check**: FPL API'den veri çekme ve DB'ye UPSERT testi
3. **ML Check**: PLEngine modellerinin 4GB RAM limiti içinde yüklenmesi ve xP hesaplama testi
4. **Solver Check**: FPLSolver'ın 100M bütçe ve 3 oyuncu limiti kısıtlarını koruması testi
5. **Strategy Check**: Strategy servisinin C/VC mantığının hatasız çalışması testi

## Çalıştırma

### Docker Container İçinde (Önerilen)

```bash
# Container'ları başlat
docker-compose up -d

# Backend container içinde test çalıştır
docker-compose exec backend python smoke_test.py

# Veya direkt container içine girip çalıştır
docker-compose exec backend bash
python smoke_test.py
```

### Yerel Ortamda

```bash
cd backend

# Bağımlılıkları yükle
pip install -r requirements.txt

# Test çalıştır
python smoke_test.py
```

## Test Senaryoları

### Test 1: Database Connection
- Veritabanına bağlanabilme kontrolü
- Basit SQL sorgusu çalıştırma

### Test 2: ETL Check
- FPL API'den bootstrap verisi çekme
- Örnek bir oyuncuyu DB'ye UPSERT etme
- UPSERT işleminin doğrulanması

### Test 3: ML Check
- PLEngine modellerinin yüklenmesi
- Örnek oyuncu için xP tahmini
- Bellek kullanımı takibi (4GB limit kontrolü)
- Model temizleme (memory management)

### Test 4: Solver Check
- 15 oyuncu (2GK, 5DEF, 5MID, 3FWD) ile test
- Bütçe kısıtı kontrolü (100M)
- Takım limiti kontrolü (max 3 oyuncu)
- Pozisyonel kısıt kontrolü (2-5-5-3)

### Test 5: Strategy Check
- C/VC Expected Value hesaplama
- Ownership arbitrage analizi
- Formül doğrulama: `Expected_Value = (xP_Capt * P_start_Capt) + (xP_VC * (1 - P_start_Capt))`

## Çıktı Formatı

Script her test için:
- ✓ Başarılı testler için onay işareti
- ✗ Başarısız testler için hata mesajı
- Detaylı log mesajları
- Bellek kullanımı bilgisi (ML test için)
- Test özeti (toplam, başarılı, başarısız)

## Hata Durumları

Script şu hataları yakalar ve raporlar:

- **MemoryError**: ML modelleri yüklenirken bellek taşması
- **ModelNotFoundError**: Model dosyaları bulunamadığında
- **DatabaseConnectionError**: DB bağlantı hatası
- **APIError**: FPL API'den veri çekilemediğinde
- **SolverError**: ILP çözücü hatası

## Exit Codes

- `0`: Tüm testler başarılı
- `1`: En az bir test başarısız

## Örnek Çıktı

```
============================================================
FPL PREDICTION PLATFORM - SMOKE TESTS
============================================================
Started at: 2025-01-XX...

============================================================
TEST 1: Database Connection
============================================================
✓ Database connection successful

============================================================
TEST 2: ETL Check (FPL API -> Database)
============================================================
Fetching bootstrap data from FPL API...
✓ Fetched 600+ players from FPL API
✓ UPSERT verified: Player Test Player (ID: 1)

...

============================================================
TEST SUMMARY
============================================================
Total Tests: 5
Passed: 5
Failed: 0

DB_CONNECTION: ✓ PASSED
ETL_CHECK: ✓ PASSED
ML_CHECK: ✓ PASSED (Memory used: 245.32 MB)
SOLVER_CHECK: ✓ PASSED
STRATEGY_CHECK: ✓ PASSED
```

## Notlar

- Testler sırayla çalıştırılır (DB testi başarısız olursa ETL testi atlanır)
- ML testi bellek kullanımını takip eder (psutil gerekli)
- Solver testi gerçek ILP optimizasyonu yapar (PuLP gerekli)
- Tüm testler gerçek servisleri kullanır (mock değil)

## Sorun Giderme

### "ModuleNotFoundError: No module named 'sqlalchemy'"
- `pip install -r requirements.txt` çalıştırın
- Veya Docker container içinde çalıştırın

### "Database connection error"
- PostgreSQL servisinin çalıştığından emin olun
- `DATABASE_URL` environment variable'ını kontrol edin

### "MemoryError" veya yüksek bellek kullanımı
- ML modelleri büyük olabilir
- 4GB RAM limiti aşılıyorsa model boyutlarını optimize edin
- `gc.collect()` çağrıları memory management için önemli

### "Solver status: Infeasible"
- Test verileri kısıtları sağlamıyor olabilir
- Sample player verilerini kontrol edin