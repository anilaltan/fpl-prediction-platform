# FPL Prediction Platform - Technical Reference Map

## İçindekiler

1. [Genel Mimari](#genel-mimari)
2. [Dosya ve Servis Yapısı](#dosya-ve-servis-yapısı)
3. [Backend Servisleri ve Fonksiyon Detayları](#backend-servisleri-ve-fonksiyon-detayları)
4. [API Endpoint'leri](#api-endpointleri)
5. [Veritabanı Modelleri](#veritabanı-modelleri)
6. [Veri Akışı (Data Flow)](#veri-akışı-data-flow)
7. [Frontend Yapısı](#frontend-yapısı)
8. [Bağımlılıklar ve Çevre Değişkenleri](#bağımlılıklar-ve-çevre-değişkenleri)

---

## Genel Mimari

### Teknoloji Stack

- **Backend**: FastAPI (Python 3.11), SQLAlchemy, PostgreSQL (TimescaleDB)
- **Frontend**: Next.js 14 (App Router), React, TypeScript, Tailwind CSS
- **ML Engine**: XGBoost, LightGBM, Random Forest, Poisson Regression
- **Orchestration**: Docker Compose
- **Database**: PostgreSQL 15 with TimescaleDB extension

### Mimari Prensipler

1. **Moneyball Principles**: Veri odaklı karar verme, istatistiksel analiz
2. **DefCon Rules**: Rate limiting, graceful error handling, resource constraints
3. **Batch Prediction Architecture**: ML hesaplamaları arka planda, API sadece DB'den okur
4. **Graceful Degradation**: Cache-first yaklaşım, fallback mekanizmaları

---

## Dosya ve Servis Yapısı

### Backend Yapısı

```
backend/
├── app/
│   ├── main.py                    # FastAPI uygulaması ve tüm endpoint'ler
│   ├── models.py                   # SQLAlchemy veritabanı modelleri
│   ├── schemas.py                  # Pydantic request/response şemaları
│   ├── database.py                 # Database bağlantı yönetimi
│   ├── services/                   # İş mantığı servisleri
│   │   ├── fpl_api.py              # FPL API entegrasyonu
│   │   ├── etl_service.py          # ETL operasyonları
│   │   ├── ml_engine.py            # ML tahmin motoru (PLEngine)
│   │   ├── predictive_engine.py    # Bileşen bazlı tahmin motoru
│   │   ├── feature_engineering.py  # Özellik mühendisliği
│   │   ├── team_solver.py          # Takım optimizasyon çözücüsü
│   │   ├── market_intelligence.py  # Piyasa zekası servisi
│   │   ├── risk_management.py      # Risk yönetimi servisi
│   │   ├── backtesting.py          # Backtest motoru
│   │   ├── entity_resolution.py   # Entity resolution servisi
│   │   ├── data_cleaning.py        # Veri temizleme servisi
│   │   ├── third_party_data.py     # Üçüncü parti veri servisleri
│   │   └── ...
│   └── scripts/                    # Yardımcı scriptler
│       ├── update_predictions.py   # Tahmin güncelleme scripti
│       └── ...
├── models/                         # Eğitilmiş ML modelleri (.pkl)
├── populate_all_tables.py          # Veritabanı doldurma scripti
└── requirements.txt                # Python bağımlılıkları
```

### Frontend Yapısı

```
frontend/
├── app/
│   ├── page.tsx                    # Ana sayfa (Dashboard)
│   ├── layout.tsx                  # Root layout
│   ├── api/                        # Next.js API routes
│   │   ├── players/
│   │   ├── team/
│   │   ├── market/
│   │   └── ...
│   ├── players/                    # Oyuncu listesi sayfası
│   ├── team-optimizer/             # Takım optimizasyon sayfası
│   ├── market-intelligence/        # Piyasa zekası sayfası
│   └── ...
└── components/                      # React bileşenleri
```

---

## Backend Servisleri ve Fonksiyon Detayları

### 1. FPLAPIService (`app/services/fpl_api.py`)

**Amaç**: FPL Official API ile entegrasyon, veri çekme, cache yönetimi, rate limiting

#### Ana Sınıflar

##### RateLimiter
- **Konum**: `app/services/fpl_api.py:39-96`
- **Amaç**: DefCon rate limiting (60 req/min) ve exponential backoff
- **Metodlar**:
  - `acquire()`: Rate limit kontrolü, bekleme
  - `record_success()`: Başarılı istek kaydı
  - `record_error()`: Hata kaydı ve backoff artırımı

##### InMemoryCache
- **Konum**: `app/services/fpl_api.py:143-173`
- **Amaç**: TTL destekli in-memory cache
- **Metodlar**:
  - `get(key)`: Cache'den veri okuma
  - `set(key, value, ttl_seconds)`: Cache'e yazma
  - `clear(key)`: Cache temizleme

##### FPLAPIService
- **Konum**: `app/services/fpl_api.py:180-1095`
- **Bağımlılıklar**: `EntityResolutionService`, `DataCleaningService`, `ETLService`

#### Önemli Metodlar

| Metod | Satır | Parametreler | Dönüş Değeri | Bağımlılıklar |
|-------|-------|--------------|--------------|---------------|
| `get_bootstrap_data()` | 221-262 | `use_cache: bool = True` | `Dict` | RateLimiter, InMemoryCache |
| `get_current_gameweek()` | 264-301 | - | `Optional[int]` | `get_bootstrap_data()` |
| `get_next_gameweek()` | 303-342 | - | `Optional[int]` | `get_bootstrap_data()` |
| `extract_players_from_bootstrap()` | 344-419 | `bootstrap_data: Dict` | `List[Dict]` | - |
| `get_player_data()` | 440-480 | `player_id: int, use_cache: bool = True` | `Dict` | RateLimiter, Cache |
| `extract_player_history()` | 482-537 | `player_summary: Dict` | `List[Dict]` | `DataCleaningService.normalize_dgw_points()` |
| `get_fixtures()` | 539-579 | `gameweek: Optional[int], future_only: bool = False` | `List[Dict]` | RateLimiter |
| `fetch_fbref_defcon_metrics()` | 635-746 | `season: str, player_name: Optional[str]` | `List[Dict]` | httpx, BeautifulSoup |
| `map_players_with_fbref()` | 764-839 | `fpl_players, fbref_players, use_fuzzy, threshold` | `Dict[int, Dict]` | `EntityResolutionService`, FuzzyWuzzy |
| `fetch_comprehensive_player_data()` | 843-920 | `player_id, season, include_fbref, normalize_dgw` | `Dict` | `get_player_data()`, `fetch_fbref_defcon_metrics()` |
| `save_player_gameweek_stats()` | 924-1011 | `player_id, gameweek, season, include_fbref` | `Dict` | `fetch_comprehensive_player_data()`, `ETLService.upsert_player_gameweek_stats()` |
| `bulk_save_gameweek_stats()` | 1013-1074 | `gameweek, season, max_players` | `Dict` | `save_player_gameweek_stats()` |

**Kullanım Yerleri**:
- `main.py`: Tüm endpoint'lerde FPL verisi çekmek için
- `populate_all_tables.py`: Veritabanı doldurma
- `market_intelligence.py`: Ownership verisi çekmek için

---

### 2. ETLService (`app/services/etl_service.py`)

**Amaç**: PostgreSQL'e veri yükleme, UPSERT operasyonları

#### Ana Metodlar

| Metod | Satır | Parametreler | Dönüş Değeri | Bağımlılıklar |
|-------|-------|--------------|--------------|---------------|
| `upsert_player()` | 50-155 | `player_data: Dict, session: Optional[AsyncSession]` | `Player` | SQLAlchemy, PostgreSQL UPSERT |
| `upsert_player_gameweek_stats()` | 157-249 | `stats_data: Dict, session: Optional[AsyncSession]` | `PlayerGameweekStats` | SQLAlchemy UPSERT |
| `upsert_team()` | 301-391 | `team_data: Dict, session: Optional[AsyncSession]` | `Team` | SQLAlchemy UPSERT |
| `bulk_upsert_players()` | 251-299 | `players_data: List[Dict], batch_size: int = 100` | `Dict[str, int]` | `upsert_player()` |
| `bulk_upsert_gameweek_stats()` | 443-491 | `stats_data: List[Dict], batch_size: int = 100` | `Dict[str, int]` | `upsert_player_gameweek_stats()` |
| `sync_from_fpl_api()` | 493-571 | `fpl_api_service, gameweek, season` | `Dict[str, int]` | `FPLAPIService`, `bulk_upsert_players()` |

**Kullanım Yerleri**:
- `main.py`: `/api/etl/*` endpoint'leri
- `populate_all_tables.py`: Toplu veri yükleme
- `FPLAPIService.save_player_gameweek_stats()`: Tekil oyuncu istatistiği kaydetme

---

### 3. PLEngine (`app/services/ml_engine.py`)

**Amaç**: ML tabanlı puan tahmini, bileşen bazlı modeller

#### Ana Sınıflar

##### XMinsModel
- **Konum**: `app/services/ml_engine.py:48-324`
- **Model**: XGBoost Classifier (fallback: RandomForestClassifier)
- **Amaç**: Başlangıç 11 olasılığı (P_start) ve beklenen dakika tahmini
- **Özellikler**: `days_since_last_match`, `is_cup_week`, `injury_status`, `recent_minutes_avg`

**Metodlar**:
- `extract_features()`: 82-175
- `train()`: 177-218
- `evaluate()`: 220-273
- `predict_start_probability()`: 275-301
- `predict_expected_minutes()`: 303-324

##### AttackModel
- **Konum**: `app/services/ml_engine.py:327-765`
- **Model**: LightGBM Regressor (fallback: RandomForestRegressor)
- **Amaç**: xG ve xA tahmini, rakip xGC normalizasyonu
- **Özellikler**: `opponent_xgc`, `normalized_xg_per_90`, `normalized_xa_per_90`, `fdr`

**Metodlar**:
- `extract_features()`: 369-480
- `train()`: 482-571
- `evaluate()`: 631-692
- `predict()`: 694-765

##### DefenseModel
- **Konum**: `app/services/ml_engine.py:768-1127`
- **Model**: LightGBM Classifier (fallback: RandomForestClassifier)
- **Amaç**: Clean sheet olasılığı (P_CS) ve DefCon puanları

**Metodlar**:
- `train()`: 800-873
- `evaluate()`: 934-996
- `predict_clean_sheet_probability()`: 998-1065
- `calculate_defcon_points()`: 1067-1115

##### PLEngine
- **Konum**: `app/services/ml_engine.py:1130-1989`
- **Amaç**: Tüm bileşen modellerini koordine eden ana motor

**Metodlar**:

| Metod | Satır | Parametreler | Dönüş Değeri | Bağımlılıklar |
|-------|-------|--------------|--------------|---------------|
| `async_load_models()` | 1189-1244 | `model_path: Optional[str]` | - | pickle, asyncio |
| `async_unload_models()` | 1251-1272 | - | - | gc.collect() |
| `calculate_expected_points()` | 1420-1636 | `player_data, fixture_data, fdr_data, team_data, opponent_data, historical_points` | `Dict[str, float]` | `XMinsModel`, `AttackModel`, `DefenseModel`, `DataCleaningService` |
| `train()` | 1638-1693 | `training_data, xmins_features, xmins_labels, attack_features, attack_xg_labels, attack_xa_labels` | - | Tüm alt modeller |
| `predict()` | 1695-1770 | `player_data, historical_points, fixture_data` | `Dict[str, float]` | `calculate_expected_points()` |
| `fit_calibration()` | 1797-1989 | `predicted_points, actual_points, method='linear'` | `Dict[str, float]` | sklearn, numpy |

**xP Hesaplama Formülü**:
```
xP = (xMins/90) * [(Gol_Puanı * xG) + (Asist_Puanı * xA) + (CS_Puanı * xCS) + DefCon_Puanı] + Appearance_Points + Expected_Bonus
```

**Kullanım Yerleri**:
- `main.py`: `/api/predictive/*` endpoint'leri
- `app/scripts/update_predictions.py`: Batch tahmin güncelleme
- `backtesting.py`: Backtest sırasında tahmin üretme

---

### 4. FeatureEngineeringService (`app/services/feature_engineering.py`)

**Amaç**: Özellik mühendisliği, form hesaplama, FDR hesaplama

#### Ana Sınıflar

##### DynamicFormAlpha
- **Konum**: `app/services/feature_engineering.py:23-315`
- **Amaç**: Dinamik form alpha katsayısı optimizasyonu (Bayesian Optimization)

**Metodlar**:
- `calculate_form()`: 38-81
- `optimize_alpha()`: 117-223

##### DixonColesFDR
- **Konum**: `app/services/feature_engineering.py:317-797`
- **Amaç**: Dixon-Coles Poisson modeli ile FDR hesaplama

**Metodlar**:
- `fit()`: 380-450
- `predict_fdr()`: 452-550
- `stochastic_fdr()`: 552-650

##### FeatureEngineeringService
- **Konum**: `app/services/feature_engineering.py:911-1232`
- **Amaç**: Tüm özellik mühendisliği operasyonlarını koordine eder

**Metodlar**:

| Metod | Satır | Parametreler | Dönüş Değeri | Bağımlılıklar |
|-------|-------|--------------|--------------|---------------|
| `calculate_all_features()` | 950-1050 | `player_data, historical_points, fixture_data, position` | `Dict` | `DynamicFormAlpha`, `DixonColesFDR`, `DefConFeatureEngine` |
| `optimize_form_alpha()` | 1052-1100 | `historical_data, lookback_weeks, n_calls` | `Dict` | `DynamicFormAlpha.optimize_alpha()` |
| `fit_fdr()` | 1102-1150 | `fixtures` | `Dict` | `DixonColesFDR.fit()` |

**Kullanım Yerleri**:
- `ml_engine.py`: Özellik çıkarımı için
- `main.py`: `/api/features/*` endpoint'leri
- `backtesting.py`: Backtest sırasında özellik hesaplama

---

### 5. TeamSolver (`app/services/team_solver.py`)

**Amaç**: Multi-period Integer Linear Programming (ILP) ile takım optimizasyonu

#### Ana Metodlar

| Metod | Satır | Parametreler | Dönüş Değeri | Bağımlılıklar |
|-------|-------|--------------|--------------|---------------|
| `create_optimization_model()` | 49-200 | `players, current_squad, locked_players, excluded_players` | `Tuple[pulp.LpProblem, Dict]` | PuLP |
| `solve()` | 202-300 | `players, current_squad, locked_players, excluded_players` | `Dict` | `create_optimization_model()`, PuLP solver |
| `optimize_multi_period()` | 302-435 | `players_by_week, current_squad, budget, free_transfers` | `Dict` | `solve()` |

**Kullanım Yerleri**:
- `main.py`: `/api/solver/optimize-team`, `/team/optimize` endpoint'leri
- `main.py`: `/team/plan` endpoint'i (multi-period planlama)

---

### 6. MarketIntelligenceService (`app/services/market_intelligence.py`)

**Amaç**: Ownership arbitrage analizi, oyuncu sıralama

#### Ana Metodlar

| Metod | Satır | Parametreler | Dönüş Değeri | Bağımlılıklar |
|-------|-------|--------------|--------------|---------------|
| `calculate_player_ranks()` | 38-123 | `db, gameweek, season, use_fpl_api_ownership` | `pd.DataFrame` | `Prediction`, `Player`, `FPLAPIService` |
| `calculate_arbitrage_scores_and_categories()` | 253-333 | `df, overvalued_ownership_threshold, differential_ownership_threshold` | `pd.DataFrame` | - |
| `persist_market_intelligence()` | 335-432 | `db, df, gameweek, season` | `Dict[str, int]` | `MarketIntelligence`, PostgreSQL UPSERT |
| `calculate_and_persist_market_intelligence()` | 434-515 | `db, gameweek, season, use_fpl_api_ownership, ...` | `Dict` | Tüm yukarıdaki metodlar |

**Arbitrage Score Formülü**:
```
arbitrage_score = xp_rank - ownership_rank
- Negatif: Differential (yüksek xP, düşük ownership)
- Pozitif: Overvalued (düşük xP, yüksek ownership)
```

**Kullanım Yerleri**:
- `main.py`: `/market/intelligence` endpoint'i
- `app/scripts/update_predictions.py`: Market intelligence güncelleme

---

### 7. RiskManagementService (`app/services/risk_management.py`)

**Amaç**: Risk analizi, ownership arbitrage, kaptan seçimi, chip timing

#### Ana Sınıflar

- `OwnershipArbitrage`: Ownership bazlı risk analizi
- `CaptainViceCaptainParadox`: Kaptan seçimi optimizasyonu
- `ChipTiming`: Chip (WC, FH, BB) zamanlama analizi
- `RiskManagementService`: Tüm risk servislerini koordine eder

**Kullanım Yerleri**:
- `main.py`: `/api/risk/*` endpoint'leri

---

### 8. BacktestingEngine (`app/services/backtesting.py`)

**Amaç**: ML modellerinin geçmiş veriler üzerinde test edilmesi

**Metodlar**:
- `run_backtest()`: Expanding/rolling window backtest
- `calculate_metrics()`: RMSE, MAE, Spearman correlation

**Kullanım Yerleri**:
- `main.py`: `/api/backtesting/run` endpoint'i
- `run_backtest.py`: Standalone backtest scripti

---

## API Endpoint'leri

### Ana Endpoint Grupları

#### 1. Health & Cache

| Endpoint | Method | Açıklama | Response |
|----------|--------|----------|----------|
| `/` | GET | API bilgisi | `{message, version, status}` |
| `/health` | GET | Health check | `{status, gameweek, cache_status}` |
| `/api/cache/status` | GET | Cache durumu | `{cache_status, gameweek}` |
| `/api/cache/refresh` | POST | Cache yenileme | `{status, gameweek}` |

#### 2. Predictions

| Endpoint | Method | Açıklama | Request Body | Response |
|----------|--------|----------|-------------|------------|
| `/api/predictions/update` | POST | Batch tahmin güncelleme | `{gameweek, season}` | `{status, updated}` |
| `/api/players/all` | GET | Tüm oyuncular ve tahminleri | Query: `gameweek, limit, use_next_gameweek` | `List[PlayerDisplayData]` |
| `/api/dream-team` | GET | Dream team (en iyi 11) | Query: `gameweek` | `DreamTeamResponse` |

#### 3. Feature Engineering

| Endpoint | Method | Açıklama | Request Body | Response |
|----------|--------|----------|-------------|------------|
| `/api/features/optimize-form-alpha` | POST | Form alpha optimizasyonu | `FormAlphaOptimizeRequest` | `FormAlphaResponse` |
| `/api/features/fit-fdr` | POST | FDR model fit | `FDRFitRequest` | `FDRResponse` |
| `/api/features/fdr/{team_name}` | GET | Takım FDR değeri | - | `FDRResponse` |
| `/api/features/stochastic-fdr` | POST | Stochastic FDR | `StochasticFDRRequest` | `StochasticFDRResponse` |
| `/api/features/defcon` | GET | DefCon özellikleri | Query: `player_id, position, minutes` | `DefConFeaturesResponse` |

#### 4. Predictive Engine

| Endpoint | Method | Açıklama | Request Body | Response |
|----------|--------|----------|-------------|------------|
| `/api/predictive/xmins` | POST | xMins tahmini | `XMinsPredictionRequest` | `XMinsPredictionResponse` |
| `/api/predictive/attack` | POST | xG/xA tahmini | `AttackPredictionRequest` | `AttackPredictionResponse` |
| `/api/predictive/defense` | POST | xCS tahmini | `DefensePredictionRequest` | `DefensePredictionResponse` |
| `/api/predictive/comprehensive` | POST | Kapsamlı tahmin | `ComprehensivePredictionRequest` | `ComprehensivePredictionResponse` |

#### 5. Team Optimization

| Endpoint | Method | Açıklama | Request Body | Response |
|----------|--------|----------|-------------|------------|
| `/api/solver/optimize-team` | POST | Tek gameweek optimizasyon | `TeamOptimizationRequest` | `TeamOptimizationResponse` |
| `/team/optimize` | POST | Takım optimizasyon (alternatif) | `TeamOptimizationRequest` | `TeamOptimizationResponse` |
| `/team/plan` | POST | Multi-period planlama | `TeamPlanRequest` | `TeamPlanResponse` |

#### 6. Market Intelligence

| Endpoint | Method | Açıklama | Query Params | Response |
|----------|--------|----------|--------------|----------|
| `/market/intelligence` | GET | Market intelligence verileri | `gameweek, season, category` | `MarketIntelligenceResponse` |

#### 7. Risk Management

| Endpoint | Method | Açıklama | Request Body | Response |
|----------|--------|----------|-------------|------------|
| `/api/risk/ownership-arbitrage` | POST | Ownership arbitrage analizi | `OwnershipArbitrageRequest` | `OwnershipArbitrageResponse` |
| `/api/risk/captain-selection` | POST | Kaptan seçimi analizi | `CaptainSelectionRequest` | `CaptainSelectionResponse` |
| `/api/risk/chip-analysis` | POST | Chip timing analizi | `ChipAnalysisRequest` | `ChipAnalysisResponse` |
| `/api/risk/comprehensive` | POST | Kapsamlı risk analizi | `ComprehensiveRiskAnalysisRequest` | `ComprehensiveRiskAnalysisResponse` |

#### 8. Backtesting

| Endpoint | Method | Açıklama | Request Body | Response |
|----------|--------|----------|-------------|------------|
| `/api/backtesting/run` | POST | Backtest çalıştırma | `BacktestRequest` | `BacktestResponse` |

#### 9. FPL API Proxy

| Endpoint | Method | Açıklama | Query Params | Response |
|----------|--------|----------|--------------|----------|
| `/api/fpl/bootstrap` | GET | FPL bootstrap data | - | `Dict` |
| `/api/fpl/player/{player_id}/history` | GET | Oyuncu geçmişi | - | `List[Dict]` |
| `/api/fpl/fixtures` | GET | Fikstürler | `gameweek` | `List[Dict]` |
| `/api/fpl/fixtures/future` | GET | Gelecek fikstürler | - | `List[Dict]` |

#### 10. ETL Endpoints

| Endpoint | Method | Açıklama | Request Body | Response |
|----------|--------|----------|-------------|------------|
| `/api/etl/sync` | POST | FPL API'den senkronizasyon | `{gameweek, season}` | `Dict` |
| `/api/etl/upsert-player` | POST | Oyuncu UPSERT | `Player data` | `Dict` |
| `/api/etl/upsert-gameweek-stats` | POST | Gameweek istatistikleri UPSERT | `Stats data` | `Dict` |
| `/api/etl/bulk-upsert-players` | POST | Toplu oyuncu UPSERT | `List[Player data]` | `Dict` |
| `/api/etl/status` | GET | ETL durumu | - | `Dict` |

---

## Veritabanı Modelleri

### Core Reference Tables (PRD Schema)

#### Team
- **Tablo**: `teams`
- **Kolonlar**:
  - `id` (PK): Integer
  - `name`: String(255)
  - `short_name`: String(10)
  - `strength_attack`: Integer
  - `strength_defense`: Integer
  - `strength_overall`: Integer
  - `created_at`, `updated_at`: DateTime
- **İlişkiler**: `players`, `home_fixtures`, `away_fixtures`, `team_stats`

#### Player
- **Tablo**: `players`
- **Kolonlar**:
  - `id` (PK): Integer (FPL player ID)
  - `name`: String(255)
  - `position`: String(3) - 'GK', 'DEF', 'MID', 'FWD'
  - `team_id` (FK): Integer → `teams.id`
  - `price`: Numeric(5, 2)
  - `ownership`: Numeric(5, 2)
  - `canonical_name`: String(255)
  - `created_at`, `updated_at`: DateTime
- **İlişkiler**: `team`, `predictions`, `entity_mapping`

#### Fixture
- **Tablo**: `fixtures`
- **Kolonlar**:
  - `id` (PK): Integer
  - `home_team_id` (FK): Integer → `teams.id`
  - `away_team_id` (FK): Integer → `teams.id`
  - `gameweek`: Integer (indexed)
  - `season`: String(9) (indexed)
  - `kickoff_time`: DateTime
  - `finished`: Boolean
  - `home_score`, `away_score`: Integer
  - `fdr_home`, `fdr_away`: Numeric(4, 2)
  - `xgc_home`, `xgc_away`, `xgs_home`, `xgs_away`: Numeric(6, 3)
- **Unique Constraint**: `(home_team_id, away_team_id, gameweek, season)`

#### EntityMapping
- **Tablo**: `entity_mappings`
- **Amaç**: FPL, Understat, FBref arası entity resolution
- **Kolonlar**:
  - `id` (PK): Integer
  - `fpl_id` (FK, Unique): Integer → `players.id`
  - `understat_name`: String(255)
  - `fbref_name`: String(255)
  - `canonical_name`: String(255)
  - `confidence_score`: Numeric(3, 2) - 0.0 to 1.0
  - `manually_verified`: Boolean

#### PlayerStats
- **Tablo**: `player_stats` (TimescaleDB hypertable)
- **Amaç**: Time-series oyuncu istatistikleri
- **Kolonlar**:
  - `id` (PK): Integer
  - `player_id` (FK, indexed): Integer → `players.id`
  - `gameweek` (indexed): Integer
  - `season` (indexed): String(9)
  - `minutes`, `goals`, `assists`, `clean_sheets`, `points`: Integer
  - `xg`, `xa`, `npxg`, `xmins`, `xp`, `defcon_points`: Numeric
  - `timestamp` (indexed): DateTime (TimescaleDB için)
- **Unique Constraint**: `(player_id, gameweek, season)`

#### TeamStats
- **Tablo**: `team_stats` (TimescaleDB hypertable)
- **Amaç**: Time-series takım istatistikleri
- **Kolonlar**:
  - `id` (PK): Integer
  - `team_id` (FK, indexed): Integer → `teams.id`
  - `gameweek` (indexed): Integer
  - `season` (indexed): String(9)
  - `xgc`, `xgs`: Numeric(6, 3)
  - `possession`: Numeric(5, 2)
  - `clean_sheets`, `goals_conceded`: Integer
  - `timestamp` (indexed): DateTime

### Prediction & ML Tables

#### Prediction
- **Tablo**: `predictions`
- **Amaç**: Batch prediction sonuçları (hızlı API okuma için)
- **Kolonlar**:
  - `id` (PK): Integer
  - `fpl_id` (indexed): Integer
  - `gameweek` (indexed): Integer
  - `season` (indexed): String
  - `xp`: Float (expected_points)
  - `xg`, `xa`, `xmins`, `xcs`, `defcon_score`: Float
  - `confidence_score`: Float (0.0-1.0)
  - `model_version`: String
  - `calculated_at`, `updated_at`: DateTime
  - `player_id` (FK): Integer → `players.id`
- **Unique Constraint**: `(fpl_id, gameweek, season)`

#### MarketIntelligence
- **Tablo**: `market_intelligence`
- **Amaç**: Ownership arbitrage analizi sonuçları
- **Kolonlar**:
  - `id` (PK): Integer
  - `player_id` (FK, indexed): Integer → `players.id`
  - `gameweek` (indexed): Integer
  - `season` (indexed): String(9)
  - `xp_rank`: Integer (descending: highest xP = rank 1)
  - `ownership_rank`: Integer (descending: highest ownership = rank 1)
  - `arbitrage_score`: Float (xp_rank - ownership_rank)
  - `category`: String(50) - 'Differential', 'Overvalued', 'Neutral'
- **Unique Constraint**: `(player_id, gameweek, season)`

#### PlayerGameweekStats
- **Tablo**: `player_gameweek_stats`
- **Amaç**: Detaylı gameweek istatistikleri (FPL API'den)
- **Kolonlar**: `fpl_id`, `gameweek`, `season`, `minutes`, `goals`, `assists`, `clean_sheets`, `total_points`, `normalized_points`, `xg`, `xa`, `xgi`, `xgc`, `ict_index`, `blocks`, `interventions`, `passes`, `defcon_floor_points`, vb.
- **Unique Constraint**: `(fpl_id, gameweek, season)`

### Model Performance Tables

#### ModelPerformance
- **Tablo**: `model_performance`
- **Kolonlar**: `model_version`, `gameweek`, `mae`, `rmse`, `accuracy`

#### BacktestResult
- **Tablo**: `backtest_results`
- **Kolonlar**: `model_version`, `methodology`, `season`, `gameweek`, `rmse`, `mae`, `spearman_corr`, `n_predictions`

#### BacktestSummary
- **Tablo**: `backtest_summary`
- **Kolonlar**: `model_version`, `methodology`, `season`, `total_weeks_tested`, `overall_rmse`, `overall_mae`, `overall_spearman_corr`, `r_squared`

---

## Veri Akışı (Data Flow)

### 1. FPL API → Database Flow

```
FPL Official API
    ↓
FPLAPIService.get_bootstrap_data()
    ↓ (Rate Limiting + Cache)
FPLAPIService.extract_players_from_bootstrap()
    ↓
ETLService.upsert_player()
    ↓
PostgreSQL (players table)
```

**Gameweek Stats Flow**:
```
FPLAPIService.get_player_data(player_id)
    ↓
FPLAPIService.extract_player_history()
    ↓ (DGW normalization)
FPLAPIService.fetch_fbref_defcon_metrics()
    ↓ (Entity resolution)
ETLService.upsert_player_gameweek_stats()
    ↓
PostgreSQL (player_gameweek_stats table)
```

### 2. Database → ML Engine Flow

```
PostgreSQL (player_gameweek_stats, players, fixtures)
    ↓
FeatureEngineeringService.calculate_all_features()
    ↓ (Form alpha, FDR, DefCon features)
PLEngine.calculate_expected_points()
    ↓
    ├─→ XMinsModel.predict_expected_minutes()
    ├─→ AttackModel.predict() (xG/xA)
    └─→ DefenseModel.predict_clean_sheet_probability()
    ↓
xP Calculation:
    xP = (xMins/90) * [(Goal_Points * xG) + (Assist_Points * xA) + (CS_Points * xCS) + DefCon_Points]
         + Appearance_Points + Expected_Bonus
    ↓
PLEngine.predict()
    ↓
PostgreSQL (predictions table) - Batch Prediction
```

### 3. ML Engine → Frontend API Flow

```
Frontend Request: GET /api/players/all?gameweek=5
    ↓
main.py.get_all_players()
    ↓
Database Query: SELECT * FROM predictions WHERE gameweek=5
    ↓ (Fast read, no ML computation)
JOIN players, teams
    ↓
Response: List[PlayerDisplayData]
    ↓
Frontend (Next.js)
```

### 4. Market Intelligence Flow

```
Predictions Table (xp values)
    ↓
MarketIntelligenceService.calculate_player_ranks()
    ↓
    ├─→ xP Rank (descending)
    └─→ Ownership Rank (from FPL API or DB)
    ↓
MarketIntelligenceService.calculate_arbitrage_scores_and_categories()
    ↓
    arbitrage_score = xp_rank - ownership_rank
    category = 'Differential' | 'Overvalued' | 'Neutral'
    ↓
MarketIntelligenceService.persist_market_intelligence()
    ↓
PostgreSQL (market_intelligence table)
    ↓
Frontend: GET /market/intelligence
```

### 5. Team Optimization Flow

```
Frontend: POST /team/optimize
    ↓
main.py.optimize_team()
    ↓
TeamSolver.create_optimization_model()
    ↓ (ILP with PuLP)
    ├─→ Decision Variables
    ├─→ Constraints (budget, squad structure, max per team)
    └─→ Objective: Maximize expected points
    ↓
PuLP Solver (CBC/GLPK)
    ↓
TeamSolver.solve()
    ↓
Response: TeamOptimizationResponse (squad, starting_xi, expected_points)
```

### Sequence Diagram: Prediction Update Workflow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Scheduler │     │  FPL API     │     │  ML Engine  │     │  PostgreSQL  │
│  (Cron Job) │     │   Service    │     │   (PLEngine) │     │              │
└──────┬──────┘     └──────┬───────┘     └──────┬──────┘     └──────┬───────┘
       │                   │                     │                    │
       │ 1. Trigger        │                     │                    │
       ├──────────────────→│                     │                    │
       │                   │                     │                    │
       │                   │ 2. get_bootstrap()  │                    │
       │                   ├─────────────────────→│                    │
       │                   │                     │                    │
       │                   │ 3. get_player_data() │                    │
       │                   ├─────────────────────→│                    │
       │                   │                     │                    │
       │                   │ 4. Save to DB       │                    │
       │                   ├─────────────────────────────────────────→│
       │                   │                     │                    │
       │                   │ 5. Load training data│                   │
       │                   │←────────────────────────────────────────┤
       │                   │                     │                    │
       │                   │ 6. calculate_features()                  │
       │                   ├─────────────────────→│                    │
       │                   │                     │                    │
       │                   │ 7. predict()         │                    │
       │                   ├─────────────────────→│                    │
       │                   │                     │                    │
       │                   │ 8. Save predictions │                    │
       │                   ├─────────────────────────────────────────→│
       │                   │                     │                    │
       │                   │ 9. Update cache     │                    │
       │                   │←─────────────────────┘                    │
```

---

## Frontend Yapısı

### Next.js App Router Yapısı

#### Sayfalar

| Route | Dosya | Açıklama |
|-------|-------|----------|
| `/` | `app/page.tsx` | Dashboard (ana sayfa) |
| `/players` | `app/players/page.tsx` | Tüm oyuncular listesi |
| `/team-optimizer` | `app/team-optimizer/page.tsx` | Takım optimizasyon sayfası |
| `/team-planner` | `app/team-planner/page.tsx` | Multi-period planlama |
| `/market-intelligence` | `app/market-intelligence/page.tsx` | Market intelligence sayfası |
| `/dream-team` | `app/dream-team/page.tsx` | Dream team görüntüleme |
| `/model-performance` | `app/model-performance/page.tsx` | Model performans metrikleri |

#### API Routes (Next.js Server Actions)

| Route | Dosya | Açıklama |
|-------|-------|----------|
| `/api/players/all` | `app/api/players/route.ts` | Oyuncu listesi API |
| `/api/team/optimize` | `app/api/team/route.ts` | Takım optimizasyon API |
| `/api/market/intelligence` | `app/api/market/intelligence/route.ts` | Market intelligence API |
| `/api/dream-team` | `app/api/dream-team/route.ts` | Dream team API |

### Frontend → Backend Communication

```
Next.js Frontend (Client Component)
    ↓ (fetch)
Next.js API Route (Server Action)
    ↓ (HTTP request)
FastAPI Backend (/api/*)
    ↓ (Database query)
PostgreSQL
    ↓ (Response)
FastAPI Response
    ↓
Next.js API Route
    ↓
Frontend Component (React)
```

---

## Bağımlılıklar ve Çevre Değişkenleri

### Environment Variables

| Değişken | Açıklama | Varsayılan |
|----------|----------|------------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://fpl_user:fpl_password@localhost:5432/fpl_db` |
| `FPL_EMAIL` | FPL API email | - |
| `FPL_PASSWORD` | FPL API password | - |
| `SECRET_KEY` | Application secret key | - |
| `POSTGRES_USER` | PostgreSQL kullanıcı adı | `fpl_user` |
| `POSTGRES_PASSWORD` | PostgreSQL şifre | `fpl_password` |
| `POSTGRES_DB` | PostgreSQL veritabanı adı | `fpl_db` |

### Python Bağımlılıkları (requirements.txt)

**Core**:
- `fastapi`, `uvicorn`, `sqlalchemy`, `alembic`
- `pandas`, `numpy`, `scipy`

**ML**:
- `xgboost`, `lightgbm`, `scikit-learn`
- `tensorflow` (opsiyonel, LSTM için)

**Utilities**:
- `httpx`, `beautifulsoup4`, `fuzzywuzzy`
- `pulp` (ILP solver)
- `python-dotenv`

### Docker Compose Servisleri

| Servis | Image | Port | Memory Limit |
|--------|-------|------|--------------|
| `db` | `timescale/timescaledb:latest-pg15` | 5432 | 512MB |
| `backend` | Custom (FastAPI) | 8000 | 1536MB |
| `frontend` | Custom (Next.js) | 3000 | 1GB |

---

## Önemli Notlar

### Cache Stratejisi

- **Bootstrap Data**: 24 saat TTL
- **Player Data**: 1 saat TTL
- **In-Memory Cache**: 10 dakika TTL (gameweek bazlı)
- **Rate Limiting**: 60 req/min (DefCon rules)

### Batch Prediction Architecture

1. ML hesaplamaları arka planda çalışır (scheduler veya manuel trigger)
2. Sonuçlar `predictions` tablosuna kaydedilir
3. API endpoint'leri sadece DB'den okur (hızlı yanıt)
4. Fallback: Eğer prediction yoksa, temel veri döner

### Memory Management

- ML modelleri lazy loading ile yüklenir
- Async model loading/unloading
- Garbage collection after model operations
- Memory limits enforced via Docker

### Error Handling (DefCon Rules)

- Exponential backoff on API errors
- Graceful degradation (fallback to cached/basic data)
- Health checks for service dependencies
- Resource constraint awareness

---

## Sonuç

Bu dokümantasyon, FPL Prediction Platform'un teknik referans haritasını içermektedir. Tüm servisler, fonksiyonlar, endpoint'ler ve veri akışları detaylı olarak dokümante edilmiştir.

**Güncelleme Tarihi**: 2025-01-XX
**Versiyon**: 2.0.0
