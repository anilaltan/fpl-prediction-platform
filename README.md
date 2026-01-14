# FPL Point Prediction Platform

A professional SaaS platform for Fantasy Premier League point prediction using Machine Learning, built with FastAPI and Next.js.

## 🏗️ Architecture

- **Backend**: FastAPI (Python 3.11-slim) with ML engine
- **Frontend**: Next.js 14 (App Router) with Tailwind CSS
- **Database**: PostgreSQL 15
- **Orchestration**: Docker Compose

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- At least 2GB RAM available

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd fpl-prediction-platform
   ```

2. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. **Start the services**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## 📁 Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI application
│   │   ├── models.py        # Database models
│   │   ├── schemas.py       # Pydantic schemas
│   │   ├── database.py      # Database configuration
│   │   └── services/
│   │       ├── ml_engine.py # ML prediction engine
│   │       └── fpl_api.py   # FPL API service
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx         # Landing page
│   │   └── globals.css
│   ├── Dockerfile
│   └── package.json
├── docker-compose.yml
├── .env.example
└── .cursorrules
```

## 🔧 Development

### Backend Development
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Development
```bash
cd frontend
npm install
npm run dev
```

## 🧠 ML Engine

The ML engine implements "Moneyball" principles:
- Statistical analysis of player performance
- Feature engineering based on FPL metrics
- Predictive modeling for point forecasts
- Confidence scoring for predictions

## 📊 Database

PostgreSQL is used for storing:
- Player data and statistics
- Prediction history
- Model performance metrics

## 🔒 Security

- Environment variables for sensitive data
- CORS configuration for API access
- Rate limiting (DefCon rules)
- Input validation with Pydantic

## 📝 License

[Your License Here]
