# AFL Fantasy Analytics Platform

End-to-end stack: **FastAPI** (ETL, feature engineering, boosted regression, PuLP + Monte Carlo trades) and **Next.js** UI.

## Prerequisites

- Python 3.10+ recommended (3.9 works; CI should pin 3.11+).
- Node.js 18+ and npm for the frontend.
- **Optional (macOS):** `brew install libomp` so **XGBoost** loads. If OpenMP is missing, training automatically uses **scikit-learn `HistGradientBoostingRegressor`** with the same feature pipeline and artifact layout.

## Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/seed_data.py    # synthetic source (explicitly marked in code)
python scripts/train_model.py
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- SQLite database: `backend/afl_fantasy.db` (override with `DATABASE_URL` for PostgreSQL, e.g. `postgresql+asyncpg://...`).
- Artifacts: `backend/models_artifact/` (`metadata.json`, `boosted_*.json` or `boosted_hgbr.joblib`, `linear_baseline.joblib`).

### API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness + model loaded |
| GET | `/players` | Projections + intervals + value |
| POST | `/predict` | Subset projection |
| POST | `/team-evaluate` | Sum projection + band |
| POST | `/trade-recommend` | MIP + Monte Carlo trade set |
| POST | `/ingest-news` | NLP availability signals |

## Frontend

```bash
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 npm run dev
```

Open `http://localhost:3000`.

## Data sources

`StatSource` in `app/data/sources/` is the extension point: plug APIs, CSV, or scrapers. The default seed uses **`SyntheticStatSource`** (simulated AFL-style stats; labeled in module docstring).

## Daily job

```bash
cd backend && python scripts/daily_update.py
```

Wire to cron or APScheduler in production.
