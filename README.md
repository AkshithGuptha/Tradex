# Tradex — RL Trading Strategy Lab

Educational simulation-only trading dashboard using a tabular Q-learning agent to choose Buy / Sell / Hold actions on market data.

> Research demo only, not investment advice. No broker connection or real orders are supported.

## Stack
- React + Vite + Recharts frontend
- FastAPI + pandas + NumPy backend
- Tabular Q-learning agent
- Synthetic OHLCV sample data and CSV upload support

## Run backend
```bash
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
API docs: http://localhost:8000/docs

## Run frontend
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173.

## API routes
- GET /api/health
- GET /api/market
- POST /api/train
- POST /api/backtest
- GET /api/trades
- GET /api/portfolio
- POST /api/market/upload

Backtests are historical simulations, not forecasts.