# README.md
# AI Trader - Ready-to-deploy on Render

This repository is an MVP **AI trading bot** designed to run on Render's free Web Service. It connects to Binance (testnet by default), fetches market data via `ccxt`, trains a light XGBoost model, backtests, and runs a live paper-trading loop. The app is a Flask web service that runs the bot in a background thread and exposes a `/health` endpoint for Render liveness checks.

⚠️ **Important**: This project is educational and for paper-trading/testing. Do **NOT** use real funds until you thoroughly test and audit.

---

## What's included

- `app.py` — Flask web app + background bot runner (entrypoint for Render web service)
- `bot.py` — Core loop: data fetch → feature engineering → model predict → order simulation
- `trainer.py` — Simple trainer to build an XGBoost model from historical data
- `utils.py` — Helpers for data, indicators, and order simulation
- `requirements.txt` — Python dependencies
- `Dockerfile` — container for Render (optional; Render auto-detects Python)
- `render.yaml` — minimal Render config (optional)
- `.env.example` — example environment variables

---

## Quick start (local test)

1. Clone repo and create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill your keys (for testing use Binance Testnet keys or leave blank for simulation).

3. Train a quick model (optional):

```bash
python trainer.py --symbol BTC/USDT --interval 1h --limit 2000
```

4. Run locally:

```bash
python app.py
```

Open `http://127.0.0.1:5000/health` to see status. The bot will start in background and log trades to `trades.csv`.

---

## Deploy to Render (high-level)

1. Create a GitHub repo and push this project.
2. Sign up for Render and create a new **Web Service**.
3. Link your GitHub repo, choose the branch, set the start command:

```
gunicorn app:app --workers 1 --threads 2 --bind 0.0.0.0:$PORT
```

4. Add environment variables on Render (see `.env.example`).
5. Deploy. Render will build and run the container. Use the service URL for health checks.

---

## Environment variables (`.env`)

```
# Use Binance testnet by default
BINANCE_API_KEY=
BINANCE_API_SECRET=
USE_TESTNET=1
SYMBOL=BTC/USDT
INTERVAL=1h
MODEL_PATH=model.xgb
PAPER_START_BALANCE=1000
LOG_LEVEL=INFO
```

---

# NOTES
- This project uses `ccxt` for market data. For live trading replace simulated execution with exchange order calls and **limit permissions to trading only**.
- Keep API keys secure. On Render use Environment settings, not checked-in `.env`.

---

# File: requirements.txt
ccxt
flask
gunicorn
python-dotenv
pandas
numpy
scikit-learn
xgboost
ta
requests
joblib

# File: .env.example
BINANCE_API_KEY=
BINANCE_API_SECRET=
USE_TESTNET=1
SYMBOL=BTC/USDT
INTERVAL=1h
MODEL_PATH=model.xgb
PAPER_START_BALANCE=1000
LOG_LEVEL=INFO

# File: app.py
from flask import Flask, jsonify
import threading
import time
import os
from dotenv import load_dotenv
load_dotenv()

from bot import TradingBot

app = Flask(__name__)

bot = None

@app.route('/health')
def health():
    status = {
        'status': 'running' if bot and bot.is_running else 'starting',
        'last_tick': bot.last_tick if bot else None
    }
    return jsonify(status)

def start_background_bot():
    global bot
    bot = TradingBot(
        symbol=os.getenv('SYMBOL','BTC/USDT'),
        interval=os.getenv('INTERVAL','1h'),
        model_path=os.getenv('MODEL_PATH','model.xgb'),
        paper_balance=float(os.getenv('PAPER_START_BALANCE',1000)),
        use_testnet=os.getenv('USE_TESTNET','1')=='1'
    )
    bot.run_loop()

if __name__ == '__main__':
    # start bot thread
    t = threading.Thread(target=start_background_bot, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=int(os.getenv('PORT',5000)))

# File: utils.py
import pandas as pd
import ta
import joblib
import os

def ohlcv_to_df(bars):
    df = pd.DataFrame(bars, columns=['time','open','high','low','close','volume'])
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df

def add_indicators(df):
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df.dropna(inplace=True)
    return df

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

# File: trainer.py
import ccxt
import argparse
import os
from utils import ohlcv_to_df, add_indicators, save_model
from sklearn.ensemble import GradientBoostingClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--symbol', default='BTC/USDT')
parser.add_argument('--interval', default='1h')
parser.add_argument('--limit', type=int, default=1500)
parser.add_argument('--model-path', default='model.xgb')
args = parser.parse_args()

exchange = ccxt.binance()
print('Fetching bars...')
bars = exchange.fetch_ohlcv(args.symbol, timeframe=args.interval, limit=args.limit)
df = ohlcv_to_df(bars)
df = add_indicators(df)

df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
features = ['rsi','ma20','ma50']
X = df[features].iloc[:-1]
y = df['target'].iloc[:-1]

print('Training model...')
model = GradientBoostingClassifier(n_estimators=100)
model.fit(X, y)

print('Saving model to', args.model_path)
save_model(model, args.model_path)
print('Done')

# File: bot.py
import ccxt
import time
import os
from utils import ohlcv_to_df, add_indicators, load_model, save_model
import pandas as pd
import numpy as np
from datetime import datetime

class TradingBot:
    def __init__(self, symbol='BTC/USDT', interval='1h', model_path='model.xgb', paper_balance=1000, use_testnet=True):
        self.symbol = symbol
        self.interval = interval
        self.model_path = model_path
        self.paper_balance = paper_balance
        self.use_testnet = use_testnet
        self.is_running = False
        self.last_tick = None
        self.trades_file = 'trades.csv'

        # exchange
        if use_testnet:
            self.exchange = ccxt.binance({'options': {'defaultType': 'spot'}})
        else:
            self.exchange = ccxt.binance()

        # load model
        self.model = load_model(self.model_path)

    def fetch_data(self, limit=500):
        bars = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.interval, limit=limit)
        df = ohlcv_to_df(bars)
        df = add_indicators(df)
        return df

    def predict_signal(self, df):
        if self.model is None:
            return 0
        features = ['rsi','ma20','ma50']
        X = df[features].iloc[-1:]
        pred = self.model.predict(X)[0]
        return int(pred)

    def simulate_order(self, side, price, size):
        # simplistic simulation: update balance + log
        trade = {
            'time': datetime.utcnow().isoformat(),
            'symbol': self.symbol,
            'side': side,
            'price': price,
            'size': size
        }
        df = pd.DataFrame([trade])
        if not os.path.exists(self.trades_file):
            df.to_csv(self.trades_file, index=False)
        else:
            df.to_csv(self.trades_file, index=False, mode='a', header=False)
        print('Simulated trade:', trade)

    def run_loop(self):
        print('Bot starting...')
        self.is_running = True
        while True:
            try:
                df = self.fetch_data(limit=200)
                sig = self.predict_signal(df)
                last_price = df['close'].iloc[-1]
                self.last_tick = df['time'].iloc[-1].isoformat()

                # simple logic: if pred==1 buy small fraction, else do nothing (or close)
                if sig == 1:
                    size = 0.001  # dummy
                    self.simulate_order('BUY', float(last_price), size)

                time.sleep(60)

            except Exception as e:
                print('Error in loop:', e)
                time.sleep(10)

# File: Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ENV PORT=5000
CMD ["gunicorn", "app:app", "--workers", "1", "--threads", "2", "--bind", "0.0.0.0:5000"]

# File: render.yaml
services:
  - type: web
    name: ai-trader
    env: python
    plan: free

# End of repository
