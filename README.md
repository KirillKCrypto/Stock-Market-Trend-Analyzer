Stock Market Regime Detection & Backtesting System
https://img.shields.io/badge/python-3.8+-blue.svg
https://img.shields.io/badge/LightGBM-3.3+-green.svg
https://img.shields.io/badge/License-MIT-yellow.svg

A complete pipeline for market regime classification and strategy backtesting.
The system downloads historical stock data, engineers over 50 technical features, detects market regimes (Bull/Neutral/Bear) using Gaussian Mixture Models, trains a two-stage LightGBM classifier, and backtests a realistic trading strategy with risk management.

Features
📥 Automated data download from Yahoo Finance via yfinance.

🧠 Feature engineering – momentum, volatility, volume, technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.).

🔍 Regime detection – unsupervised clustering (GMM) to label Bull/Neutral/Bear regimes.

🤖 Two-stage ML model:

trend model – predicts market regime (Bull/Neutral/Bear).

direction model – predicts Long/Short within trending regimes.

📈 Backtesting engine with realistic constraints:

ATR-based stop‑loss, take‑profit, trailing stop.

Partial profit‑taking.

Position sizing based on risk per trade and volatility.

Commission and slippage simulation.

📊 Visualization – regime overlays on price charts, equity curves, drawdowns.

🧪 Multi‑asset backtesting – run the same strategy across multiple tickers.

🚀 End‑to‑end pipeline – one command (python setup.py) runs everything.

Project Structure
text
.
├── config.py                 # User configuration (ticker, period)
├── data_acquisition.py       # Download OHLCV data from Yahoo Finance
├── model_training.py         # Feature engineering, regime detection, model training
├── backtest.py               # Backtesting engine and trade simulation
├── setup.py                  # Orchestrates the complete pipeline
├── requirements.txt          # Python dependencies
└── README.md                 # This file
Installation
Clone the repository

bash
git clone https://github.com/yourusername/stock-regime-backtest.git
cd stock-regime-backtest
Create a virtual environment (recommended)

bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
Install dependencies

bash
pip install -r requirements.txt
If you don’t have a requirements.txt, install manually:

bash
pip install yfinance pandas numpy lightgbm scikit-learn matplotlib scipy joblib
Configuration
Edit config.py to set your desired ticker and data period:

python
ticker = "AAPL"        # Any valid Yahoo Finance symbol
period = "10y"         # Data period (must be ≥5 years for meaningful regimes)
period accepts strings like "5y", "10y", "max", or specific dates (see yfinance docs).

Usage
Run the full pipeline
bash
python setup.py
This will:

Create the directory structure (<ticker>/, models/, plots/, <ticker>_backtest/).

Download historical data and save as <ticker>/<ticker>_data.csv.

Engineer features, detect regimes, train the two‑stage model, and save:

Trained model: <ticker>/models/<ticker>_trend_analyzer.pkl

Predictions: <ticker>/<ticker>_predictions.csv

Regime plot: <ticker>/plots/<ticker>_regimes.png

Run the backtest and save:

Equity curve: <ticker>/<ticker>_backtest/equity_curve.csv

Trade log: <ticker>/<ticker>_backtest/results.csv

Equity plot: <ticker>/plots/<ticker>_equity.png

Run individual modules
Download data only

bash
python data_acquisition.py
Train model only (requires existing data CSV)

bash
python model_training.py
Run backtest only (requires trained model)

bash
python backtest.py
Multi‑asset backtest
Edit the list of tickers at the bottom of backtest.py and run:

bash
python backtest.py
How It Works
1. Data Acquisition (data_acquisition.py)
Downloads OHLCV data using yfinance, flattens multi‑index columns, and saves a clean CSV with columns open, high, low, close, volume.

2. Feature Engineering & Regime Detection (model_training.py)
Features – Over 50 features are created:

Returns over multiple horizons (1,3,5,10,21,63 days).

Price relative to moving averages (20,50,200).

Volatility (5,20,50 days), volatility ratio, acceleration.

ATR, Bollinger Bands, MACD, RSI, ADX, MFI, volume ratios, etc.

Regime detection – Unsupervised GMM clusters rolling windows of return‑based features into 3 regimes. Regimes are then mapped to Bull, Neutral, Bear based on Sharpe ratio (best → Bull, worst → Bear).

Two‑stage model:

Trend model – predicts the 3‑class regime (Bull/Neutral/Bear).
Direction model – trained only on trending periods (Bull & Bear) to predict Long (1) vs Short (2).
Feature selection – Top 40 features are selected using LightGBM importance to reduce noise.

3. Backtesting Engine (backtest.py)
The strategy is simple:

When the direction model predicts Long (1) → enter long.

When it predicts Short (2) → enter short.

Neutral (0) → stay out.

Risk management:

Position size = (risk_per_trade * cash) / (ATR * 2) (volatility‑adjusted).

Maximum position size limited to max_position_pct of cash.

Stop‑loss = entry price ± atr_mult_sl * ATR.

Take‑profit = entry price ± atr_mult_tp * ATR.

Trailing stop = best price ± atr_mult_trail * ATR.

Partial take‑profit at partial_tp_mult * ATR for half the position.

Max hold days (exit if held longer).

Signal flip exit (if signal changes from Long to Short or vice‑versa).

Optional trend filter: do not short if price is above 200‑day MA.

All parameters are configurable in the backtest_advanced() call (see setup.py for example).

Key Configuration Parameters (Backtest)
Parameter	Description	Default
capital	Starting capital	10_000
risk_per_trade	Fraction of capital risked per trade (based on stop‑loss distance)	0.01
max_position_pct	Maximum fraction of capital allocated to one position	0.30
atr_mult_sl	Stop‑loss distance in ATR units	2.0
atr_mult_tp	Take‑profit distance in ATR units	3.5
atr_mult_trail	Trailing stop distance in ATR units (0 to disable)	1.5
partial_tp_ratio	Fraction of position to close at partial take‑profit	0.5
partial_tp_mult	Partial take‑profit distance in ATR units	2.0
min_hold_days	Minimum holding days before any exit (except stop‑loss)	1
max_hold_days	Maximum holding days (force exit)	30
use_trend_filter	Do not short when price > 200‑day MA	True
signal_flip_exit	Exit if model signal flips from Long to Short (or vice‑versa)	True
commission	Round‑trip commission fraction (0.001 = 0.1%)	0.001
Output & Interpretation
After running setup.py, you will find:

<ticker>_data.csv – Raw OHLCV data.

<ticker>_predictions.csv – Contains close, true regime (trend), predicted regime (trend_pred), predicted direction (direction_pred), and final regime (final_regime where 1=Long, 2=Short, 0=Neutral).

<ticker>/plots/<ticker>_regimes.png – Price chart with colored backgrounds for true (top) and predicted (bottom) regimes.

<ticker>/plots/<ticker>_equity.png – Equity curve, drawdown, and price with trade markers.

<ticker>/<ticker>_backtest/results.csv – Detailed trade log (entry/exit dates, prices, PnL, reason, etc.).

<ticker>/<ticker>_backtest/equity_curve.csv – Daily equity values.

Sample trade log columns
entry_date, exit_date

direction (Long/Short)

entry_price, exit_price

size

pnl (net after commission)

hold_days

reason (SL, TP, PartialTP, SignalFlip, MaxHold, End)

Customization & Extending
Add new features – Edit create_features_optimized() in model_training.py.

Change model architecture – Modify TREND_PARAMS / DIR_PARAMS or replace LightGBM with another classifier.

Adjust regime detection – Tune n_regimes, window_years, step_days in CONFIG.

Modify backtest logic – Extend TradeManager class in backtest.py.

Dependencies
Python ≥ 3.8

yfinance

pandas

numpy

lightgbm

scikit-learn

matplotlib

scipy

joblib

License
This project is licensed under the MIT License – see the LICENSE file for details.

Disclaimer
This software is for educational and research purposes only.
It is not financial advice. Past performance does not guarantee future results. Use at your own risk.

Happy backtesting! 🚀

This response is AI-generated, for reference only.
Можешь скинуть полную MD разметку для вставки
We need to provide the full markdown content for the README.md. The user asked: "Можешь скинуть полную MD разметку для вставки". So we need to output the markdown code that they can copy and paste into their README.md file. We have already generated a README in the previous response. We'll present that as a code block with markdown syntax.

But we must ensure it's the full markdown. Let's check the previous response: it's a well-structured README with headings, lists, code blocks, etc. We'll just output that content within a triple backtick block with language "markdown" or just plain text? Usually we just output the raw markdown without additional backticks around it because they want to copy it. However, to clearly indicate that it's markdown, we can put it in a code block with markdown annotation. But that would render as a code block in the final answer, not as markdown. The user wants to insert it into their README.md file, so they need the raw markdown text. We can output it as plain text, but it's better to provide it inside a code block to avoid any formatting issues. The user can then copy the code block content and paste. We'll provide the content within triple backticks with language "markdown" so it's clear.

Thus, final answer: a code block containing the full markdown content.

markdown
# Stock Market Regime Detection & Backtesting System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3+-green.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete pipeline for **market regime classification** and **strategy backtesting**.  
The system downloads historical stock data, engineers over 50 technical features, detects market regimes (Bull/Neutral/Bear) using Gaussian Mixture Models, trains a two-stage LightGBM classifier, and backtests a realistic trading strategy with risk management.

## Features

- 📥 **Automated data download** from Yahoo Finance via `yfinance`.
- 🧠 **Feature engineering** – momentum, volatility, volume, technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.).
- 🔍 **Regime detection** – unsupervised clustering (GMM) to label Bull/Neutral/Bear regimes.
- 🤖 **Two-stage ML model**:
  - `trend` model – predicts market regime (Bull/Neutral/Bear).
  - `direction` model – predicts Long/Short within trending regimes.
- 📈 **Backtesting engine** with realistic constraints:
  - ATR-based stop‑loss, take‑profit, trailing stop.
  - Partial profit‑taking.
  - Position sizing based on risk per trade and volatility.
  - Commission and slippage simulation.
- 📊 **Visualization** – regime overlays on price charts, equity curves, drawdowns.
- 🧪 **Multi‑asset backtesting** – run the same strategy across multiple tickers.
- 🚀 **End‑to‑end pipeline** – one command (`python setup.py`) runs everything.

## Project Structure
.
├── config.py # User configuration (ticker, period)
├── data_acquisition.py # Download OHLCV data from Yahoo Finance
├── model_training.py # Feature engineering, regime detection, model training
├── backtest.py # Backtesting engine and trade simulation
├── setup.py # Orchestrates the complete pipeline
├── requirements.txt # Python dependencies
└── README.md # This file

text

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/stock-regime-backtest.git
   cd stock-regime-backtest
Create a virtual environment (recommended)

bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
Install dependencies

bash
pip install -r requirements.txt
If you don’t have a requirements.txt, install manually:

bash
pip install yfinance pandas numpy lightgbm scikit-learn matplotlib scipy joblib
Configuration
Edit config.py to set your desired ticker and data period:

python
ticker = "AAPL"        # Any valid Yahoo Finance symbol
period = "10y"         # Data period (must be ≥5 years for meaningful regimes)
period accepts strings like "5y", "10y", "max", or specific dates (see yfinance docs).

Usage
Run the full pipeline
bash
python setup.py
This will:

Create the directory structure (<ticker>/, models/, plots/, <ticker>_backtest/).

Download historical data and save as <ticker>/<ticker>_data.csv.

Engineer features, detect regimes, train the two‑stage model, and save:

Trained model: <ticker>/models/<ticker>_trend_analyzer.pkl

Predictions: <ticker>/<ticker>_predictions.csv

Regime plot: <ticker>/plots/<ticker>_regimes.png

Run the backtest and save:

Equity curve: <ticker>/<ticker>_backtest/equity_curve.csv

Trade log: <ticker>/<ticker>_backtest/results.csv

Equity plot: <ticker>/plots/<ticker>_equity.png

Run individual modules
Download data only

bash
python data_acquisition.py
Train model only (requires existing data CSV)

bash
python model_training.py
Run backtest only (requires trained model)

bash
python backtest.py
Multi‑asset backtest
Edit the list of tickers at the bottom of backtest.py and run:

bash
python backtest.py
How It Works
1. Data Acquisition (data_acquisition.py)
Downloads OHLCV data using yfinance, flattens multi‑index columns, and saves a clean CSV with columns open, high, low, close, volume.

2. Feature Engineering & Regime Detection (model_training.py)
Features – Over 50 features are created:

Returns over multiple horizons (1,3,5,10,21,63 days).

Price relative to moving averages (20,50,200).

Volatility (5,20,50 days), volatility ratio, acceleration.

ATR, Bollinger Bands, MACD, RSI, ADX, MFI, volume ratios, etc.

Regime detection – Unsupervised GMM clusters rolling windows of return‑based features into 3 regimes. Regimes are then mapped to Bull, Neutral, Bear based on Sharpe ratio (best → Bull, worst → Bear).

Two‑stage model:

Trend model – predicts the 3‑class regime (Bull/Neutral/Bear).
Direction model – trained only on trending periods (Bull & Bear) to predict Long (1) vs Short (2).
Feature selection – Top 40 features are selected using LightGBM importance to reduce noise.

3. Backtesting Engine (backtest.py)
The strategy is simple:

When the direction model predicts Long (1) → enter long.

When it predicts Short (2) → enter short.

Neutral (0) → stay out.

Risk management:

Position size = (risk_per_trade * cash) / (ATR * 2) (volatility‑adjusted).

Maximum position size limited to max_position_pct of cash.

Stop‑loss = entry price ± atr_mult_sl * ATR.

Take‑profit = entry price ± atr_mult_tp * ATR.

Trailing stop = best price ± atr_mult_trail * ATR.

Partial take‑profit at partial_tp_mult * ATR for half the position.

Max hold days (exit if held longer).

Signal flip exit (if signal changes from Long to Short or vice‑versa).

Optional trend filter: do not short if price is above 200‑day MA.

All parameters are configurable in the backtest_advanced() call (see setup.py for example).

Key Configuration Parameters (Backtest)
Parameter	Description	Default
capital	Starting capital	10_000
risk_per_trade	Fraction of capital risked per trade (based on stop‑loss distance)	0.01
max_position_pct	Maximum fraction of capital allocated to one position	0.30
atr_mult_sl	Stop‑loss distance in ATR units	2.0
atr_mult_tp	Take‑profit distance in ATR units	3.5
atr_mult_trail	Trailing stop distance in ATR units (0 to disable)	1.5
partial_tp_ratio	Fraction of position to close at partial take‑profit	0.5
partial_tp_mult	Partial take‑profit distance in ATR units	2.0
min_hold_days	Minimum holding days before any exit (except stop‑loss)	1
max_hold_days	Maximum holding days (force exit)	30
use_trend_filter	Do not short when price > 200‑day MA	True
signal_flip_exit	Exit if model signal flips from Long to Short (or vice‑versa)	True
commission	Round‑trip commission fraction (0.001 = 0.1%)	0.001
Output & Interpretation
After running setup.py, you will find:

<ticker>_data.csv – Raw OHLCV data.

<ticker>_predictions.csv – Contains close, true regime (trend), predicted regime (trend_pred), predicted direction (direction_pred), and final regime (final_regime where 1=Long, 2=Short, 0=Neutral).

<ticker>/plots/<ticker>_regimes.png – Price chart with colored backgrounds for true (top) and predicted (bottom) regimes.

<ticker>/plots/<ticker>_equity.png – Equity curve, drawdown, and price with trade markers.

<ticker>/<ticker>_backtest/results.csv – Detailed trade log (entry/exit dates, prices, PnL, reason, etc.).

<ticker>/<ticker>_backtest/equity_curve.csv – Daily equity values.

Sample trade log columns
entry_date, exit_date

direction (Long/Short)

entry_price, exit_price

size

pnl (net after commission)

hold_days

reason (SL, TP, PartialTP, SignalFlip, MaxHold, End)

Customization & Extending
Add new features – Edit create_features_optimized() in model_training.py.

Change model architecture – Modify TREND_PARAMS / DIR_PARAMS or replace LightGBM with another classifier.

Adjust regime detection – Tune n_regimes, window_years, step_days in CONFIG.

Modify backtest logic – Extend TradeManager class in backtest.py.

Dependencies
Python ≥ 3.8

yfinance

pandas

numpy

lightgbm

scikit-learn

matplotlib

scipy

joblib

License
This project is licensed under the MIT License – see the LICENSE file for details.

Disclaimer
This software is for educational and research purposes only.
It is not financial advice. Past performance does not guarantee future results. Use at your own risk.

Happy backtesting! 🚀
