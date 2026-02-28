📈 Quantitative Market Regime Detection & Trading System

A production-grade two-stage market regime classification and backtesting framework built with:

LightGBM

Gaussian Mixture Models (GMM)

Advanced feature engineering

Risk-managed trade execution

Multi-asset backtesting engine

This project implements a regime-aware trading architecture:

Detect structural market regimes (Bull / Neutral / Bear)

Predict future regimes using supervised ML

Generate directional signals

Execute trades with strict risk management

Evaluate performance via realistic backtesting

🧠 System Architecture
Data Acquisition
        ↓
Feature Engineering
        ↓
Unsupervised Regime Detection (GMM)
        ↓
Supervised Trend Model (LightGBM)
        ↓
Directional Model (LightGBM)
        ↓
Backtest Engine (ATR-based risk management)
        ↓
Performance Analytics
📂 Project Structure
├── config.py
├── setup.py
├── data_acquisition.py
├── model_training.py
├── backtest.py
│
├── <TICKER>/
│   ├── <TICKER>_data.csv
│   ├── <TICKER>_predictions.csv
│   ├── models/
│   │   └── <TICKER>_trend_analyzer.pkl
│   ├── plots/
│   │   └── <TICKER>_regimes.png
│   └── <TICKER>_backtest/
⚙️ Installation
pip install pandas numpy yfinance lightgbm scikit-learn scipy matplotlib joblib
🚀 Full Pipeline Execution

Run the complete pipeline:

python setup.py

This performs:

✅ Step 1 — Data Download

Downloads historical OHLCV data via yfinance

Cleans column structure

Saves to CSV

✅ Step 2 — Model Training

Advanced feature engineering

Rolling GMM regime detection

Label smoothing

Feature selection via LightGBM importance

Two-stage training:

Trend Model (Bull / Neutral / Bear)

Direction Model (Long / Short)

✅ Step 3 — Visualization

Saves regime overlay plot

📊 Regime Detection (Unsupervised)

Uses:

Rolling windows

Statistical return features

Volatility regime metrics

Autocorrelation

Drawdown metrics

Gaussian Mixture Model clustering

Regimes are automatically mapped based on Sharpe:

Cluster	Mapped Regime
Lowest Sharpe	Bear
Middle	Neutral
Highest Sharpe	Bull

Labels are smoothed using median filtering to reduce noise.

🤖 Supervised Modeling
Stage 1 — Trend Model

Predicts:

Bull

Neutral

Bear

Uses:

Top-N feature selection

Class balancing

Early stopping

Time-series split

Stage 2 — Direction Model

Activated only during trending regimes.

Predicts:

Long

Short

🧪 Backtesting Engine

The backtester is not naive.

It includes:

ATR-based stop loss

ATR-based take profit

Trailing stop

Partial take profit

Maximum holding time

Signal flip exits

Trend filters (e.g., MA200)

Volatility-adjusted position sizing

Commission modeling

Realistic equity tracking

Cash-based accounting

📈 Risk Management

Position sizing is based on:

Risk per trade
ATR stop distance
Volatility scaling
Maximum portfolio exposure

This prevents over-leveraging in volatile environments.

📊 Performance Metrics

The system outputs:

Total Return

Max Drawdown

Sharpe Ratio

Calmar Ratio

Win Rate

Profit Factor

Avg Win / Avg Loss

Long vs Short performance

Exit reason breakdown

Signal filtering statistics

🌍 Multi-Asset Backtesting

Run:

run_multi_asset(["AAPL", "MSFT", "SPY", "QQQ"])

Outputs comparative table:

Return

Drawdown

Sharpe

Calmar

Win rate

Profit factor

🔬 Feature Engineering Highlights

Includes:

Multi-horizon returns

Moving average spreads

Volatility regimes

ATR and ATR %

Bollinger position & width

RSI & divergence

ADX & DI spread

Volume regime metrics

Trend strength via rolling regression

Drawdown duration

Z-score normalization

Regime lag features

📌 Configuration

Modify in config.py:

ticker = "AAPL"
period = "10y"

Modify training config in model_training.py:

CONFIG = {
    'n_regimes': 3,
    'window_years': 1,
    'step_days': 10,
    'n_top_features': 40,
    'test_size': 0.35,
}
🧩 Design Philosophy

This project avoids:

❌ Naive return prediction
❌ Random entry systems
❌ Single-stage classification
❌ Unrealistic backtests

Instead it focuses on:

✔ Structural regime modeling
✔ Time-series safe splits
✔ Risk-first execution
✔ Production-like architecture

⚠️ Disclaimer

This project is for research and educational purposes only.
It does not constitute financial advice.

🏁 Future Improvements

Walk-forward optimization

Cross-asset regime transfer learning

Bayesian regime modeling

Online model updating

Portfolio-level capital allocation

Feature importance drift tracking

Transaction cost sensitivity analysis

🧠 Author Notes

This is a quantitative research framework designed to explore:

Market structure modeling

Regime-aware strategies

Risk-controlled ML trading systems

Production-grade backtesting
