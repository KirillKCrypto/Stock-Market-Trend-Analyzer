# setup.py
# Complete pipeline - Data download, model training, backtest results

import sys
import time
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

required_files = ['config.py', 'data_acquisition.py', 'model_training.py', 'backtest.py']
missing_files = [f for f in required_files if not Path(f).exists()]
if missing_files:
    print(f"❌ ERROR: Missing required files: {missing_files}")
    sys.exit(1)

try:
    from config import ticker, period
    from data_acquisition import download_data
except ImportError as e:
    print(f"❌ ERROR: Could not import required modules: {e}")
    print("Make sure config.py and data_acquisition.py exist")
    sys.exit(1)


# Create directory structure
print(f"\n{'=' * 70}")
print("STEP 0/3: Setting up Directory Structure")
print(f"{'=' * 70}\n")

try:
    # Create main ticker directory
    ticker_dir = Path(ticker)
    ticker_dir.mkdir(exist_ok=True)
    print(f"✓ Created/verified directory: {ticker_dir}/")

    # Create backtest subdirectory
    backtest_dir = ticker_dir / f"{ticker}_backtest"
    backtest_dir.mkdir(exist_ok=True)
    print(f"✓ Created/verified directory: {backtest_dir}/")

    # Create additional subdirectories (optional, for organization)
    models_dir = ticker_dir / "models"
    models_dir.mkdir(exist_ok=True)
    print(f"✓ Created/verified directory: {models_dir}/")

    plots_dir = ticker_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    print(f"✓ Created/verified directory: {plots_dir}/")

    print(f"\nDirectory structure ready!")

except Exception as e:
    print(f"❌ ERROR creating directories: {e}")
    sys.exit(1)

# Data Download
print(f"\n{'=' * 70}")
print("STEP 1/3: Downloading Market Data")
print(f"{'=' * 70}\n")

try:
    df = download_data(ticker, period)
    output_path = ticker_dir / f"{ticker}_data.csv"
    df.to_csv(output_path)

    print(f"✓ Successfully saved data to {output_path}")
    print(f"✓ Data shape: {df.shape}")
    print(f"✓ Date range: {df.index[0].date()} to {df.index[-1].date()}")

except Exception as e:
    print(f"❌ ERROR in data download: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Model training
print(f"\n{'=' * 70}")
print("STEP 2/3: Training Two-Stage Regime Model")
print(f"{'=' * 70}\n")

try:
    from model_training import train_production_model, plot_regimes_fast

    start_time = time.time()

    # Train model
    df_trained = train_production_model(ticker)

    elapsed = time.time() - start_time
    print(f"\n⏱  Training completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # Generate regime visualization (save to plots subdirectory)
    print("\nGenerating regime plots...")
    regimes_plot_path = plots_dir / f"{ticker}_regimes.png"
    plot_regimes_fast(
        df_trained,
        window=(-500, None),  # Last 500 days
        save_path=str(regimes_plot_path)
    )
    print(f"✓ Regime plot saved to: {regimes_plot_path}")

except Exception as e:
    print(f"❌ ERROR in model training: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Backtesting
print(f"\n{'=' * 70}")
print("STEP 3/3: Running Backtest")
print(f"{'=' * 70}\n")

try:
    from backtest import backtest_advanced

    # Define output path for backtest results
    backtest_results_path = backtest_dir / "results.csv"
    equity_plot_path = plots_dir / f"{ticker}_equity.png"

    results, trades = backtest_advanced(
        ticker,
        capital=10_000,
        risk_per_trade=0.01,
        max_position_pct=0.30,
        atr_mult_sl=2.0,
        atr_mult_tp=3.5,
        atr_mult_trail=1.5,
        partial_tp_ratio=0.5,
        partial_tp_mult=2.0,
        min_hold_days=1,
        max_hold_days=30,
        use_trend_filter=True,
        signal_flip_exit=True,
        commission=0.001
    )
    # Display trade summary
    if len(trades) > 0:
        print("\n" + "=" * 70)
        print("TRADE SUMMARY")
        print("=" * 70)
        print(f"\nTotal trades: {len(trades)}")
        print(f"\nTrade breakdown by exit reason:")
        print(trades['reason'].value_counts())

        print(f"\n\nLast 10 trades:")
        print(trades.tail(10).to_string(index=False))

        # Save trades to CSV in backtest directory
        trades.to_csv(backtest_results_path, index=False)
        print(f"\n✓ Trades saved to: {backtest_results_path}")

        # Save results DataFrame
        results_df_path = backtest_dir / "equity_curve.csv"
        results.to_csv(results_df_path)
        print(f"✓ Equity curve saved to: {results_df_path}")
    else:
        print("\n⚠ No trades generated in backtest period")

except Exception as e:
    print(f"❌ ERROR in backtesting: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Pipeline complete
print(f"\n{'=' * 70}")
print("✓ PIPELINE COMPLETED SUCCESSFULLY")
print(f"{'=' * 70}")

print(f"\nGenerated directory structure:")
print(f"""
{ticker}/
├── {ticker}_data.csv              - Raw OHLCV data
├── {ticker}_trend_analyzer.pkl    - Trained model
├── {ticker}_predictions.csv       - Regime predictions
├── models/                        - Model artifacts
├── plots/
│   ├── {ticker}_regimes.png      - Regime visualization
│   └── {ticker}_equity.png       - Backtest equity curve
└── {ticker}_backtest/
    ├── results.csv               - Trade log
    └── equity_curve.csv          - Full equity timeseries
""")

print(f"\n{'=' * 70}")
print("NEXT STEPS:")
print(f"{'=' * 70}")
print(f"""
1. Review regime predictions:
   - Check {ticker}/plots/{ticker}_regimes.png for visual confirmation

2. Analyze backtest results:
   - Review {ticker}/plots/{ticker}_equity.png
   - Examine {ticker}/{ticker}_backtest/results.csv for trade quality

3. Adjust parameters in config.py:
   - Try different tickers
   - Modify TP/SL levels in backtest()

4. Re-run: python setup.py
""")
