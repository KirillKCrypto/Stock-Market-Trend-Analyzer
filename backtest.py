import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from config import ticker


# ==================== FEATURE ENGINEERING ====================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    s = df.shift(1)

    # Returns & momentum
    s["ret"] = s["close"].pct_change()
    for period in [3, 5, 10, 21]:
        s[f"mom_{period}"] = s["close"].pct_change(period)

    # Moving averages
    for window in [20, 50, 200]:
        s[f"ma_{window}"] = s["close"].rolling(window).mean()
        s[f"price_ma_{window}"] = s["close"] / (s[f"ma_{window}"] + 1e-8)
        s[f"ma_{window}_slope"] = s[f"ma_{window}"].diff(5)

    # Volatility
    ret = s["close"].pct_change()
    for window in [5, 20, 50]:
        s[f"vol_{window}"] = ret.rolling(window).std() * np.sqrt(252)
    s["vol_ratio_20_50"] = s["vol_20"] / (s["vol_50"] + 1e-8)

    # ATR (True Range)
    prev_close = s["close"].shift(1)
    tr = pd.concat([
        s["high"] - s["low"],
        (s["high"] - prev_close).abs(),
        (s["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    s["atr_14"] = tr.rolling(14).mean()
    s["atr_pct"] = s["atr_14"] / (s["close"] + 1e-8)

    # Bollinger Bands
    bb_mid = s["close"].rolling(20).mean()
    bb_std = s["close"].rolling(20).std()
    s["bb_width"] = (2 * bb_std) / (bb_mid + 1e-8)
    s["bb_position"] = (s["close"] - bb_mid) / (bb_std + 1e-8)

    # RSI
    delta = s["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    s["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    # Volume
    s["volume_ma20"] = s["volume"].rolling(20).mean()
    s["volume_ratio"] = s["volume"] / (s["volume_ma20"] + 1e-8)

    # Lag features for model consistency
    s["trend_lag1"] = 0
    s["trend_lag2"] = 0
    s["trend_lag5"] = 0

    return s.dropna()


def _sharpe(returns: pd.Series, periods: int = 252) -> float:
    if returns.std() < 1e-10:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods))


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / (peak + 1e-8)
    return float(dd.max()) * 100


def _calmar(total_ret_pct: float, max_dd_pct: float) -> float:
    return total_ret_pct / (max_dd_pct + 1e-8)


# ==================== TRADE EXECUTION LOGIC ====================
class TradeManager:
    """Handles all trade execution logic with realistic constraints and correct cash/equity."""

    def __init__(self, capital: float, commission: float):
        self.capital = capital
        self.commission = commission
        self.cash = capital
        self.trades = []
        self.equity_history = []
        self.current_position = None

    def calculate_position_size(self, entry_price: float, sl_distance: float,
                                risk_per_trade: float, max_position_pct: float,
                                volatility_factor: float = 1.0) -> float:
        """Calculate position size with risk management and volatility adjustment."""
        adjusted_risk = risk_per_trade / volatility_factor
        risk_amount = self.cash * adjusted_risk
        raw_size = risk_amount / sl_distance

        max_size = (self.cash * max_position_pct) / entry_price
        position_size = min(raw_size, max_size)

        min_size = 0.1
        return max(position_size, min_size)

    def open_position(self, date: pd.Timestamp, price: float, atr: float,
                      direction: int, risk_per_trade: float, max_position_pct: float,
                      volatility: float = 1.0) -> Dict:
        """Open a new position with proper cash flow for LONG/SHORT."""
        sl_distance = atr * 2.0
        size = self.calculate_position_size(price, sl_distance, risk_per_trade,
                                            max_position_pct, volatility)

        if direction == 1:
            cost = size * price * (1 + self.commission)
            self.cash -= cost
            entry_value = size * price
        else:
            proceeds = size * price * (1 - self.commission)
            self.cash += proceeds
            entry_value = size * price  # Reference value for PnL

        position = {
            "entry_date": date,
            "entry_price": price,
            "size": size,
            "direction": direction,
            "entry_atr": atr,
            "hold_days": 0,
            "trail_best": price,
            "partial_done": False,
            "entry_value": entry_value  # For equity calc
        }
        self.current_position = position
        return position

    def close_position(self, position: Dict, exit_price: float, exit_date: pd.Timestamp, reason: str) -> Dict:
        """Close a position and calculate PnL with correct cash flow."""
        size = position["size"]
        direction = position["direction"]
        entry_price = position["entry_price"]

        if direction == 1:
            proceeds = size * exit_price * (1 - self.commission)
            self.cash += proceeds
            pnl = size * (exit_price - entry_price)
        else:
            cost = size * exit_price * (1 + self.commission)
            self.cash -= cost
            pnl = size * (entry_price - exit_price)

        net_pnl = pnl - (size * exit_price * self.commission)

        trade_record = {
            "entry_date": position["entry_date"],
            "exit_date": exit_date,
            "direction": "Long" if direction == 1 else "Short",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "pnl": net_pnl,
            "hold_days": position["hold_days"],
            "reason": reason,
            "cash_after": self.cash
        }

        self.trades.append(trade_record)
        self.current_position = None
        return trade_record

    def partial_close(self, position: Dict, exit_price: float, exit_date: pd.Timestamp,
                      close_ratio: float = 0.5) -> Dict:
        size = position["size"]
        close_size = size * close_ratio
        remain_size = size - close_size
        direction = position["direction"]
        entry_price = position["entry_price"]

        if direction == 1:
            proceeds = close_size * exit_price * (1 - self.commission)
            self.cash += proceeds
            pnl = close_size * (exit_price - entry_price)
        else:
            cost = close_size * exit_price * (1 + self.commission)
            self.cash -= cost
            pnl = close_size * (entry_price - exit_price)

        net_pnl = pnl - (close_size * exit_price * self.commission)

        # Update position
        position["size"] = remain_size
        position["partial_done"] = True

        trade_record = {
            "entry_date": position["entry_date"],
            "exit_date": exit_date,
            "direction": "Long" if direction == 1 else "Short",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": close_size,
            "pnl": net_pnl,
            "hold_days": position["hold_days"],
            "reason": "PartialTP",
            "cash_after": self.cash
        }

        self.trades.append(trade_record)
        return trade_record

    def get_current_equity(self, current_price: float) -> float:
        if self.current_position is None:
            return self.cash

        position = self.current_position
        size = position["size"]
        direction = position["direction"]

        if direction == 1:
            position_value = size * current_price
            equity = self.cash + position_value
        else:
            entry_value = position["entry_value"]
            current_value = size * current_price
            equity = self.cash + (entry_value - current_value)

        return equity


# Backtest engine
def backtest_advanced(
        ticker: str,
        model_filename: Optional[str] = None,
        capital: float = 10_000.0,
        risk_per_trade: float = 0.01,
        max_position_pct: float = 0.30,
        atr_mult_sl: float = 2.0,
        atr_mult_tp: float = 3.5,
        atr_mult_trail: float = 1.5,
        partial_tp_ratio: float = 0.5,
        partial_tp_mult: float = 2.0,
        min_hold_days: int = 1,
        max_hold_days: int = 30,
        use_trend_filter: bool = True,
        signal_flip_exit: bool = True,
        commission: float = 0.001,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        plot: bool = True,
        allow_neutral_trading: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Load model
    if model_filename is None:
        model_filename = f"{ticker}/models/{ticker}_trend_analyzer.pkl"

    if not Path(model_filename).exists():
        raise FileNotFoundError(f"Model not found: {model_filename}")

    model_data = joblib.load(model_filename)
    trend_model = model_data["trend_model"]
    direction_model = model_data["direction_model"]
    scaler = model_data["scaler"]
    features = list(model_data["features"])

    # Load and prepare data
    df_raw = pd.read_csv(f"{ticker}/{ticker}_data.csv", index_col=0, parse_dates=True)
    if start_date: df_raw = df_raw.loc[start_date:]
    if end_date: df_raw = df_raw.loc[:end_date]

    df = create_features(df_raw)

    for f in features:
        if f not in df.columns:
            df[f] = 0.0

    core_features = [f for f in features if "lag" not in f]
    df = df.dropna(subset=core_features).copy()

    dates = df.index
    n = len(df)
    feat_mat = df[features].to_numpy(dtype=np.float64)
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    atrs = df["atr_14"].to_numpy()
    ma200 = df["ma_200"].to_numpy() if "ma_200" in df.columns else np.full(n, -np.inf)
    volatility = df["vol_20"].to_numpy() if "vol_20" in df.columns else np.ones(n)

    # Initialize trade manager
    trade_mgr = TradeManager(capital, commission)
    positions_arr = np.zeros(n, dtype=int)
    trend_history = np.zeros(n, dtype=int)
    sig_history = np.zeros(n, dtype=int)

    signal_stats = {
        'total_signals': 0,
        'neutral_filtered': 0,
        'trend_filtered': 0,
        'executed_trades': 0
    }

    lag_idx = {
        name: features.index(name)
        for name in ["trend_lag1", "trend_lag2", "trend_lag5"]
        if name in features
    }

    current_position = None
    pending_signal = 0

    for i in range(n):
        for lag_name, col_idx in lag_idx.items():
            lag_val = int(lag_name.replace("trend_lag", ""))
            if i >= lag_val:
                feat_mat[i, col_idx] = trend_history[i - lag_val]

        X = scaler.transform(feat_mat[i].reshape(1, -1))
        trend_pred = int(trend_model.predict(X)[0])
        trend_history[i] = trend_pred

        cur_sig = 0
        if trend_pred != 1 or allow_neutral_trading:
            if direction_model is not None:
                cur_sig = int(direction_model.predict(X)[0])
            else:
                if trend_pred == 0:
                    cur_sig = 1
                elif trend_pred == 2:
                    cur_sig = 2
                else:
                    cur_sig = 0
        sig_history[i] = cur_sig

        if current_position is None and pending_signal != 0:
            want = pending_signal
            signal_stats['total_signals'] += 1

            if use_trend_filter and want == 2:
                if closes[i - 1] > ma200[i - 1]:
                    want = 0
                    signal_stats['trend_filtered'] += 1

            if want != 0:
                vol_factor = volatility[i] / np.mean(volatility[max(0, i - 252):i + 1]) if i > 0 else 1.0
                vol_factor = max(0.5, min(2.0, vol_factor))

                current_position = trade_mgr.open_position(
                    date=dates[i],
                    price=opens[i],
                    atr=atrs[i - 1] if i > 0 else atrs[i],
                    direction=want,
                    risk_per_trade=risk_per_trade,
                    max_position_pct=max_position_pct,
                    volatility=vol_factor
                )
                signal_stats['executed_trades'] += 1
                pending_signal = 0
            else:
                signal_stats['neutral_filtered'] += 1

        if current_position is not None:
            current_position["hold_days"] += 1

            sl_dist = current_position["entry_atr"] * atr_mult_sl
            tp_dist = current_position["entry_atr"] * atr_mult_tp

            if current_position["direction"] == 1:
                sl_level = current_position["entry_price"] - sl_dist
                tp_level = current_position["entry_price"] + tp_dist

                if atr_mult_trail > 0:
                    current_position["trail_best"] = max(
                        current_position["trail_best"],
                        closes[i - 1] if i > 0 else closes[i]
                    )
                    trail_sl = current_position["trail_best"] - current_position["entry_atr"] * atr_mult_trail
                    sl_level = max(sl_level, trail_sl)
            else:
                sl_level = current_position["entry_price"] + sl_dist
                tp_level = current_position["entry_price"] - tp_dist

                if atr_mult_trail > 0:
                    current_position["trail_best"] = min(
                        current_position["trail_best"],
                        closes[i - 1] if i > 0 else closes[i]
                    )
                    trail_sl = current_position["trail_best"] + current_position["entry_atr"] * atr_mult_trail
                    sl_level = min(sl_level, trail_sl)

            exit_price = None
            reason = ""

            if (partial_tp_ratio > 0 and not current_position["partial_done"] and
                    current_position["hold_days"] >= min_hold_days):

                partial_tp = (
                    current_position["entry_price"] + current_position["entry_atr"] * partial_tp_mult
                    if current_position["direction"] == 1
                    else current_position["entry_price"] - current_position["entry_atr"] * partial_tp_mult
                )

                hit = (current_position["direction"] == 1 and highs[i] >= partial_tp) or \
                      (current_position["direction"] == -1 and lows[i] <= partial_tp)

                if hit:
                    trade_mgr.partial_close(
                        current_position,
                        partial_tp,
                        dates[i],
                        partial_tp_ratio
                    )

            if current_position["hold_days"] >= min_hold_days:
                if (current_position["direction"] == 1 and lows[i] <= sl_level) or \
                        (current_position["direction"] == -1 and highs[i] >= sl_level):
                    exit_price = sl_level
                    reason = "SL"

                elif exit_price is None and (
                        (current_position["direction"] == 1 and highs[i] >= tp_level) or
                        (current_position["direction"] == -1 and lows[i] <= tp_level)
                ):
                    exit_price = tp_level
                    reason = "TP"

                elif exit_price is None and signal_flip_exit:
                    if (current_position["direction"] == 1 and cur_sig == 2) or \
                            (current_position["direction"] == -1 and cur_sig == 1):
                        exit_price = closes[i]
                        reason = "SignalFlip"

                elif exit_price is None and current_position["hold_days"] >= max_hold_days:
                    exit_price = closes[i]
                    reason = "MaxHold"

            if exit_price is not None:
                trade_mgr.close_position(current_position, exit_price, dates[i], reason)
                current_position = None

        if current_position is None and cur_sig != 0:
            pending_signal = cur_sig

        equity = trade_mgr.get_current_equity(closes[i])
        trade_mgr.equity_history.append(equity)
        positions_arr[i] = current_position["direction"] if current_position else 0

    if current_position is not None:
        trade_mgr.close_position(current_position, closes[-1], dates[-1], "End")
        trade_mgr.equity_history[-1] = trade_mgr.cash

    result_df = pd.DataFrame({
        "equity": trade_mgr.equity_history,
        "pos": positions_arr
    }, index=dates)

    trades_df = pd.DataFrame(trade_mgr.trades)

    _print_stats(ticker, trades_df, np.array(trade_mgr.equity_history), capital, dates, sig_history, signal_stats)

    if plot:
        _plot_results(ticker, dates, trade_mgr.equity_history, closes, positions_arr, capital, trades_df)

    return result_df, trades_df


# Statistics
def _print_stats(ticker: str, trades: pd.DataFrame, equity: np.ndarray,
                 capital: float, dates: pd.DatetimeIndex, sig_history: Optional[np.ndarray] = None,
                 signal_stats: Optional[Dict] = None) -> None:
    total_ret = (equity[-1] / capital - 1) * 100
    max_dd = _max_drawdown(equity)

    eq_series = pd.Series(equity, index=dates)
    daily_returns = eq_series.pct_change().dropna()
    sharpe = _sharpe(daily_returns)
    calmar = _calmar(total_ret, max_dd)

    n_trades = len(trades)
    win_rate = (trades["pnl"] > 0).mean() * 100 if n_trades > 0 else 0

    print(f"Total Return:      {total_ret:+.2f}%")
    print(f"Max Drawdown:      {max_dd:.2f}%")
    print(f"Sharpe Ratio:      {sharpe:.2f}")
    print(f"Calmar Ratio:      {calmar:.2f}")
    print(f"{'-' * 70}")
    print(f"Number of Trades:  {n_trades}")
    print(f"Win Rate:          {win_rate:.1f}%")

    if sig_history is not None:
        sig_counts = np.bincount(sig_history, minlength=3)
        total_signals = len(sig_history)
        print(f"{'-' * 70}")
        print("Model Signal Distribution:")
        print(f"  Neutral (No Trade): {sig_counts[0]} ({sig_counts[0] / total_signals * 100:.1f}%)")
        print(f"  Long Signals:       {sig_counts[1]} ({sig_counts[1] / total_signals * 100:.1f}%)")
        print(f"  Short Signals:      {sig_counts[2]} ({sig_counts[2] / total_signals * 100:.1f}%)")

        if signal_stats:
            print(f"{'-' * 70}")
            print("Signal Processing Stats:")
            print(f"  Total Generated Signals: {signal_stats['total_signals']}")
            print(f"  Trend Filtered:          {signal_stats['trend_filtered']}")
            print(f"  Neutral Filtered:        {signal_stats['neutral_filtered']}")
            print(f"  Executed Trades:         {signal_stats['executed_trades']}")

    if n_trades > 0:
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]
        avg_win = wins["pnl"].mean()
        avg_loss = losses["pnl"].mean()
        profit_factor = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 else np.inf

        longs = trades[trades["direction"] == "Long"]
        shorts = trades[trades["direction"] == "Short"]

        reasons = trades["reason"].value_counts()

        print(f"Avg Win:           ${avg_win:+.2f}")
        print(f"Avg Loss:          ${avg_loss:+.2f}")
        print(f"Profit Factor:     {profit_factor:.2f}")
        print(f"{'-' * 70}")
        print(f"Long Trades:       {len(longs)} (WR: {(longs['pnl'] > 0).mean() * 100:.1f}%)")
        print(f"Short Trades:      {len(shorts)} (WR: {(shorts['pnl'] > 0).mean() * 100:.1f}%)")
        print(f"{'-' * 70}")
        print("Exit Reasons:")
        for reason, count in reasons.items():
            print(f"  {reason:<12}: {count}")
    print(f"{'=' * 70}")


def _plot_results(ticker: str, dates: pd.DatetimeIndex, equity: List[float],
                  closes: np.ndarray, positions: np.ndarray, capital: float,
                  trades: pd.DataFrame) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
    fig.suptitle(f"{ticker} - Backtest Results", fontsize=16, fontweight='bold')

    ax1.plot(dates, equity, 'steelblue', linewidth=1.5, label='Equity')
    ax1.axhline(capital, color='gray', linestyle='--', linewidth=0.8)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    if not trades.empty:
        for _, trade in trades.iterrows():
            color = 'green' if trade['pnl'] > 0 else 'red'
            ax1.axvline(trade['exit_date'], color=color, alpha=0.15, linewidth=0.8)

    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / (peak + 1e-8) * 100
    ax2.fill_between(dates, 0, -dd, color='tomato', alpha=0.6, label='Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    ax3.plot(dates, closes, 'black', linewidth=0.8, label='Price')
    long_mask = positions == 1
    short_mask = positions == -1
    ax3.fill_between(dates, closes, where=long_mask, color='green', alpha=0.25, label='Long')
    ax3.fill_between(dates, closes, where=short_mask, color='red', alpha=0.25, label='Short')
    ax3.set_ylabel('Price ($)')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==================== MULTI-ASSET BACKTEST ====================
def run_multi_asset(tickers: List[str], **kwargs) -> pd.DataFrame:
    results = []
    for ticker in tickers:
        try:
            result_df, trades_df = backtest_advanced(ticker, plot=True, **kwargs)

            equity = result_df["equity"].values
            capital = kwargs.get("capital", 10_000.0)
            dates = result_df.index

            total_ret = (equity[-1] / capital - 1) * 100
            max_dd = _max_drawdown(equity)

            daily_returns = pd.Series(equity, index=dates).pct_change().dropna()
            sharpe = _sharpe(daily_returns)
            calmar = _calmar(total_ret, max_dd)

            n_trades = len(trades_df)
            win_rate = (trades_df["pnl"] > 0).mean() * 100 if n_trades > 0 else 0

            if n_trades > 0:
                wins = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
                losses = abs(trades_df[trades_df["pnl"] <= 0]["pnl"].sum())
                profit_factor = wins / losses if losses > 0 else np.inf
            else:
                profit_factor = 0

            results.append({
                "Ticker": ticker,
                "Return": f"{total_ret:.2f}%",
                "MaxDD": f"{max_dd:.2f}%",
                "Sharpe": f"{sharpe:.2f}",
                "Calmar": f"{calmar:.2f}",
                "Trades": n_trades,
                "WinRate": f"{win_rate:.1f}%",
                "ProfitFactor": f"{profit_factor:.2f}"
            })

        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            results.append({
                "Ticker": ticker,
                "Return": "Error",
                "MaxDD": "Error",
                "Sharpe": "Error",
                "Calmar": "Error",
                "Trades": 0,
                "WinRate": "Error",
                "ProfitFactor": "Error"
            })

    results_df = pd.DataFrame(results).set_index("Ticker")
    print("\n" + "=" * 80)
    print("MULTI-ASSET BACKTEST RESULTS")
    print("=" * 80)
    print(results_df.to_string())
    print("=" * 80)
    return results_df

if __name__ == "__main__":

    run_multi_asset(
        ["AAPL", "MSFT", "SPY", "QQQ", "META", "V", "GOOGL"], # Put your tickers for multi-asset backtesting(Run setup.py before backtest)
        capital=10_000,
        risk_per_trade=0.08,
        atr_mult_sl=2.0,
        atr_mult_tp=3.5,
        atr_mult_trail=1.5,
        partial_tp_ratio=0.5,
        partial_tp_mult=2.0,
        min_hold_days=3,
        max_hold_days=14,
        use_trend_filter=True,
        allow_neutral_trading=False
    )