import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple
from lightgbm import LGBMClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.mixture import GaussianMixture
from scipy.ndimage import median_filter

warnings.filterwarnings("ignore")

# Import from your existing config
from config import ticker, period

# ==================== CONFIGURATION ====================
CONFIG = {
    'n_regimes': 3,
    'window_years': 1,
    'step_days': 10,
    'n_top_features': 40,
    'test_size': 0.35,
    'use_pca': False,
    'early_stopping_rounds': 100,
    'label_smoothing': 5
}

# Optimized hyperparameters
TREND_PARAMS = {
    'n_estimators': 1000,
    'max_depth': 4,
    'learning_rate': 0.03,
    'num_leaves': 20,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_samples': 30,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'class_weight': 'balanced',
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1
}

DIR_PARAMS = {
    'n_estimators': 500,
    'max_depth': 3,
    'learning_rate': 0.05,
    'num_leaves': 15,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'class_weight': 'balanced',
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1
}


# FEATURE ENGINEERING
def create_features_optimized(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    for period_val in [1, 3, 5, 10, 21, 63]:
        df[f'ret_{period_val}'] = close.pct_change(period_val)

    for window in [10, 20, 50, 200]:
        ma = close.rolling(window).mean()
        df[f'price_to_ma{window}'] = close / ma
        df[f'ma{window}_slope'] = ma.diff(5)

    ret = close.pct_change()
    for window in [5, 10, 20, 50]:
        df[f'vol_{window}'] = ret.rolling(window).std() * np.sqrt(252)

    df['vol_ratio_20_50'] = df['vol_20'] / (df['vol_50'] + 1e-8)
    df['vol_acceleration'] = df['vol_20'].diff(5)

    tr = np.maximum(
        high - low,
        np.maximum(
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        )
    )
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / close

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_width'] = (2 * bb_std) / bb_mid
    df['bb_position'] = (close - bb_mid) / (bb_std + 1e-8)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd - macd_signal
    df['macd_hist_slope'] = df['macd_hist'].diff(3)

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_divergence'] = df['rsi'].diff(5)

    up = high.diff()
    down = -low.diff()
    pos_dm = ((up > down) & (up > 0)) * up
    neg_dm = ((down > up) & (down > 0)) * down
    atr_smooth = tr.rolling(14).mean()
    pos_di = 100 * pos_dm.ewm(span=14, adjust=False).mean() / (atr_smooth + 1e-8)
    neg_di = 100 * neg_dm.ewm(span=14, adjust=False).mean() / (atr_smooth + 1e-8)
    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di + 1e-8)
    df['adx'] = dx.rolling(14).mean()
    df['di_spread'] = pos_di - neg_di

    df['volume_ma20'] = volume.rolling(20).mean()
    df['volume_ratio'] = volume / (df['volume_ma20'] + 1e-8)
    df['volume_trend'] = df['volume_ma20'].pct_change(10)

    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    df['mfi'] = money_flow.rolling(14).sum() / (volume.rolling(14).sum() + 1e-8)

    rolling_high = high.rolling(21).max()
    rolling_low = low.rolling(21).min()
    df['price_position'] = (close - rolling_low) / (rolling_high - rolling_low + 1e-8)

    df['trend_strength'] = close.rolling(20).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 20 else 0,
        raw=True
    )

    df['drawdown'] = (close / close.cummax()) - 1
    df['drawdown_duration'] = (df['drawdown'] < 0).astype(int).groupby(
        (df['drawdown'] >= 0).cumsum()
    ).cumsum()

    for window in [20, 50]:
        mean = close.rolling(window).mean()
        std = close.rolling(window).std()
        df[f'zscore_{window}'] = (close - mean) / (std + 1e-8)

    return df


# Regime Detection
class ProductionRegimeDetector:

    def __init__(self, n_regimes: int = 3, window_years: int = 2,
                 step_days: int = 21, smooth_kernel: int = 5):
        self.n_regimes = n_regimes
        self.window = int(window_years * 252)
        self.step = step_days
        self.smooth_kernel = smooth_kernel
        self.labels_ = None

    def _extract_regime_features(self, window_df: pd.DataFrame) -> Optional[np.ndarray]:
        ret = window_df['close'].pct_change().dropna()
        if len(ret) < 30:
            return None

        close = window_df['close']

        features = [
            # Returns
            ret.mean() * 252,
            ret.std() * np.sqrt(252),
            ret.skew(),
            ret.kurtosis(),

            # Quantiles
            ret.quantile(0.05),
            ret.quantile(0.95),

            # Multi-period returns
            (close.iloc[-1] / close.iloc[-21] - 1) if len(close) > 21 else 0,
            (close.iloc[-1] / close.iloc[-63] - 1) if len(close) > 63 else 0,

            # Volatility regime
            ret.rolling(21).std().mean() * np.sqrt(252),
            ret.rolling(5).std().mean() / (ret.rolling(21).std().mean() + 1e-8),

            # Drawdown
            ((close - close.cummax()) / close.cummax()).min(),

            # Autocorrelation
            ret.autocorr(lag=1) if len(ret) > 1 else 0,
            ret.autocorr(lag=5) if len(ret) > 5 else 0
        ]

        return np.array(features)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        labels = np.zeros(n, dtype=int)

        for i in range(self.window, n, self.step):
            window_df = df.iloc[max(0, i - self.window):i]

            # Extract features from sub-windows
            features = []
            for j in range(0, len(window_df) - 40, 20):
                feat = self._extract_regime_features(window_df.iloc[j:j + 60])
                if feat is not None:
                    features.append(feat)

            if len(features) < self.n_regimes * 3:
                continue

            X = np.array(features)

            # Standardize
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

            # Fit GMM
            try:
                gmm = GaussianMixture(
                    n_components=self.n_regimes,
                    covariance_type='full',
                    random_state=42,
                    n_init=3,
                    max_iter=100
                )
                gmm.fit(X)

                # Predict current regime
                curr_feat = self._extract_regime_features(window_df)
                if curr_feat is not None:
                    curr_feat_scaled = (curr_feat - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
                    curr_label = gmm.predict(curr_feat_scaled.reshape(1, -1))[0]
                    labels[i:min(i + self.step, n)] = curr_label
            except Exception:
                continue

        # Fill initial period
        first_valid = next((i for i, x in enumerate(labels) if x != 0), self.window)
        labels[:first_valid] = labels[first_valid] if first_valid < n else 0

        # Smooth labels to remove noise
        if self.smooth_kernel > 1:
            labels = median_filter(labels, size=self.smooth_kernel, mode='nearest')

        self.labels_ = labels
        return labels


def add_regime_labels_production(df: pd.DataFrame) -> pd.DataFrame:

    detector = ProductionRegimeDetector(
        n_regimes=CONFIG['n_regimes'],
        window_years=CONFIG['window_years'],
        step_days=CONFIG['step_days'],
        smooth_kernel=CONFIG['label_smoothing']
    )

    cluster_labels = detector.fit_transform(df)
    df['cluster'] = cluster_labels

    past_ret = df['close'].pct_change(21)

    regime_stats = {}
    for c in range(CONFIG['n_regimes']):
        mask = df['cluster'] == c
        if mask.sum() > 0:
            regime_stats[c] = {
                'return': past_ret[mask].mean(),
                'vol': df.loc[mask, 'ret_1'].std() * np.sqrt(252) if 'ret_1' in df else 0,
                'sharpe': (past_ret[mask].mean() * 252) / (past_ret[mask].std() * np.sqrt(252) + 1e-8),
                'count': mask.sum()
            }
        else:
            regime_stats[c] = {'return': 0, 'vol': 0, 'sharpe': 0, 'count': 0}

    sorted_regimes = sorted(
        regime_stats.keys(),
        key=lambda x: regime_stats[x]['sharpe']
    )

    # Mapping: Bear (2) → Neutral (1) → Bull (0)
    mapping = {
        sorted_regimes[0]: 2,  # Worst Sharpe = Bear
        sorted_regimes[1]: 1,  # Middle = Neutral
        sorted_regimes[2]: 0   # Best Sharpe = Bull
    }

    df['trend'] = df['cluster'].map(mapping).fillna(1).astype(int)

    # Direction labels
    df['direction'] = 0
    df.loc[df['trend'] == 0, 'direction'] = 1
    df.loc[df['trend'] == 2, 'direction'] = 2

    # Print distribution
    print("\nRegime Distribution:")
    for c in [0, 1, 2]:
        name = ['Bull', 'Neutral', 'Bear'][c]
        count = (df['trend'] == c).sum()
        pct = count / len(df) * 100
        stats = regime_stats[next(k for k, v in mapping.items() if v == c)]
        print(f"  {name:8s}: {count:5d} ({pct:5.1f}%) | "
              f"Sharpe: {stats['sharpe']:+.2f} | Vol: {stats['vol']:.1f}%")

    df.drop('cluster', axis=1, inplace=True)
    return df


# Training Pipeline
def train_production_model(ticker_symbol: str, output_dir: Optional[Path] = None) -> pd.DataFrame:
    if output_dir is None:
        output_dir = Path(ticker_symbol)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 1. Load & engineer features
    data_path = output_dir / f"{ticker_symbol}_data.csv"
    df = pd.read_csv(data_path, parse_dates=True, index_col=0)
    print(f"Raw data: {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")

    df = create_features_optimized(df)
    print(f"After features: {len(df)} rows")

    # 2. Regime detection
    df = add_regime_labels_production(df)

    # 3. Lag features
    for lag_val in [1, 2, 5]:
        df[f'trend_lag{lag_val}'] = df['trend'].shift(lag_val)
    df = df.dropna()

    # 4. Feature selection
    exclude = ['open', 'high', 'low', 'close', 'volume', 'trend', 'direction']
    all_features = [c for c in df.columns
                    if c not in exclude and df[c].dtype in ['float64', 'int64']]

    X = df[all_features].values
    y_trend = df['trend'].values
    y_dir = df['direction'].values

    print(f"\nSelecting top {CONFIG['n_top_features']} features via LGBM importance...")
    temp_model = LGBMClassifier(n_estimators=100, max_depth=3, verbose=-1, random_state=42)
    temp_model.fit(X, y_trend)

    importances = pd.Series(temp_model.feature_importances_, index=all_features)
    selected_features = importances.nlargest(CONFIG['n_top_features']).index.tolist()

    print(f"Top 10 features: {selected_features[:10]}")

    X_selected = df[selected_features].values

    # 5. Scale
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # 6. Time-series split
    split_idx = int(len(df) * (1 - CONFIG['test_size']))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_trend_train, y_trend_test = y_trend[:split_idx], y_trend[split_idx:]
    y_dir_train, y_dir_test = y_dir[:split_idx], y_dir[split_idx:]

    # Validation set for early stopping (last 20% of training)
    val_idx = int(len(X_train) * 0.8)
    X_train_fit, X_val = X_train[:val_idx], X_train[val_idx:]
    y_train_fit, y_val = y_trend_train[:val_idx], y_trend_train[val_idx:]

    print(f"\nSplit: Train={len(X_train_fit)} | Val={len(X_val)} | Test={len(X_test)}")

    # Trend Model

    trend_model = LGBMClassifier(**TREND_PARAMS)

    sample_weights = compute_sample_weight('balanced', y_train_fit)

    trend_model.fit(
        X_train_fit, y_train_fit,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(CONFIG['early_stopping_rounds'], verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )

    # Test evaluation
    y_trend_pred = trend_model.predict(X_test)

    print("\n=== Test Set Performance ===")
    print(classification_report(
        y_trend_test, y_trend_pred,
        target_names=["Bull", "Neutral", "Bear"],
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(y_trend_test, y_trend_pred)
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm,
                       index=['True Bull', 'True Neutral', 'True Bear'],
                       columns=['Pred Bull', 'Pred Neutral', 'Pred Bear']))

    # Direction Model

    mask_dir_train = y_dir_train > 0
    direction_model = None

    if mask_dir_train.sum() > 100:
        direction_model = LGBMClassifier(**DIR_PARAMS)

        dir_weights = compute_sample_weight('balanced', y_dir_train[mask_dir_train])
        direction_model.fit(
            X_train[mask_dir_train],
            y_dir_train[mask_dir_train],
            sample_weight=dir_weights
        )

        # Evaluate on trend periods only
        mask_test_trend = (y_trend_test != 1)
        if mask_test_trend.sum() > 0:
            y_dir_pred = direction_model.predict(X_test[mask_test_trend])
            y_dir_true = y_dir_test[mask_test_trend]

            print("\n=== Direction Model (Trend Periods) ===")
            print(classification_report(
                y_dir_true, y_dir_pred,
                target_names=["Long", "Short"],
                zero_division=0
            ))
    else:
        print("Not enough directional samples.")

    # Save Model

    model_path = output_dir / f"models/{ticker_symbol}_trend_analyzer.pkl"
    joblib.dump({
        'trend_model': trend_model,
        'direction_model': direction_model,
        'scaler': scaler,
        'features': selected_features,
        'config': CONFIG,
        'feature_importance': importances.to_dict()
    }, model_path)

    # Data Prediction

    df['trend_pred'] = trend_model.predict(X_scaled)
    df['direction_pred'] = 0

    if direction_model is not None:
        mask_trend = (df['trend_pred'] != 1)
        df.loc[mask_trend, 'direction_pred'] = direction_model.predict(X_scaled[mask_trend])

    df['final_regime'] = 0
    df.loc[df['trend_pred'] == 0, 'final_regime'] = 1
    df.loc[df['trend_pred'] == 2, 'final_regime'] = 2

    # Save predictions
    pred_path = output_dir / f"{ticker_symbol}_predictions.csv"
    df[['close', 'trend', 'trend_pred', 'direction_pred', 'final_regime']].to_csv(pred_path)
    print(f"✓ Predictions saved to {pred_path}")

    return df


# Visualization
def plot_regimes_fast(df: pd.DataFrame, window: Optional[Tuple[Optional[int], Optional[int]]] = None,
                      save_path: Optional[str] = None) -> None:
    """Optimized plotting (10x faster than loop-based axvspan)."""
    if window:
        start_idx = window[0] if window[0] is not None and window[0] >= 0 else 0
        end_idx = window[1] if window[1] is not None else len(df)
        df = df.iloc[start_idx:end_idx]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Plot price
    ax1.plot(df.index, df['close'], 'k-', linewidth=0.8, label='Price')
    ax2.plot(df.index, df['close'], 'k-', linewidth=0.8, label='Price')

    # Vectorized background coloring
    for ax, col in [(ax1, 'trend'), (ax2, 'final_regime')]:
        regime_series = df[col].copy()

        # Find regime change points
        changes = regime_series.ne(regime_series.shift()).cumsum()

        for _, group in df.groupby(changes):
            regime_val = group[col].iloc[0]
            start, end = group.index[0], group.index[-1]

            if regime_val == 0:  # Bull
                color, alpha = 'green', 0.15
            elif regime_val == 2:  # Bear
                color, alpha = 'red', 0.15
            else:
                continue

            ax.axvspan(start, end, color=color, alpha=alpha)

    ax1.set_title('TRUE Regimes (GMM Labels)', fontweight='bold', fontsize=12)
    ax2.set_title('PREDICTED Regimes (Model)', fontweight='bold', fontsize=12)

    for ax in [ax1, ax2]:
        ax.set_ylabel('Price ($)')
        ax.grid(alpha=0.2)
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")

    plt.show()


# ==================== MAIN ====================
if __name__ == "__main__":
    start = time.time()

    print(f"Using ticker: {ticker}")
    print(f"Using period: {period}")

    df = train_production_model(ticker)

    elapsed = time.time() - start
    print(f"\n⏱  Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # Visualize last 500 days
    plot_regimes_fast(
        df,
        window=(-1500, None),
        save_path=f"{ticker}/{ticker}_regimes.png"
    )