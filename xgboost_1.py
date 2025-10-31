import math
import numpy as np
import pandas as pd
import yfinance
import yfin
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from typing import Union


def prepare_dataset(ticker: str, max_workers: int = 8) -> pd.DataFrame:
    calls, puts = yfin.get_option_chains_all(ticker, max_workers=max_workers)
    if calls.empty and puts.empty:
        raise RuntimeError("No option data fetched. Check ticker or network.")

    df = pd.concat([calls, puts], ignore_index=True)

    for col in ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    stock = yfinance.Ticker(ticker)
    hist = stock.history(period="1d")
    if hist.empty:
        raise RuntimeError("Could not fetch underlying price.")
    S = float(hist['Close'].iloc[-1])
    df['underlying_price'] = S

    def compute_mid(r):
        bid = r.get('bid', np.nan)
        ask = r.get('ask', np.nan)
        last = r.get('lastPrice', np.nan)
        if not np.isnan(bid) and not np.isnan(ask) and ask >= bid and (ask - bid) > 1e-8:
            return 0.5 * (bid + ask)
        if not np.isnan(last) and last > 0:
            return last
        m = r.get('mark', np.nan)
        if pd.notna(m) and m > 0:
            return m
        return np.nan

    df['mid_price'] = df.apply(compute_mid, axis=1)
    df['moneyness'] = df['underlying_price'] / df['strike']
    df['log_moneyness'] = np.log(
        df['underlying_price'] / df['strike'].replace(0, np.nan))
    df['intrinsic'] = df.apply(
        lambda r: max(r['underlying_price'] - r['strike'], 0.0)
        if r['option_type'] == 'call'
        else max(r['strike'] - r['underlying_price'], 0.0),
        axis=1,
    )
    df['time_value'] = df['mid_price'] - df['intrinsic']

    if 'impliedVolatility' not in df.columns:
        raise RuntimeError(
            "yfinance did not return impliedVolatility column for this ticker/expiry set.")
    df['implied_vol'] = pd.to_numeric(df['impliedVolatility'], errors='coerce')

    cols_keep = [
        'underlying_price', 'strike', 'TTM', 'option_type',
        'mid_price', 'bid', 'ask', 'volume', 'openInterest',
        'moneyness', 'log_moneyness', 'intrinsic', 'time_value',
        'implied_vol'
    ]
    present = [c for c in cols_keep if c in df.columns]
    df = df[present].copy()

    df = df.dropna(subset=['implied_vol', 'mid_price', 'TTM', 'strike'])
    df = df[(df['implied_vol'] > 1e-6) & (df['implied_vol'] < 5.0)]
    df = df[df['TTM'] > 0]
    df.reset_index(drop=True, inplace=True)
    return df


def train_xgb(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.copy()
    X['is_call'] = (X['option_type'] == 'call').astype(int)

    features = [
        'underlying_price', 'strike', 'TTM', 'mid_price', 'moneyness',
        'log_moneyness', 'intrinsic', 'time_value', 'is_call', 'volume', 'openInterest'
    ]
    features = [f for f in features if f in X.columns]
    Xf = X[features].fillna(0.0)
    y = df['implied_vol'].values

    X_train, X_test, y_train, y_test = train_test_split(
        Xf, y, test_size=test_size, random_state=random_state
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        objective='reg:squarederror'
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))

    importances = sorted(
        zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True)
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'feature_importances': importances
    }
    return model, metrics


def build_features_for_model(S: float, K: float, T: float, option_type: str,
                             mid_price: Union[float, None], volume: float = 0.0,
                             openInterest: float = 0.0) -> pd.DataFrame:
    """
    Return single-row DataFrame containing the features used in the model.
    If mid_price is None the caller should provide one or a synthetic mid will be used.
    """
    moneyness = S / K
    log_moneyness = np.log(moneyness if moneyness > 0 else np.nan)
    intrinsic = max(
        S - K, 0.0) if option_type.upper() == "C" else max(K - S, 0.0)
    time_value = (mid_price - intrinsic) if mid_price is not None else np.nan

    d = {
        'underlying_price': [S],
        'strike': [K],
        'TTM': [T],
        'option_type': ['call' if option_type.upper() == 'C' else 'put'],
        'mid_price': [mid_price],
        'volume': [volume],
        'openInterest': [openInterest],
        'moneyness': [moneyness],
        'log_moneyness': [log_moneyness],
        'intrinsic': [intrinsic],
        'time_value': [time_value],
    }
    return pd.DataFrame(d)


def rf(tickers: list[str], save: bool = False):
    print("Fetching option data and preparing dataset...")
    df = pd.concat(
        [prepare_dataset(t, max_workers=8) for t in tickers],
        ignore_index=True
    )
    print(f"Rows after cleaning: {len(df)}")
    if len(df) < 10:
        print("Warning: too few rows to train a robust model.")
    model, metrics = train_xgb(df)
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print("Top features:")
    for feat, imp in metrics['feature_importances'][:5]:
        print(f"  {feat:15s} {imp:.4f}")

    if save:
        fname = f"xgb_models/{'_'.join(tickers)}_xgb_model.json"
        model.save_model(fname)
        print(f"Model saved to {fname}")
    return model


def test_model(ticker: str, model_path: str):
    model = XGBRegressor()
    model.load_model(model_path)

    df = prepare_dataset(ticker, max_workers=4)
    if df.empty:
        print("No data for testing.")
        return None

    X = df.copy()
    X['is_call'] = (X['option_type'] == 'call').astype(int)
    features = [
        'underlying_price', 'strike', 'TTM', 'mid_price', 'moneyness',
        'log_moneyness', 'intrinsic', 'time_value', 'is_call', 'volume', 'openInterest'
    ]
    features = [f for f in features if f in X.columns]
    Xf = X[features].fillna(0.0)
    y_true = df['implied_vol'].values

    y_pred = model.predict(Xf)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"Test MAE: {mae:.6f}")
    print(f"Test RMSE: {rmse:.6f}")
    return mae, rmse
