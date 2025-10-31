import random_forest_1 as rf1
import pandas as pd
import xgboost as xgb
import numpy as np
from typing import Union, Tuple

def build_features_for_model(S: float, K: float, T: float, option_type: str,
                              mid_price: Union[float, None], volume: float = 0.0,
                              openInterest: float = 0.0) -> pd.DataFrame:
    """
    Return single-row DataFrame containing the features used in the model.
    If mid_price is None the caller should provide one or a synthetic mid will be used.
    """
    moneyness = S / K
    log_moneyness = np.log(moneyness if moneyness > 0 else np.nan)
    intrinsic = max(S - K, 0.0) if option_type.upper() == "C" else max(K - S, 0.0)
    time_value = (mid_price - intrinsic) if mid_price is not None else np.nan

    d = {
        'underlying_price': [S],
        'strike': [K],
        'TTM': [T],
        'option_type': [ 'call' if option_type.upper() == 'C' else 'put' ],
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
        [rf1.prepare_dataset(t, max_workers=8) for t in tickers],
        ignore_index=True
    )
    print(f"Rows after cleaning: {len(df)}")
    if len(df) < 10:
        print("Warning: too few rows to train a robust model.")
    model, metrics = rf1.train_xgb(df)
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print("Top features:")
    for feat, imp in metrics['feature_importances'][:5]:
        print(f"  {feat:15s} {imp:.4f}")

    if save:
        fname = f"{'_'.join(tickers)}_xgb_model.json"
        model.save_model(fname)
        print(f"Model saved to {fname}")
    return model


def test_model(ticker: str, model_path: str):
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    df = rf1.prepare_dataset(ticker, max_workers=4)
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


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"]
    model = rf(tickers, save=True)
    test_model("AAPL", f"{'_'.join(tickers)}_xgb_model.json")