import xgboost_1 as xgb1

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"]
    model = xgb1.rf(tickers, save=True)
    xgb1.test_model("AAPL", f"{'_'.join(tickers)}_xgb_model.json")