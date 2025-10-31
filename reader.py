import pandas as pd
import yfin as yf


def read_chain():
    df = pd.read_csv("spy_eod_202312.txt")
    return df


def read_yfin():
    df = yf.get_option_chains_all("AAPL")
    return df


calls, puts = read_yfin()
df = calls

print(df.columns)
