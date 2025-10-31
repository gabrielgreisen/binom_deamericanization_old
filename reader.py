import pandas as pd
import yfin as yf

def read_chain():
    df = pd.read_csv("spy_eod_202312.txt")
    return df

df = read_chain()
print(df)