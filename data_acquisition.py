"""
Module: Stock Data Downloader
Description: Downloads historical market data using yfinance and saves to CSV.
"""

import yfinance as yf
import pandas as pd
from config import ticker, period


def download_data(ticker: str, period: str, interval: str = "1d") -> pd.DataFrame:
    """
    Downloads historical stock data and cleans column names.
    """
    print(f"Downloading data for {ticker} (Period: {period})...")

    # Download data
    df = yf.download(ticker, period=period, interval=interval, progress=False)

    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Check the symbol.")

    # Flatten multi-index columns if they exist (yfinance v0.2.x fix)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Standardize column names to lowercase
    df.columns = [col.lower() for col in df.columns]

    # Ensure specific order and selection
    # Columns typically: open, high, low, close, volume, adj close
    df = df[['open', 'high', 'low', 'close', 'volume']]

    return df


def main():

    try:
        # Download and Save
        df = download_data(ticker, period)
        output_path = f"{ticker}/{ticker}_data.csv"
        df.to_csv(output_path)

        print(f"Successfully saved data to {output_path}")
        print(f"Data shape: {df.shape}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()