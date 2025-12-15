import yfinance as yf
print("Downloading NVDA...")
try:
    df = yf.download("NVDA", period="1mo", interval="1d")
    print("Shape:", df.shape)
    print("Columns:", df.columns)
    print("Head:\n", df.head())
    
    if df.empty:
        print("DATAFRAME IS EMPTY!")
    else:
        print("Data check passed.")
except Exception as e:
    print(f"CRASH: {e}")
