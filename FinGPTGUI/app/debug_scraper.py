import pandas as pd
import lxml
import ssl

print("Testing S&P 500 Scraper...")
try:
    # Hack for some docker SSL verification issues
    ssl._create_default_https_context = ssl._create_unverified_context
    
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    print(f"Reading {url}...")
    tables = pd.read_html(url)
    print(f"Found {len(tables)} tables.")
    df = tables[0]
    print(df.head())
    tickers = df['Symbol'].tolist()
    print(f"Successfully scraped {len(tickers)} tickers.")
except Exception as e:
    print(f"CRASH: {e}")
    import traceback
    traceback.print_exc()
