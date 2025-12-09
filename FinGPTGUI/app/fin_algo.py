import torch
import torch.nn as nn
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import os
import requests
from polygon import RESTClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# CONFIGURATION
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
if not POLYGON_API_KEY:
    print("WARNING: POLYGON_API_KEY not found in environment variables.")

MODEL_PATH = "fingpt_model.pth"

# Technical Analysis Helper
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class FinGPTTrader:
    def __init__(self, ticker="AAPL"):
        self.ticker = ticker
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing FinGPTTrader on {self.device}")
        
        # Init Polygon Client & VADER
        self.poly_client = RESTClient(POLYGON_API_KEY)
        self.vader = SentimentIntensityAnalyzer()
        
        # Load existing model if available
        self.load_model()
        
    def check_gpu(self):
        if torch.cuda.is_available():
            return f"✅ GPU Available: {torch.cuda.get_device_name(0)}"
        return "❌ GPU Not Available"
        
    def save_model(self):
        """Saves current model state to disk"""
        if hasattr(self, 'model') and self.model is not None:
             torch.save(self.model.state_dict(), MODEL_PATH)

    def load_model(self):
        """Loads model from disk if exists"""
        if os.path.exists(MODEL_PATH):
             print("Loading persisted model from disk...")
             try:
                 model = LSTMModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1).to(self.device)
                 model.load_state_dict(torch.load(MODEL_PATH))
                 self.model = model
             except Exception as e:
                 print(f"Failed to load model: {e}")

    def fetch_data(self, ticker=None, period="2y", interval="1d"):
        """Fetches historic data from Polygon.io"""
        target_ticker = ticker if ticker else self.ticker
        
        try:
            # Calculate dates based on period
            end_date = pd.Timestamp.now()
            if period == "6mo": start_date = end_date - pd.DateOffset(months=6)
            elif period == "1y": start_date = end_date - pd.DateOffset(years=1)
            elif period == "2y": start_date = end_date - pd.DateOffset(years=2)
            elif period == "5y": start_date = end_date - pd.DateOffset(years=5)
            else: start_date = end_date - pd.DateOffset(years=2)
            
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            aggs = []
            for a in self.poly_client.list_aggs(target_ticker, 1, 'day', from_=start_str, to=end_str, limit=50000):
                aggs.append(a)
                
            if not aggs:
                print(f"Polygon returned no data for {target_ticker}, falling back to Yahoo...")
                return self.fetch_data_yahoo(target_ticker, period, interval)
                
            data = []
            for a in aggs:
                data.append({
                    "Date": pd.Timestamp(a.timestamp, unit='ms'),
                    "Open": a.open,
                    "High": a.high,
                    "Low": a.low,
                    "Close": a.close,
                    "Volume": a.volume
                })
            df = pd.DataFrame(data)
            df.set_index("Date", inplace=True)
            
            df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
            df['RSI'] = calculate_rsi(df['Close'])
            df['MACD'], df['Signal_Line'] = calculate_macd(df['Close'])
            df.dropna(inplace=True)
            
            if ticker is None or ticker == self.ticker:
                self.data = df
                
            return df
            
        except Exception as e:
            print(f"Error fetching {target_ticker} from Polygon: {e}")
            return self.fetch_data_yahoo(target_ticker, period, interval)

    def fetch_data_yahoo(self, target_ticker, period, interval):
        try:
            df = yf.download(target_ticker, period=period, interval=interval, progress=False)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
            df['RSI'] = calculate_rsi(df['Close'])
            df['MACD'], df['Signal_Line'] = calculate_macd(df['Close'])
            df.dropna(inplace=True)
            if target_ticker == self.ticker:
                self.data = df
            return df
        except: return None
        
    def analyze_sentiment(self, ticker):
        """Fetches news and calculates sentiment score (-1 to 1)"""
        try:
            news = self.poly_client.list_ticker_news(ticker, limit=5)
            scores = []
            headlines = []
            
            for item in news:
                text = f"{item.title} {item.description}"
                vs = self.vader.polarity_scores(text)
                scores.append(vs['compound'])
                headlines.append(item.title)
                
            if not scores: return 0, []
            
            avg_score = sum(scores) / len(scores)
            return avg_score, headlines
        except Exception as e:
            print(f"Sentiment failed for {ticker}: {e}")
            return 0, []

    def get_model(self, hidden_dim=32, num_layers=2):
        if hasattr(self, 'model') and self.model is not None:
             return self.model
        print("Initializing new Generalized Market Model...")
        model = LSTMModel(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1).to(self.device)
        self.model = model
        return model

    def train_lstm_model(self, df, epochs=10, hidden_dim=32, num_layers=2):
        model = self.get_model(hidden_dim, num_layers)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
        
        data = df['Log_Ret'].values.reshape(-1, 1)
        
        if not hasattr(self, 'scaler'):
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            data_scaled = self.scaler.fit_transform(data)
        else:
            data_scaled = self.scaler.fit_transform(data) 

        lookback = 60
        x_train, y_train = [], []
        
        if len(data_scaled) <= lookback: return None
            
        for i in range(len(data_scaled) - lookback):
            x_train.append(data_scaled[i:i+lookback])
            y_train.append(data_scaled[i+lookback])
            
        x_train = torch.FloatTensor(np.array(x_train)).to(self.device)
        y_train = torch.FloatTensor(np.array(y_train)).to(self.device)
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            yield epoch + 1, loss.item()
            
        self.lookback = lookback
        self.save_model() 
        return model

    def train_on_tickers(self, tickers, epochs_per_ticker=5):
        total_steps = len(tickers) * epochs_per_ticker
        current_step = 0
        for t in tickers:
            try:
                df = self.fetch_data(ticker=t, period="2y")
                if df is None or df.empty: continue
                gen = self.train_lstm_model(df, epochs=epochs_per_ticker)
                if gen:
                    for epoch, loss in gen:
                        current_step += 1
                        yield current_step, total_steps, t, loss
            except Exception as e:
                print(f"Skipping {t}: {e}")
                
    def get_market_tickers(self):
        tickers = set()
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            response = requests.get(url, headers=headers)
            df = pd.read_html(response.text)[0]
            tickers.update(df['Symbol'].tolist())
        except: tickers.update(["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL"]) 

        try:
            url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
            response = requests.get(url, headers=headers)
            tables = pd.read_html(response.text)
            for t in tables:
                if 'Ticker' in t.columns:
                    tickers.update(t['Ticker'].tolist())
                    break
                elif 'Symbol' in t.columns and len(t) > 50:
                    tickers.update(t['Symbol'].tolist())
                    break
        except: pass
        clean_tickers = [t.replace('.', '-') for t in list(tickers)]
        return list(set(clean_tickers)) 

    def scan_market(self, tickers, limit=1000):
        results = []
        count = 0
        total = min(len(tickers), limit)
        print(f"Starting Scan of {total} tickers...")
        
        for t in tickers[:limit]:
            count += 1
            yield count, total, t 
            
            try:
                df = self.fetch_data(ticker=t, period="1y") 
                if df is None or len(df) < 100: continue
                
                # 1. Train 
                gen = self.train_lstm_model(df, epochs=3)
                if gen:
                    for _ in gen: pass 
                else: continue
                
                # 2. Predict (Technical)
                horizon = self.predict_horizon(df, days=7)
                if not horizon: continue
                
                current_price = df['Close'].iloc[-1]
                next_week_price = horizon[-1]
                tech_return_pct = ((next_week_price - current_price) / current_price) * 100
                
                # 3. Sentiment (Fundamental)
                sentiment_score, headlines = self.analyze_sentiment(t) 
                
                # 4. Hybrid Conviction Score
                # Combine Technical Return with Sentiment
                # If Sentiment is positive, boost prediction. If negative, penalize.
                # Score = Tech_Return + (Sentiment * 5)
                # E.g., 2% return + (0.5 * 5) = 4.5
                conviction = tech_return_pct + (sentiment_score * 5)
                
                results.append({
                    "Ticker": t,
                    "Current Price": current_price,
                    "Predicted 7D": next_week_price,
                    "Tech Return %": tech_return_pct,
                    "News Sentiment": sentiment_score,
                    "Conviction Score": conviction,
                    "Top Headline": headlines[0] if headlines else "No News"
                })
                
            except Exception as e:
                print(f"Failed {t}: {e}")
                
        # Rank by Conviction
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df[results_df["Tech Return %"] < 500] 
            results_df = results_df.sort_values(by="Conviction Score", ascending=False)
            
        return results_df

    def predict_horizon(self, df, days=7):
        if not hasattr(self, 'model'): return []
        self.model.eval()
        current_data = df['Log_Ret'].values.reshape(-1, 1)
        if hasattr(self, 'scaler'):
            current_scaled = self.scaler.transform(current_data)
        else: return [] 
            
        seq = current_scaled[-self.lookback:]
        seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
        predictions = []
        current_price = df['Close'].iloc[-1]
        
        with torch.no_grad():
            for _ in range(days):
                pred_scaled = self.model(seq_tensor)
                pred_val = self.scaler.inverse_transform(pred_scaled.cpu().numpy())[0][0]
                next_price = current_price * np.exp(pred_val)
                predictions.append(next_price)
                new_step = torch.FloatTensor([[pred_scaled.item()]]).to(self.device)
                seq_tensor = torch.cat((seq_tensor[:, 1:, :], new_step.unsqueeze(0)), dim=1)
                current_price = next_price
        return predictions

    def generate_signal(self):
        predictions = self.predict_horizon(self.data, days=5)
        if not predictions: return "HOLD", 0
        
        next_day = predictions[0]
        next_week = predictions[-1]
        current = self.data['Close'].iloc[-1]
        rsi = self.data['RSI'].iloc[-1]
        
        score = 0
        if next_day > current: score += 1
        if next_week > current: score += 1
        if rsi < 35: score += 1
        elif rsi > 65: score -= 1
        
        if score >= 2: return "STRONG BUY", next_day
        elif score >= 1: return "BUY", next_day
        elif score <= -2: return "STRONG SELL", next_day
        elif score <= -1: return "SELL", next_day
        else: return "HOLD", next_day
