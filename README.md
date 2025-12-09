# 🧠 FinGPT: Financial AI Agent

> **An advanced, containerized AI agent for real-time stock analysis, multi-model forecasting, and high-alpha opportunity hunting.**

![FinGPT Demo](generated_image_placeholder.png)

## 🚀 Features

### 1. **Multi-Model Prediction Engine**
- **LSTM (Long Short-Term Memory)**: Captures sequential dependencies in daily price movements.
- **Transfomer (TFT)**: A lightweight Temporal Fusion Transformer for attention-based time series forecasting.
- **Hybrid "Week Hunter" Strategy**: Combines models to identify stocks with >50% weekly upside potential.

### 2. **Interactive Streamlit Dashboard**
- **Single Stock Deep Dive**: Real-time charts, RSI/MACD indicators, and AI prediction overlays.
- **Market Brain**: Mass training capability to let the model learn generic market physics from multiple tickers.
- **Opportunity Scanner**: Auto-scans the S&P 500/NASDAQ for the best Conviction setups.
- **50% Week Hunter**: Specialized high-volatility scanner for aggressive growth opportunities.

### 3. **Robust Backend**
- **Dockerized Environment**: One-click deployment with `docker-compose`.
- **GPU Acceleration**: CUDA-enabled PyTorch for fast training and inference.
- **Polygon.io Integration**: High-fidelity market data fetching.
- **Sentiment Analysis**: VADER-based news sentiment scoring.

---

## 🛠️ Installation

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU (Optional, for acceleration)
- Polygon.io API Key

### Quick Start

1. **Clone the Repo**
   ```bash
   git clone https://github.com/poorman/FinGPT.git
   cd FinGPT
   ```

2. **Configure Environment**
   Create a `.env` file in the root directory:
   ```bash
   POLYGON_API_KEY=your_api_key_here
   ```

3. **Launch the Stack**
   ```bash
   docker-compose up -d --build
   ```

4. **Access the Dashboard**
   Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

---

## 🖥️ Usage Guide

### **1. Single Stock Analysis**
- Enter a ticker (e.g., `NVDA`).
- Select a historical period (`1y`, `2y`).
- View the AI's **Next Day Target** and **Signal** (BUY/SELL/HOLD).

### **2. Switching Models**
- Use the **Sidebar Selector** to switch between `LSTM` (Default) and `TFT` (Transformer).
- Compare predictions to confirm trends.

### **3. The "50% Week Hunter"**
- Go to the **High Alpha Hunter** tab.
- Click **HUNT**.
- The AI will scan high-volatility tickers and rank them by **Conviction Score** (Volatility + AI Confidence + Sentiment).

---

## 🏗️ Project Structure

```
FinGPT/
├── FinGPTGUI/
│   ├── app/
│   │   ├── gui_dashboard.py    # Streamlit Frontend
│   │   ├── fin_algo.py         # AI Backend (LSTM, TFT, Data)
│   │   ├── requirements.txt    # Python Dependencies
│   │   └── Dockerfile          # GUI Container Config
│   └── ...
├── docker-compose.yml          # Service Orchestration
└── README.md                   # This file
```

## ⚠️ Disclaimer
*This software is for educational and research purposes only. It is not financial advice. Trading stocks involves risk.*
