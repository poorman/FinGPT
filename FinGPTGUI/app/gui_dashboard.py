import streamlit as st
import plotly.graph_objects as go
from fin_algo import FinGPTTrader
import pandas as pd
import time

st.set_page_config(page_title="FinGPT Trader", layout="wide")

st.title("🧠 FinGPT: Generalized Market Model")

# Initialize Session State
if 'data' not in st.session_state:
    st.session_state.data = None
if 'trader' not in st.session_state:
    st.session_state.trader = FinGPTTrader(ticker="NVDA")
if 'prediction_horizon' not in st.session_state:
    st.session_state.prediction_horizon = []
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

trader = st.session_state.trader

# Sidebar: Global Settings
st.sidebar.header("Global Settings")
trader.ticker = st.sidebar.text_input("Active Ticker", value=trader.ticker)
st.sidebar.info(trader.check_gpu())
model_choice = st.sidebar.selectbox("Prediction Model", options=["LSTM (default)", "TFT (Transformer)"], index=0)
st.session_state.model_choice = "tft" if model_choice.startswith("TFT") else "lstm"

# TABS
tab_single, tab_brain, tab_scanner, tab_alpha = st.tabs(["📉 Single Stock Analysis", "🧠 Market Brain (Mass Training)", "🚀 Opportunity Scanner", "💰 50% Week Hunter"])

# ==========================================
# SUPER TAB 1: Single Stock (Classic)
# ==========================================
with tab_single:
    col_act, col_chart = st.columns([1, 3])
    
    with col_act:
        st.subheader("Actions")
        period = st.selectbox("History Period", ["6mo", "1y", "2y", "5y"], index=1)
        
        if st.button("Fetch Data"):
            with st.spinner("Fetching Data..."):
                try:
                    df = trader.fetch_data(period=period)
                    st.session_state.data = df
                    st.success(f"Loaded {len(df)} days.")
                except Exception as e:
                    st.error(str(e))
                    
        if st.session_state.data is not None:
            st.markdown("---")
            st.write("#### AI Signal (30-Day Outlook)")
            result = trader.generate_signal()
            # Handle both old (2-tuple) and new (3-tuple) return format
            if len(result) == 3:
                signal, pred_price, details = result
            else:
                signal, pred_price = result
                details = {}
            
            # Override prediction price if TFT model selected
            if st.session_state.get('model_choice', 'lstm') == 'tft':
                try:
                    tft_preds = trader.predict_horizon_tft(st.session_state.data, days=30)
                    if tft_preds:
                        pred_price = tft_preds[-1]  # 30-day price
                except Exception as e:
                    st.error(f"TFT prediction error: {e}")
            
            current_price = st.session_state.data['Close'].iloc[-1]
            month_return = ((pred_price - current_price) / current_price) * 100 if pred_price else 0
            
            st.metric("Recommendation", signal, f"{month_return:.2f}% Expected (30d)")
            
            # Show prediction breakdown if available
            if details:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tomorrow", f"${details.get('next_day', 0):.2f}", 
                              f"{details.get('day_return_pct', 0):.2f}%")
                with col2:
                    st.metric("1 Week", f"${details.get('next_week', 0):.2f}",
                              f"{details.get('week_return_pct', 0):.2f}%")
                with col3:
                    st.metric("1 Month", f"${details.get('next_month', 0):.2f}",
                              f"{details.get('month_return_pct', 0):.2f}%")
                st.caption(f"RSI: {details.get('rsi', 0):.1f} | Score: {details.get('score', 0)}")
            else:
                st.metric("30-Day Target", f"${pred_price:.2f}" if pred_price else "N/A")

    with col_chart:
        if st.session_state.data is not None:
            df = st.session_state.data
            
            # Sub-tabs for charts
            c_tab1, c_tab2, c_tab3 = st.tabs(["Price", "RSI", "MACD"])
            
            with c_tab1:
                fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
                
                # Overlay Prediction if available
                if 'prediction_horizon' in st.session_state and st.session_state.prediction_horizon:
                    preds = st.session_state.prediction_horizon
                    # Create future dates
                    last_date = df.index[-1]
                    future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(len(preds))]
                    
                    fig.add_trace(go.Scatter(x=future_dates, y=preds, mode='lines+markers', name='AI Forecast', line=dict(color='orange', dash='dot')))
                    
                fig.update_layout(height=600, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
            with c_tab2: st.line_chart(df['RSI'])
            with c_tab3: st.line_chart(df['MACD'])

# ==========================================
# SUPER TAB 2: Market Brain (Mass Training)
# ==========================================
with tab_brain:
    st.header("Generalized Market Training")
    st.markdown("Train the **SAME** neural network on multiple stocks to learn general market patterns (Log Returns).")
    
    col_sel, col_train = st.columns([1, 2])
    
    with col_sel:
        # 100+ High Volume Stocks, ETFs, and Crypto Proxies
        default_tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD", "INTC", "NFLX",
            "JPM", "V", "PG", "MA", "UNH", "HD", "DIS", "BAC", "XOM", "KO",
            "CSCO", "PEP", "AVGO", "COST", "TMO", "ABBV", "ADBE", "CRM", "WFC", "MCD",
            "NKE", "ABT", "LIN", "DHR", "ACN", "TXN", "NEE", "PM", "ORCL", "UPS",
            "BMY", "RTX", "HON", "UNP", "LOW", "IBM", "QCOM", "SPGI", "CAT", "GS",
            "AMGN", "GE", "INTU", "ISRG", "SBUX", "PLD", "DE", "MS", "BLK", "T",
            "BA", "MMM", "C", "F", "GM", "WMT", "JNJ", "CVX", "MRK",
            "LLY", "PFE", "VZ", "CMCSA", "ADP", "GILD", "MDLZ", "TJX", "AXP", "CVS",
            "SCHW", "UBER", "ABNB", "PYPL", "SQ", "SHOP", "SNOW", "PLTR", "COIN", "ROKU",
            "DKNG", "HOOD", "MSTR", "MARA", "RIOT", "CLSK", "SOFI", "AFRM", "UPST", "AI",
            "SPY", "QQQ", "IWM", "ARKK", "SMH", "XLF", "XLE", "XLK", "XLV", "XLY"
        ]
        
        # Custom Ticker Logic
        if 'custom_tickers' not in st.session_state:
            st.session_state.custom_tickers = []
            
        custom_input = st.text_input("Add Custom Ticker", placeholder="e.g. BTC-USD")
        if custom_input:
            t = custom_input.upper().strip()
            if t and t not in default_tickers and t not in st.session_state.custom_tickers:
                st.session_state.custom_tickers.append(t)
                st.success(f"Added {t}")
                
        all_options = st.session_state.custom_tickers + default_tickers
        selected_tickers = st.multiselect("Select Tickers for Mass Training", all_options, default=all_options[:5])
        
        epochs_per = st.slider("Epochs per Stock", 1, 20, 5)
        
    with col_train:
        if st.button("Start Deep Training Session"):
            progress_bar = st.progress(0)
            status = st.empty()
            chart = st.empty()
            loss_history = []
            
            # Mass Training Generator
            gen = trader.train_on_tickers(selected_tickers, epochs_per_ticker=epochs_per)
            
            for step, total_steps, ticker, loss in gen:
                loss_history.append(loss)
                progress_bar.progress(step / total_steps)
                status.markdown(f"**Training on {ticker}...** (Loss: {loss:.5f})")
                
                if step % 5 == 0:
                    chart.line_chart(loss_history)
                    
            st.success("Market Brain Updated! The model has learned from all selected stocks.")
            
    st.subheader("🔮 Multi-Horizon Forecasting (Next 30 Days)")
    
    col_p_in, col_p_btn = st.columns([3, 1])
    target_stock = col_p_in.text_input("Stock to Predict", value=trader.ticker).upper()
    
    if col_p_btn.button(f"Predict {target_stock}"):
        with st.spinner(f"Fetching data and calculating 30-day forecast for {target_stock}..."):
            try:
                # Fetch fresh data
                df = trader.fetch_data(ticker=target_stock, period="1y")
                if df is not None:
                    # Update global state so charts in Tab 1 work too
                    st.session_state.data = df
                    st.session_state.trader.ticker = target_stock
                    
                    # Predict 30 days
                    if st.session_state.get('model_choice', 'lstm') == 'tft':
                        preds = trader.predict_horizon_tft(df, days=30)
                    else:
                        preds = trader.predict_horizon(df, days=30)
                        
                    st.session_state.prediction_horizon = preds
                    
                    if preds and len(preds) >= 30:
                        # Show key milestones
                        st.subheader("Key Milestones")
                        cols = st.columns(4)
                        milestones = [(0, "Tomorrow"), (6, "1 Week"), (13, "2 Weeks"), (29, "1 Month")]
                        curr = df['Close'].iloc[-1]
                        
                        for i, (idx, label) in enumerate(milestones):
                            with cols[i]:
                                price = preds[idx]
                                change = ((price - curr) / curr) * 100
                                st.metric(label, f"${price:.2f}", f"{change:+.2f}%")
                        
                        # Full 30-day breakdown
                        st.subheader("Daily Predictions")
                        pred_cols = st.columns(6)
                        for i, p in enumerate(preds[:30]):
                            with pred_cols[i % 6]:
                                change = ((p - curr) / curr) * 100
                                st.metric(f"Day {i+1}", f"${p:.2f}", f"{change:+.1f}%")
                        
                        st.info("Check 'Single Stock Analysis' tab for the visual chart.")
                    else:
                        st.warning("Model could not generate 30-day predictions. Try training first.")
                else:
                    st.error(f"Could not fetch data for {target_stock}")
            except Exception as e:
                st.error(f"Prediction Error: {e}")

# ==========================================
# SUPER TAB 3: Opportunity Scanner
# ==========================================
with tab_scanner:
    st.header("🚀 Market Scanner (S&P 500)")
    st.markdown("Scan hundreds of stocks to find the highest predicted gainers for Next Week.")
    
    col_scan_cfg, col_leaderboard = st.columns([1, 2])
    
    with col_scan_cfg:
        st.subheader("Scan Configuration")
        limit = st.number_input("Max Stocks to Scan", min_value=10, max_value=1000, value=50)
        
        st.info("Limit set to 50 for quick testing. Increase to 500 for full S&P scan.")
        
        if st.button("RUN MARKET SCAN"):
            with st.spinner("Initializing Scanner (S&P 500 + NASDAQ 100)..."):
                # 1. Get Tickers
                tickers = trader.get_market_tickers()
                st.write(f"Found {len(tickers)} potential candidates.")
                
                progress = st.progress(0)
                status = st.empty()
                
                # 2. Run Scan
                gen = trader.scan_market(tickers, limit=limit)
                
                # We need to accumulate results from the generator
                # The generator yields status updates, and finally returns something? 
                # No, the method yields progress, and finally returns the dataframe. 
                # Wait, Python generators don't return values like that easily in a loop.
                # Let's fix usage: The scanner yields tuples. We need to collect results differently or modify scanner to store state.
                # Actually, my `scan_market` yields progress. It constructs `results` internally but returns `results_df` at the RETURN statement.
                # In Python `return` in a generator raises StopIteration with value. 
                # Better approach: Pass a callback or accumulate in `gui`.
                # Let's trust the `fin_algo` implementation I wrote: it yields progress. 
                # Ah, I see `yield` and `return`. To get the return value from a generator:
                # result = yield from ... (not here).
                
                # Let's correct `fin_algo` logic conceptually or handle it here. 
                # Since I already wrote `fin_algo`, let's see: 
                # It yields (count, total, t). It returns `results_df`.
                # We can iterate the generator, and capture the return value? 
                # The pythonic way: 
                # scan_gen = trader.scan_market(...)
                # for ... in scan_gen: update_ui()
                # But getting return value is tricky. 
                # I will modify the displayed logic to just rely on the final output IF I can get it.
                # Actually, simpler: I'll assume `scan_market` accumulates results in an attribute or I'll fix the interaction.
                # Wait, I cannot change `fin_algo` easily now without another tool call. 
                # I'll use a wrapper loop.
                
                pass
                
            # EXECUTION
            # Since `return` value of generator is hard to access in loop, 
            # I will assume `fin_algo` is OK but I might need to just run it differently 
            # OR I realized I made a mistake in `fin_algo` design for easy consumption.
            # I'll modify the `scan_market` call to just be a normal function that calls a `yield`ing helper if possible? 
            # Or I can use: 
            # val = None
            # while True: try: val = next(gen); update_ui(val) except StopIteration as e: result = e.value; break
            
            gen = trader.scan_market(tickers, limit=limit)
            result_df = None
            
            while True:
                try:
                    update = next(gen)
                    # update is (count, total, t)
                    count, total, t = update
                    progress.progress(count / total)
                    status.markdown(f"Scanning **{t}** ({count}/{total})...")
                except StopIteration as e:
                    result_df = e.value
                    break
                except Exception as e:
                    st.error(f"Scanner Crash: {e}")
                    break
            
            if result_df is not None and not result_df.empty:
                st.session_state.scan_results = result_df
                st.success("Scan Complete!")
    
    with col_leaderboard:
        if st.session_state.scan_results is not None:
            df = st.session_state.scan_results
            
            st.subheader("🏆 Growth Leaderboard (Next Week)")
            
            # Format for display
            display_df = df.copy()
            display_df['Tech Return %'] = display_df['Tech Return %'].map('{:.2f}%'.format)
            display_df['Predicted 7D'] = display_df['Predicted 7D'].map('${:.2f}'.format)
            display_df['Current Price'] = display_df['Current Price'].map('${:.2f}'.format)
            
            st.dataframe(display_df, height=600)
            
            # CSV Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results CSV", csv, "fingpt_scan_results.csv", "text/csv")

st.markdown("---")
st.caption("Powered by PyTorch Recursive LSTM | Scale-Invariant Log Returns")

# ==========================================
# SUPER TAB 4: 50% Week Hunter (High Alpha)
# ==========================================
with tab_alpha:
    st.header("💰 High Alpha Hunter (Targeting 50%+ Weekly Moves)")
    st.write("This specialized scanner targets **High Volatility** stocks and utilizes **Implied Volatility indicators** to find stocks poised for massive breakout moves.")
    
    col_alpha_act, col_alpha_res = st.columns([1, 2])
    
    with col_alpha_act:
        st.info("Scanner targets: Leveraged ETFs (TQQQ, SOXL), Crypto Proxies (MSTR, COIN), and High Beta Tech.")
        
        if st.button("🔎 HUNT FOR 50% RETURNS"):
            with st.spinner("Analyzing Volatility & Momentum..."):
                 progress_bar = st.progress(0)
                 status_text = st.empty()
                 
                 # Define callback for progress
                 def update_progress(p, msg):
                     progress_bar.progress(p)
                     status_text.text(msg)
                 
                 results = trader.scan_high_alpha_opportunities(progress_callback=update_progress)
                 st.session_state.alpha_results = results
                 st.success(f"Scan Complete. Found {len(results)} opportunities.")

    with col_alpha_res:
        if 'alpha_results' in st.session_state and st.session_state.alpha_results:
             results = st.session_state.alpha_results
             
             if len(results) == 0:
                 st.warning("No stocks met the criteria (High Volatility, Momentum, News). Market might be flat.")
             else:
                 # Display cards for top 3
                 top_3 = results[:3]
                 cols = st.columns(len(top_3))
                 for i, res in enumerate(top_3):
                     with cols[i]:
                         st.markdown(f"### {res['Ticker']}")
                         st.metric("Potential Return", res['Weekly Potential'], delta=res['Daily Volatility'] + " Daily Vol")
                         st.write(f"**News Sentiment:** {res['News Sentiment']:.2f}")
                         st.caption(res['Headline'])
                         
                 st.markdown("### Full Conviction Leaderboard")
                 df_res = pd.DataFrame(results)
                 st.dataframe(df_res[['Ticker', 'Weekly Potential', 'Daily Volatility', 'News Sentiment', 'Conviction']], height=500)
        else:
            st.info("Click 'HUNT' to start the analysis.")
