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

# TABS
tab_single, tab_brain, tab_scanner = st.tabs(["📉 Single Stock Analysis", "🧠 Market Brain (Mass Training)", "🚀 Opportunity Scanner"])

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
             st.write("#### AI Signal")
             signal, pred_price = trader.generate_signal()
             
             delta = 0
             current_price = st.session_state.data['Close'].iloc[-1]
             if pred_price:
                 delta = ((pred_price - current_price) / current_price) * 100
             
             st.metric("Recommendation", signal, f"{delta:.2f}% Expected")
             st.metric("Next Day Target", f"${pred_price:.2f}")

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
        default_tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "AMD", "INTC"]
        selected_tickers = st.multiselect("Select Tickers for Mass Training", default_tickers, default=default_tickers[:3])
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
            
    st.markdown("---")
    st.subheader("🔮 Multi-Horizon Forecasting")
    
    if st.button("Generate Future Forecast (Next Week)"):
        if st.session_state.data is None:
            st.error("Please fetch data in Single Stock tab first.")
        else:
            with st.spinner("Recursive Inferencing..."):
                preds = trader.predict_horizon(st.session_state.data, days=7)
                st.session_state.prediction_horizon = preds
                
                cols = st.columns(7)
                for i, p in enumerate(preds):
                    with cols[i]:
                        st.metric(f"Day +{i+1}", f"${p:.2f}")
                        
                st.info("Check 'Price Chart' in Tab 1 to see the visualization.")

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
            display_df['Expected Return %'] = display_df['Expected Return %'].map('{:.2f}%'.format)
            display_df['Predicted 7D'] = display_df['Predicted 7D'].map('${:.2f}'.format)
            display_df['Current Price'] = display_df['Current Price'].map('${:.2f}'.format)
            
            st.dataframe(display_df, height=600)
            
            # CSV Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results CSV", csv, "fingpt_scan_results.csv", "text/csv")

st.markdown("---")
st.caption("Powered by PyTorch Recursive LSTM | Scale-Invariant Log Returns")
