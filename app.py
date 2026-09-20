import os
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError

# ==========================================
# 1. ENTERPRISE UI CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AlphaPulse | Enterprise Indian Market Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Bloomberg/Dark Terminal Styling
st.markdown("""
    <style>
        .main { background-color: #07090E; }
        .stMetric {
            background-color: #0F131C;
            padding: 14px 18px;
            border-radius: 8px;
            border: 1px solid #1E2536;
        }
        div[data-testid="stSidebar"] {
            background-color: #0B0E14;
            border-right: 1px solid #1E2536;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #0F131C;
            border-radius: 4px;
            padding: 10px 20px;
            color: #8A94A6;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1E2536 !important;
            color: #00E5FF !important;
            border-bottom: 2px solid #00E5FF !important;
        }
        .signal-card-buy {
            background-color: #0A291D;
            border: 1px solid #00E676;
            padding: 15px;
            border-radius: 8px;
            color: #00E676;
        }
        .signal-card-sell {
            background-color: #331018;
            border: 1px solid #FF5252;
            padding: 15px;
            border-radius: 8px;
            color: #FF5252;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CLIENT & SECRETS INITIALIZATION
# ==========================================
@st.cache_resource
def init_genai_client():
    api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("🔑 Configuration Error: GEMINI_API_KEY is missing in Streamlit Secrets.")
        st.stop()
    return genai.Client(api_key=api_key)

client = init_genai_client()

# ==========================================
# 3. QUANTITATIVE DATA ENGINE (NSE/BSE & F&O)
# ==========================================
@st.cache_data(ttl=120)
def fetch_market_engine(ticker_symbol: str, period: str = "1y"):
    """
    Retrieves exchange price data, technical indicators, and option chains.
    Calculates technical parameters locally to avoid missing API values.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            return None, None, None, None

        info = stock.info if isinstance(stock.info, dict) else {}
        
        # Hard Technical Calculations
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        
        # Relative Strength Index (RSI 14)
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))

        latest_close = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else latest_close
        price_change = latest_close - prev_close
        pct_change = (price_change / prev_close) * 100

        # Extract Key Quantitative Metrics
        metrics = {
            "name": info.get('longName') or info.get('shortName') or ticker_symbol,
            "sector": info.get('sector', 'N/A'),
            "industry": info.get('industry', 'N/A'),
            "price": info.get('currentPrice') or info.get('regularMarketPrice') or latest_close,
            "change": price_change,
            "pct_change": pct_change,
            "rsi": hist['RSI'].iloc[-1] if not np.isnan(hist['RSI'].iloc[-1]) else 50.0,
            "volume": hist['Volume'].iloc[-1],
            "52w_high": hist['High'].max(),
            "52w_low": hist['Low'].min(),
            "summary": info.get('longBusinessSummary', 'No corporate profile available.')
        }

        # Retrieve Option Chain if available (for F&O stocks and Indices)
        options_data = None
        try:
            expirations = stock.options
            if expirations:
                opt_chain = stock.option_chain(expirations[0])
                options_data = {
                    "expiration": expirations[0],
                    "calls": opt_chain.calls,
                    "puts": opt_chain.puts
                }
        except Exception:
            options_data = None

        return hist, metrics, options_data, stock
    except Exception:
        return None, None, None, None

# ==========================================
# 4. RESILIENT AI SYNTHESIS ENGINE
# ==========================================
def generate_market_intelligence(ticker: str, metrics: dict, max_retries: int = 3):
    """Executes live web-grounded research synthesis for intraday and positional setups."""
    delay = 2
    prompt = f"""
    You are an Institutional Head of Trading Strategy for Indian Equity & Derivatives markets (NSE/BSE).
    Analyze ticker: {ticker}.

    LIVE MARKET PARAMETERS:
    - Company: {metrics['name']}
    - Sector: {metrics['sector']}
    - Spot Price: INR {metrics['price']:,.2f} ({metrics['pct_change']:+.2f}%)
    - 14-Period RSI: {metrics['rsi']:.2f}
    - 52-Week Range: INR {metrics['52w_low']:,.2f} - INR {metrics['52w_high']:,.2f}

    REQUIRED STRUCTURE:
    1. **Quant Setup & Trend Assessment** (Identify support/resistance zones, RSI momentum, moving average orientation)
    2. **Real-Time Catalyst & News Grounding** (Search the web for events from the last 14-30 days, FII/DII flow context, quarterly earnings updates)
    3. **Derivatives & F&O Strategic Positioning** (Recommended Futures or Options Strategy: e.g., Bull Call Spread, Iron Condor, Long Futures with hedge)
    4. **Key Risk Parameters** (Invalidation level/Stop Loss and Upside Target Ratios)
    """

    models_to_try = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.5-pro"]

    for model_name in models_to_try:
        for attempt in range(1, max_retries + 1):
            try:
                config = types.GenerateContentConfig(
                    temperature=0.15,
                    max_output_tokens=4096,
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config
                )
                return response.text
            except ClientError as e:
                error_code = getattr(e, 'code', 'UNKNOWN')
                if error_code in [404, 400]:
                    break  # Try next model in sequence
                if error_code == 429 and attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                if model_name == models_to_try[-1]:
                    st.error(f"❌ Gemini Service Error [{error_code}]: {getattr(e, 'message', str(e))}")
                    st.stop()
            except APIError:
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue

# ==========================================
# 5. DASHBOARD LAYOUT & USER INTERFACE
# ==========================================
st.sidebar.title("⚡ AlphaPulse")
st.sidebar.caption("Institutional Market Terminal")

ticker_input = st.sidebar.text_input("SYMBOL / TICKER (NSE/BSE)", value="RELIANCE.NS").strip().upper()
time_frame = st.sidebar.selectbox("CHART TIMEFRAME", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
run_pipeline = st.sidebar.button("🚀 EXECUTE LIVE ANALYSIS", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Supported Formats:**
- **NSE Stocks:** `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`
- **Indices:** `^NSEI` (Nifty 50), `^NSEBANK` (Bank Nifty)
- **BSE Stocks:** `500325.BO`
""")

if run_pipeline:
    with st.spinner(f"Ingesting exchange data and options context for {ticker_input}..."):
        hist, metrics, options_data, stock_obj = fetch_market_engine(ticker_input, period=time_frame)

    if hist is not None and metrics is not None:
        # Header Metrics Banner
        st.title(f"{metrics['name']} ({ticker_input})")
        st.caption(f"Sector: {metrics['sector']} | Industry: {metrics['industry']}")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Spot Price", f"₹{metrics['price']:,.2f}", f"{metrics['pct_change']:+.2f}%")
        c2.metric("RSI (14)", f"{metrics['rsi']:.1f}")
        c3.metric("52-Wk High", f"₹{metrics['52w_high']:,.2f}")
        c4.metric("52-Wk Low", f"₹{metrics['52w_low']:,.2f}")
        
        # Technical Signal Indicator
        signal_type = "BULLISH MOMENTUM" if metrics['rsi'] > 55 else ("BEARISH PRESSURE" if metrics['rsi'] < 45 else "NEUTRAL RANGE")
        c5.metric("Technical Bias", signal_type)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabbed Workspace
        tab_ai, tab_charts, tab_fo, tab_statements = st.tabs([
            "🧠 AI Market Intelligence", 
            "📈 Interactive Technicals", 
            "🎯 F&O & Option Chain", 
            "📊 Financial Statements"
        ])

        # TAB 1: AI Market Intelligence Memo
        with tab_ai:
            st.subheader("Live Grounded Market Strategy Memo")
            with st.status("Gathering real-time market grounding and evaluating derivative positioning...", expanded=True) as status:
                ai_memo = generate_market_intelligence(ticker_input, metrics)
                status.update(label="Analysis Complete!", state="complete", expanded=False)
            st.markdown(ai_memo)

        # TAB 2: Advanced Interactive Technical Charting
        with tab_charts:
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.03, 
                row_heights=[0.75, 0.25]
            )
            
            # Candlesticks
            fig.add_trace(go.Candlestick(
                x=hist.index, open=hist['Open'], high=hist['High'],
                low=hist['Low'], close=hist['Close'], name="OHLC"
            ), row=1, col=1)
            
            # Moving Averages
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_20'], name="20 SMA", line=dict(color='#00E5FF', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], name="50 SMA", line=dict(color='#FFD700', width=1.5)), row=1, col=1)

            # Volume Bar Chart
            fig.add_trace(go.Bar(
                x=hist.index, y=hist['Volume'], name="Volume", marker_color='#1E2536'
            ), row=2, col=1)

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_rangeslider_visible=False,
                height=580,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        # TAB 3: Futures & Options Analytics
        with tab_fo:
            st.subheader("Derivatives & Option Chain Matrix")
            if options_data:
                st.caption(f"Nearest Expiration Window: {options_data['expiration']}")
                col_calls, col_puts = st.columns(2)
                
                with col_calls:
                    st.markdown("##### **Call Options (CE)**")
                    st.dataframe(
                        options_data['calls'][['strike', 'lastPrice', 'bid', 'ask', 'openInterest', 'impliedVolatility']].tail(10),
                        use_container_width=True
                    )
                
                with col_puts:
                    st.markdown("##### **Put Options (PE)**")
                    st.dataframe(
                        options_data['puts'][['strike', 'lastPrice', 'bid', 'ask', 'openInterest', 'impliedVolatility']].tail(10),
                        use_container_width=True
                    )
            else:
                st.info("Option Chain matrix is unavailable for this ticker or the security is not traded in the F&O segment.")

        # TAB 4: Financial Statements
        with tab_statements:
            st.subheader("Exchange Financial Statements")
            if stock_obj is not None:
                st_option = st.radio("Select Statement", ["Income Statement", "Balance Sheet", "Cash Flow"], horizontal=True)
                if st_option == "Income Statement" and isinstance(stock_obj.financials, pd.DataFrame):
                    st.dataframe(stock_obj.financials, use_container_width=True)
                elif st_option == "Balance Sheet" and isinstance(stock_obj.balance_sheet, pd.DataFrame):
                    st.dataframe(stock_obj.balance_sheet, use_container_width=True)
                elif st_option == "Cash Flow" and isinstance(stock_obj.cashflow, pd.DataFrame):
                    st.dataframe(stock_obj.cashflow, use_container_width=True)
                else:
                    st.info("Statement data unavailable for this ticker.")

    else:
        st.error(f"Failed to fetch market data for '{ticker_input}'. Verify ticker syntax (e.g., use `.NS` for NSE stocks).")
else:
    st.info("Enter a stock or index symbol in the sidebar and select **EXECUTE LIVE ANALYSIS**.")
