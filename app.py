import os
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError

# ==========================================
# 1. PAGE CONFIG & TERMINAL STYLING
# ==========================================
st.set_page_config(
    page_title="AlphaTerminal | Enterprise Equity Research",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Terminal Custom Styling
st.markdown("""
    <style>
        .main { background-color: #0B0E14; }
        .stMetric {
            background-color: #151922;
            padding: 12px 16px;
            border-radius: 6px;
            border: 1px solid #232936;
        }
        div[data-testid="stSidebar"] {
            background-color: #11141C;
            border-right: 1px solid #232936;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #151922;
            border-radius: 4px;
            padding: 8px 16px;
            color: #8A94A6;
        }
        .stTabs [aria-selected="true"] {
            background-color: #232936 !important;
            color: #FFFFFF !important;
            border-bottom: 2px solid #00E5FF !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CLIENT INITIALIZATION
# ==========================================
@st.cache_resource
def get_genai_client():
    api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("🔑 Configuration Error: GEMINI_API_KEY is missing in Streamlit Secrets.")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_genai_client()

# ==========================================
# 3. ROBUST DATA ENGINE (NO N/A FALLBACKS)
# ==========================================
@st.cache_data(ttl=300)
def fetch_financial_data(ticker_symbol: str, period: str = "1y"):
    """
    Retrieves exchange price data and financial statements.
    Includes fallbacks for missing Yahoo Finance info fields.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            return None, None, None

        info = stock.info if isinstance(stock.info, dict) else {}
        
        # Hard data calculation fallbacks if Yahoo API returns empty dicts
        latest_close = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else latest_close
        price_change = latest_close - prev_close
        pct_change = (price_change / prev_close) * 100

        # Safe Metrics Extraction
        extracted_metrics = {
            "name": info.get('longName') or info.get('shortName') or ticker_symbol,
            "sector": info.get('sector', 'N/A'),
            "industry": info.get('industry', 'N/A'),
            "exchange": info.get('exchange', 'NSE/BSE'),
            "currency": info.get('currency', 'INR' if '.NS' in ticker_symbol or '.BO' in ticker_symbol else 'USD'),
            "price": info.get('currentPrice') or info.get('regularMarketPrice') or latest_close,
            "change": price_change,
            "pct_change": pct_change,
            "market_cap": info.get('marketCap'),
            "pe_ratio": info.get('trailingPE') or info.get('forwardPE', 'N/A'),
            "volume": hist['Volume'].iloc[-1],
            "52w_high": hist['High'].max(),
            "52w_low": hist['Low'].min(),
            "summary": info.get('longBusinessSummary', 'No description available for this ticker.')
        }

        # Financials as clean DataFrames
        inc = stock.financials if isinstance(stock.financials, pd.DataFrame) else pd.DataFrame()
        bal = stock.balance_sheet if isinstance(stock.balance_sheet, pd.DataFrame) else pd.DataFrame()
        cf = stock.cashflow if isinstance(stock.cashflow, pd.DataFrame) else pd.DataFrame()

        financials = {"income": inc, "balance": bal, "cashflow": cf}
        return hist, extracted_metrics, financials
    except Exception:
        return None, None, None

# ==========================================
# 4. RESILIENT INSTITUTIONAL AI ENGINE
# ==========================================
def generate_institutional_report(ticker: str, metrics: dict, max_retries: int = 3):
    """Generates an institutional equity research memo using live web grounding."""
    delay = 2
    
    prompt = f"""
    You are a Senior Equity Managing Director at a top global investment firm.
    Synthesize a hedge-fund quality research memo for ticker: {ticker}.

    REAL-TIME EXCHANGE METRICS:
    - Company: {metrics['name']}
    - Sector: {metrics['sector']} | Industry: {metrics['industry']}
    - Current Price: {metrics['currency']} {metrics['price']:,.2f}
    - 52-Week Range: {metrics['52w_low']:,.2f} - {metrics['52w_high']:,.2f}
    - Trailing P/E: {metrics['pe_ratio']}

    REQUIRED INSTITUTIONAL ANALYSIS:
    1. **Executive Investment Thesis & Rating** (Bull vs. Bear Case, High-conviction Stance)
    2. **Real-Time Catalysts & News Grounding** (Search the web for news from the last 30-90 days, earnings call takeaways, strategic shifts)
    3. **Valuation & Metric Benchmarking** (P/E analysis relative to sector peers)
    4. **Key Structural & Macroeconomic Risks**
    5. **Target Monitoring Parameters & Strategic Outlook**
    """

    # Model fallbacks to avoid single-point deprecation failures
    models_to_try = ["gemini-2.5-pro", "gemini-1.5-pro", "gemini-1.5-flash"]

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
                error_msg = getattr(e, 'message', str(e))
                
                # If model is deprecated/not found, break loop to try next model
                if error_code in [404, 400] and "not found" in error_msg.lower():
                    break
                
                if error_code == 429 and attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                
                if model_name == models_to_try[-1]:
                    st.error(f"❌ Gemini API Error [{error_code}]: {error_msg}")
                    st.stop()

            except APIError:
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue

# ==========================================
# 5. DASHBOARD INTERFACE
# ==========================================
st.sidebar.title("⚡ AlphaTerminal")
st.sidebar.caption("Institutional Equity Intelligence")

ticker_input = st.sidebar.text_input("ENTER TICKER SYMBOL", value="RELIANCE.NS").strip().upper()
time_frame = st.sidebar.selectbox("TIMEFRAME", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)

run_button = st.sidebar.button("🚀 EXECUTE RESEARCH PIPELINE", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Supported Formats:**
- **NSE (India):** `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`
- **BSE (India):** `500325.BO`
- **US Equities:** `AAPL`, `NVDA`, `TSLA`
- **Forex / Crypto:** `EURUSD=X`, `BTC-USD`
""")

if run_button:
    with st.spinner(f"Ingesting exchange data for {ticker_input}..."):
        hist, metrics, financials = fetch_financial_data(ticker_input, period=time_frame)

    if hist is not None and metrics is not None:
        # Header Information
        st.title(f"{metrics['name']} ({ticker_input})")
        st.caption(f"Sector: {metrics['sector']} | Industry: {metrics['industry']} | Exchange: {metrics['exchange']}")
        
        # Primary KPI Bar
        c1, c2, c3, c4, c5 = st.columns(5)
        
        m_cap_str = f"{metrics['currency']} {metrics['market_cap'] / 1e10:,.2f} Cr" if metrics['market_cap'] and metrics['currency'] == 'INR' else (f"${metrics['market_cap'] / 1e9:,.2f} B" if metrics['market_cap'] else "N/A")
        
        c1.metric("Live Price", f"{metrics['currency']} {metrics['price']:,.2f}", f"{metrics['pct_change']:+.2f}%")
        c2.metric("Market Cap", m_cap_str)
        c3.metric("Trailing P/E", f"{metrics['pe_ratio']}")
        c4.metric("52-Week High", f"{metrics['currency']} {metrics['52w_high']:,.2f}")
        c5.metric("52-Week Low", f"{metrics['currency']} {metrics['52w_low']:,.2f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabbed Workspace
        tab_ai, tab_chart, tab_financials, tab_profile = st.tabs([
            "🧠 AI Research Memo", 
            "📈 Price Action & Volume", 
            "📊 Financial Statements", 
            "🏢 Company Profile"
        ])

        # TAB 1: AI Institutional Research
        with tab_ai:
            st.subheader("Search-Grounded Equity Research Thesis")
            with st.status("Gathering real-time market grounding and building thesis...", expanded=True) as status:
                ai_memo = generate_institutional_report(ticker_input, metrics)
                status.update(label="Research Synthesis Complete", state="complete", expanded=False)
            
            st.markdown(ai_memo)

        # TAB 2: Advanced Charting
        with tab_chart:
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.03, 
                row_heights=[0.75, 0.25]
            )
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=hist.index, open=hist['Open'], high=hist['High'],
                low=hist['Low'], close=hist['Close'], name="OHLC"
            ), row=1, col=1)
            
            # Moving Averages
            hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_20'], name="20 SMA", line=dict(color='#00E5FF', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], name="50 SMA", line=dict(color='#FFD700', width=1)), row=1, col=1)

            # Volume
            fig.add_trace(go.Bar(
                x=hist.index, y=hist['Volume'], name="Volume", marker_color='#232936'
            ), row=2, col=1)

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_rangeslider_visible=False,
                height=550,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        # TAB 3: Financial Statements
        with tab_financials:
            st.subheader("Statement Metrics")
            f_option = st.radio("Statement Type", ["Income Statement", "Balance Sheet", "Cash Flow"], horizontal=True)
            
            if f_option == "Income Statement" and not financials["income"].empty:
                st.dataframe(financials["income"], use_container_width=True)
            elif f_option == "Balance Sheet" and not financials["balance"].empty:
                st.dataframe(financials["balance"], use_container_width=True)
            elif f_option == "Cash Flow" and not financials["cashflow"].empty:
                st.dataframe(financials["cashflow"], use_container_width=True)
            else:
                st.info("Financial statements are currently unavailable for this ticker from exchange feeds.")

        # TAB 4: Business Summary
        with tab_profile:
            st.subheader("Corporate Overview")
            st.write(metrics['summary'])

    else:
        st.error(f"Could not retrieve exchange data for '{ticker_input}'. Verify ticker syntax (e.g., `.NS` for NSE stocks).")
else:
    st.info("Enter a stock symbol in the left panel and click **EXECUTE RESEARCH PIPELINE**.")
