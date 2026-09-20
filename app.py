import os
import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError

# ==========================================
# 1. PAGE LAYOUT & STYLING
# ==========================================
st.set_page_config(
    page_title="Terminal | Enterprise Financial Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .metric-card {
            background-color: #0e1117;
            border: 1px solid #262730;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .stButton>button {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CLIENT & SECRETS INITIALIZATION
# ==========================================
@st.cache_resource
def get_genai_client():
    api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("🔑 Configuration Error: GEMINI_API_KEY not found in Streamlit Secrets.")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_genai_client()

# ==========================================
# 3. DATA FETCHING (yfinance Engine)
# ==========================================
def fetch_stock_data(ticker_symbol: str, period: str = "1y"):
    """Fetch raw numerical market data directly from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period=period)
        info = stock.info
        return stock, hist, info
    except Exception as e:
        st.error(f"Failed to retrieve data for '{ticker_symbol}': {str(e)}")
        return None, None, None

# ==========================================
# 4. RESILIENT GENAI RESEARCH PIPELINE
# ==========================================
def generate_institutional_report(ticker: str, info_dict: dict, max_retries: int = 3):
    """
    Combines hard fundamental metrics with real-time Google Search grounding
    to produce an executive research memo.
    """
    delay = 2
    prompt = f"""
    You are an Managing Director at an institutional equity research firm. 
    Synthesize an exhaustive, hedge-fund quality research memo for ticker: {ticker}.

    HARD FUNDAMENTAL DATA (FROM EXCHANGE):
    - Company: {info_dict.get('longName', ticker)}
    - Sector: {info_dict.get('sector', 'N/A')} | Industry: {info_dict.get('industry', 'N/A')}
    - Current Price: {info_dict.get('currency', 'USD')} {info_dict.get('currentPrice', 'N/A')}
    - Market Cap: {info_dict.get('marketCap', 'N/A')}
    - Trailing P/E: {info_dict.get('trailingPE', 'N/A')} | Forward P/E: {info_dict.get('forwardPE', 'N/A')}
    - Enterprise Value / EBITDA: {info_dict.get('enterpriseToEbitda', 'N/A')}
    - Profit Margins: {info_dict.get('profitMargins', 'N/A')}
    - Return on Equity (ROE): {info_dict.get('returnOnEquity', 'N/A')}

    REQUIRED ANALYSIS STRUCTURE:
    1. **Executive Summary & Investment Thesis** (Bull vs. Bear Case)
    2. **Real-Time Catalysts & Recent News** (Search the web for news from the last 30-90 days, earnings call takeaways, and management updates)
    3. **Valuation & Financial Health Analysis** (Assess P/E, EBITDA multiples relative to historical averages)
    4. **Competitive Moat & Industry Risks** (Macroeconomic environment, regulatory threats, supply chain factors)
    5. **Final Institutional Rating & Outlook** (BUY / HOLD / SELL classification with key risks to monitor)
    """

    for attempt in range(1, max_retries + 1):
        try:
            # Configure LLM with real-time Web Search Grounding enabled
            config = types.GenerateContentConfig(
                temperature=0.2, # Precise, factual output
                max_output_tokens=4096,
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=config
            )
            return response.text

        except ClientError as e:
            error_code = getattr(e, 'code', 'UNKNOWN')
            error_msg = getattr(e, 'message', str(e))
            
            if error_code == 429 and attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            st.error(f"❌ Gemini API Error [{error_code}]: {error_msg}")
            st.stop()

        except APIError as e:
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            st.error(f"❌ Google Service Error: {str(e)}")
            st.stop()

# ==========================================
# 5. USER INTERFACE & CONTROL PANEL
# ==========================================
st.sidebar.title("🔍 Research Terminal")
ticker_input = st.sidebar.text_input("Ticker Symbol or Company", value="RELIANCE.NS").strip().upper()
time_frame = st.sidebar.selectbox("Historical Chart Window", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
run_analysis = st.sidebar.button("🚀 GENERATE RESEARCH MEMO", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Supported Ticker Formats:**
- **US Stocks:** `AAPL`, `NVDA`, `TSLA`, `MSFT`
- **Indian Stocks (NSE):** `RELIANCE.NS`, `TCS.NS`, `INFY.NS`
- **Indian Stocks (BSE):** `500325.BO`
""")

# Main Content Dashboard
if run_analysis:
    with st.spinner(f"Fetching Exchange Data and Web Context for {ticker_input}..."):
        stock, hist, info = fetch_stock_data(ticker_input, period=time_frame)

    if stock is not None and hist is not None and not hist.empty:
        # ----------------------------------
        # Header & High-Level Metrics
        # ----------------------------------
        comp_name = info.get('longName', ticker_input)
        curr = info.get('currency', 'USD')
        
        st.title(f"{comp_name} ({ticker_input})")
        st.caption(f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}")

        col1, col2, col3, col4 = st.columns(4)
        
        m_price = info.get('currentPrice') or (hist['Close'].iloc[-1] if not hist.empty else "N/A")
        m_cap = f"{info.get('marketCap', 0):,}" if info.get('marketCap') else "N/A"
        pe_ratio = info.get('trailingPE', 'N/A')
        beta = info.get('beta', 'N/A')

        col1.metric("Current Price", f"{curr} {m_price}")
        col2.metric("Market Cap", f"{m_cap}")
        col3.metric("Trailing P/E", f"{pe_ratio}")
        col4.metric("Beta", f"{beta}")

        st.markdown("---")

        # ----------------------------------
        # Historical Interactive Candlestick Chart
        # ----------------------------------
        st.subheader("📈 Historical Price Action & Volume")
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name="Price"
        ))
        
        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=450,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ----------------------------------
        # AI Executive Memo Execution
        # ----------------------------------
        st.subheader("🧠 Institutional AI Research Synthesis")
        
        with st.status("Gathering live web search grounding & synthesizing report...", expanded=True) as status:
            report_markdown = generate_institutional_report(ticker_input, info)
            status.update(label="Research Synthesis Completed!", state="complete", expanded=False)

        st.markdown(report_markdown)

    else:
        st.warning("No financial data found. Check ticker syntax (e.g., use `.NS` for NSE India tickers like `TCS.NS`).")
else:
    st.info("Enter a stock ticker on the sidebar and click **GENERATE RESEARCH MEMO** to run the terminal.")
