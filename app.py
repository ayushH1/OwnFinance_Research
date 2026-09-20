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
# 1. PAGE CONFIG & ENTERPRISE STYLING
# ==========================================
st.set_page_config(
    page_title="AlphaTerminal | Enterprise Financial Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Terminal Styling
st.markdown("""
    <style>
        .main {
            background-color: #0B0E14;
        }
        .stMetric {
            background-color: #151922;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #232936;
        }
        div[data-testid="stSidebar"] {
            background-color: #11141C;
            border-right: 1px solid #232936;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
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
# 2. CLIENT & SECRETS INITIALIZATION
# ==========================================
@st.cache_resource
def get_genai_client():
    api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("🔑 Configuration Error: GEMINI_API_KEY missing in Streamlit Secrets.")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_genai_client()

# ==========================================
# 3. DATA ENGINE (yfinance Integration)
# ==========================================
@st.cache_data(ttl=300)
def fetch_financial_data(ticker_symbol: str, period: str = "1y"):
    """Fetches market price, historical candles, key metrics, and financial statements."""
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period=period)
        info = stock.info
        
        # Pull financials safely
        financials = {
            "income": stock.financials,
            "balance": stock.balance_sheet,
            "cashflow": stock.cashflow
        }
        return stock, hist, info, financials
    except Exception as e:
        return None, None, None, None

# ==========================================
# 4. RESILIENT AI SYNTHESIS ENGINE
# ==========================================
def generate_institutional_report(ticker: str, info_dict: dict, max_retries: int = 3):
    """Executes search-grounded institutional thesis generation with error catching."""
    delay = 2
    prompt = f"""
    You are a Senior Managing Director in Equity Research at a top global investment bank.
    Synthesize an institutional-grade, hedge-fund quality research thesis for: {ticker}.

    FINANCIAL CONTEXT PROVIDED:
    - Company: {info_dict.get('longName', ticker)}
    - Sector: {info_dict.get('sector', 'N/A')} | Industry: {info_dict.get('industry', 'N/A')}
    - Current Price: {info_dict.get('currency', 'USD')} {info_dict.get('currentPrice', 'N/A')}
    - Market Cap: {info_dict.get('marketCap', 'N/A')}
    - Trailing P/E: {info_dict.get('trailingPE', 'N/A')} | Forward P/E: {info_dict.get('forwardPE', 'N/A')}
    - EV/EBITDA: {info_dict.get('enterpriseToEbitda', 'N/A')}
    - Operating Margin: {info_dict.get('operatingMargins', 'N/A')}
    - Free Cash Flow: {info_dict.get('freeCashflow', 'N/A')}

    REQUIRED STRUCTURE:
    1. **Executive Investment Thesis** (Core Catalyst, Bull vs. Bear Case)
    2. **Real-Time News & Catalyst Analysis** (Search for recent 30-90 day events, regulatory developments, earnings outcomes)
    3. **Fundamental & Valuation Health** (Assessment of multiples vs historical averages)
    4. **Structural Risks & Competitive Moat**
    5. **Final Institutional Stance & Target Risk Parameters**
    """

    for attempt in range(1, max_retries + 1):
        try:
            config = types.GenerateContentConfig(
                temperature=0.15,
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
# 5. DASHBOARD LAYOUT & CONTROLS
# ==========================================
st.sidebar.title("⚡ AlphaTerminal")
st.sidebar.caption("Institutional Financial Intelligence")

ticker_input = st.sidebar.text_input("ENTER TICKER / SYMBOL", value="RELIANCE.NS").strip().upper()
time_frame = st.sidebar.selectbox("TIMEFRAME", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)

run_button = st.sidebar.button("🚀 EXECUTE RESEARCH PIPELINE", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Ticker Format Guide:**
- **US Equities:** `AAPL`, `MSFT`, `NVDA`
- **India NSE:** `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`
- **India BSE:** `500325.BO`
- **Forex / Crypto:** `EURUSD=X`, `BTC-USD`
""")

if run_button:
    with st.spinner(f"Ingesting exchange metrics and web data for {ticker_input}..."):
        stock, hist, info, financials = fetch_financial_data(ticker_input, period=time_frame)

    if stock is not None and hist is not None and not hist.empty:
        comp_name = info.get('longName', ticker_input)
        currency = info.get('currency', 'USD')
        
        # Header Banner
        st.title(f"{comp_name} ({ticker_input})")
        st.caption(f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')} | Exchange: {info.get('exchange', 'N/A')}")
        
        # Key Metrics Bar
        c1, c2, c3, c4, c5 = st.columns(5)
        m_price = info.get('currentPrice') or hist['Close'].iloc[-1]
        m_cap = f"{info.get('marketCap', 0):,}" if info.get('marketCap') else "N/A"
        pe_ratio = info.get('trailingPE', 'N/A')
        forward_pe = info.get('forwardPE', 'N/A')
        fcf = f"{info.get('freeCashflow', 0):,}" if info.get('freeCashflow') else "N/A"

        c1.metric("Live Price", f"{currency} {m_price:,.2f}")
        c2.metric("Market Cap", f"{m_cap}")
        c3.metric("Trailing P/E", f"{pe_ratio}")
        c4.metric("Forward P/E", f"{forward_pe}")
        c5.metric("Free Cash Flow", f"{fcf}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Main Workspace Navigation Tabs
        tab_chart, tab_ai, tab_financials, tab_profile = st.tabs([
            "📈 Interactive Chart & Volume", 
            "🧠 AI Research Memo", 
            "📊 Financial Statements", 
            "🏢 Company Profile"
        ])

        # TAB 1: Advanced Charting
        with tab_chart:
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.03, 
                row_heights=[0.75, 0.25]
            )
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name="OHLC"
            ), row=1, col=1)
            
            # Volume
            fig.add_trace(go.Bar(
                x=hist.index,
                y=hist['Volume'],
                name="Volume",
                marker_color='#232936'
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

        # TAB 2: AI Institutional Research Memo
        with tab_ai:
            st.subheader("Search-Grounded Equity Research Thesis")
            with st.status("Gathering live market grounding and executing analysis...", expanded=True) as status:
                ai_memo = generate_institutional_report(ticker_input, info)
                status.update(label="Research Synthesis Complete", state="complete", expanded=False)
            
            st.markdown(ai_memo)

        # TAB 3: Financial Statements
        with tab_financials:
            st.subheader("Exchange Financial Statements")
            f_option = st.radio("Statement Type", ["Income Statement", "Balance Sheet", "Cash Flow"], horizontal=True)
            
            if f_option == "Income Statement" and financials["income"] is not None:
                st.dataframe(financials["income"], use_container_width=True)
            elif f_option == "Balance Sheet" and financials["balance"] is not None:
                st.dataframe(financials["balance"], use_container_width=True)
            elif f_option == "Cash Flow" and financials["cashflow"] is not None:
                st.dataframe(financials["cashflow"], use_container_width=True)
            else:
                st.info("Financial statements unavailable for this symbol.")

        # TAB 4: Profile & Governance
        with tab_profile:
            st.subheader("Business Summary")
            st.write(info.get('longBusinessSummary', 'No description available.'))
            
            st.markdown("---")
            st.subheader("Key Corporate Metrics")
            p1, p2, p3 = st.columns(3)
            p1.write(f"**Profit Margin:** {info.get('profitMargins', 'N/A')}")
            p2.write(f"**Return on Equity:** {info.get('returnOnEquity', 'N/A')}")
            p3.write(f"**Debt to Equity:** {info.get('debtToEquity', 'N/A')}")

    else:
        st.error(f"Unable to retrieve data for ticker '{ticker_input}'. Please check the symbol syntax (e.g., use `.NS` suffix for NSE India stocks).")
else:
    st.info("Enter a stock symbol in the left control panel and select **EXECUTE RESEARCH PIPELINE**.")
