import os
import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from duckduckgo_search import DDGS
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from markdown_pdf import MarkdownPdf, Section

# ==========================================
# STREAMLIT APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="FinAI | Institutional Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark institutional theme styling
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .metric-card {
        background-color: #1e222d;
        border: 1px solid #2a2e39;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-title { font-size: 13px; color: #848e9c; margin-bottom: 4px; }
    .metric-value { font-size: 22px; font-weight: bold; color: #f0f3fa; }
    .metric-sub { font-size: 12px; font-weight: 500; }
    .positive { color: #089981; }
    .negative { color: #f23645; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.title("⚙️ Engine Settings")

api_key = st.sidebar.text_input("Gemini API Key:", type="password")
if not api_key:
    api_key = os.environ.get("GEMINI_API_KEY", "")

if not api_key:
    st.sidebar.warning("⚠️ Please provide a Gemini API Key to proceed.")

MODEL_ID = st.sidebar.selectbox("Model Endpoint:", ["gemini-3.6-flash", "gemini-flash-latest"])

# ==========================================
# AGENT TOOL DEFINITIONS
# ==========================================
def identify_company(query: str) -> dict:
    search_term = query.strip()
    if "reliance" in search_term.lower():
        return {"symbol": "RELIANCE.NS", "name": "Reliance Industries Limited", "exchange": "NSE"}
    try:
        ticker = yf.Ticker(search_term)
        info = ticker.info
        symbol = info.get("symbol", f"{search_term.upper()}.NS")
        name = info.get("shortName", search_term)
        return {"symbol": symbol, "name": name, "exchange": info.get("exchange", "NSE")}
    except Exception:
        return {"symbol": f"{search_term.upper()}.NS", "name": search_term, "exchange": "NSE"}

def get_current_market_data(symbol: str) -> dict:
    ticker = yf.Ticker(symbol)
    info = ticker.info
    return {
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "currency": info.get("currency", "INR"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "price_to_book": info.get("priceToBook"),
        "52_week_high": info.get("fiftyTwoWeekHigh"),
        "52_week_low": info.get("fiftyTwoWeekLow"),
        "debt_to_equity": info.get("debtToEquity")
    }

def get_historical_and_metrics(symbol: str, period: str = "1y") -> dict:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    if df.empty:
        return {"error": "Failed to fetch historical data"}
    
    df['Daily_Return'] = df['Close'].pct_change()
    total_return = float((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0])
    annualized_volatility = float(df['Daily_Return'].std() * np.sqrt(252))
    cagr = float(((df['Close'].iloc[-1] / df['Close'].iloc[0]) ** (1 / 1.0)) - 1)
    max_drawdown = float(((df['Close'] - df['Close'].cummax()) / df['Close'].cummax()).min())
    
    return {
        "period": period,
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "annualized_volatility_pct": round(annualized_volatility * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "start_price": round(float(df['Close'].iloc[0]), 2),
        "end_price": round(float(df['Close'].iloc[-1]), 2)
    }

def retrieve_recent_news(company_name: str, max_results: int = 5) -> list:
    results = []
    with DDGS() as ddgs:
        news_gen = ddgs.news(f"{company_name} stock business financial news", max_results=max_results)
        for item in news_gen:
            results.append({
                "title": item.get("title"),
                "source": item.get("source"),
                "date": item.get("date"),
                "snippet": item.get("body")
            })
    return results

def get_company_financials(symbol: str) -> dict:
    ticker = yf.Ticker(symbol)
    try:
        financials = ticker.financials
        revenue = float(financials.loc['Total Revenue'].iloc[0]) if 'Total Revenue' in financials.index else None
        net_income = float(financials.loc['Net Income'].iloc[0]) if 'Net Income' in financials.index else None
        return {"latest_revenue": revenue, "latest_net_income": net_income}
    except Exception as e:
        return {"error": f"Could not retrieve financials: {str(e)}"}

AVAILABLE_TOOLS = {
    "identify_company": identify_company,
    "get_current_market_data": get_current_market_data,
    "get_historical_and_metrics": get_historical_and_metrics,
    "retrieve_recent_news": retrieve_recent_news,
    "get_company_financials": get_company_financials
}

gemini_tools = [
    identify_company,
    get_current_market_data,
    get_historical_and_metrics,
    retrieve_recent_news,
    get_company_financials
]

# ==========================================
# RETRY & EXECUTION ENGINE
# ==========================================
def send_message_with_retry(chat_session, content, max_retries=4, initial_delay=3):
    for attempt in range(max_retries):
        try:
            return chat_session.send_message(content)
        except ServerError as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(initial_delay * (2 ** attempt))

def generate_pdf_report(markdown_text: str) -> str:
    pdf = MarkdownPdf(toc_level=2)
    pdf.add_section(Section(markdown_text))
    output_filename = "Institutional_Equity_Research_Report.pdf"
    pdf.save(output_filename)
    return output_filename

# ==========================================
# MAIN APP DASHBOARD
# ==========================================
st.title("🏛️ Institutional Equity Research AI Agent")
st.markdown("Automated financial data aggregation, risk modeling, and institutional report generation.")

target_company = st.text_input("Enter Company Name or Ticker:", "Reliance Industries", help="Supports NSE/BSE & Global exchanges")

if st.button("🚀 Run Institutional Analysis", type="primary"):
    if not api_key:
        st.error("API Key is required to run analysis.")
        st.stop()

    client = genai.Client(api_key=api_key)
    
    with st.status("Agent Executing Multi-Tool Workflow...", expanded=True) as status:
        st.write("🔍 Identifying market entity and ticker exchange...")
        entity = identify_company(target_company)
        symbol = entity.get("symbol")
        
        st.write(f"📊 Fetching real-time market data for **{symbol}**...")
        market_data = get_current_market_data(symbol)
        
        st.write("📈 Computing historical volatility, CAGR, and Max Drawdown...")
        quant_metrics = get_historical_and_metrics(symbol)
        
        st.write("📰 Scraping recent business news and catalyst events...")
        news_data = retrieve_recent_news(entity.get("name"))

        st.write("🎨 Rendering Price Action Visualization...")
        ticker_obj = yf.Ticker(symbol)
        hist_df = ticker_obj.history(period="1y")

        status.update(label="Data Aggregation Complete. Generating Report...", state="running")

        system_instruction = (
            "You are an Institutional Equity Analyst. Produce a professional investment research report "
            "based on structured tools. Use Markdown, clear bullet points, risk matrices, and financial tables."
        )

        chat = client.chats.create(
            model=MODEL_ID,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=gemini_tools,
                temperature=0.2
            )
        )

        prompt = f"Analyze {target_company} completely using all available tools."
        response = send_message_with_retry(chat, prompt)

        while True:
            if response.function_calls:
                tool_response_parts = []
                for call in response.function_calls:
                    func_name = call.name
                    func_args = call.args
                    st.write(f"⚙️ Running tool: `{func_name}`")
                    tool_func = AVAILABLE_TOOLS[func_name]
                    result = tool_func(**func_args)
                    tool_response_parts.append(
                        types.Part.from_function_response(name=func_name, response={"result": result})
                    )
                response = send_message_with_retry(chat, tool_response_parts)
            else:
                break

        status.update(label="Analysis Complete!", state="complete", expanded=False)

    st.divider()
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">CURRENT PRICE</div>
            <div class="metric-value">{market_data.get('currency', 'INR')} {market_data.get('current_price', 'N/A')}</div>
            <div class="metric-sub">P/E: {market_data.get('pe_ratio', 'N/A')}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m2:
        tot_ret = quant_metrics.get('total_return_pct', 0)
        color = "positive" if tot_ret >= 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">1Y RETURN</div>
            <div class="metric-value {color}">{tot_ret}%</div>
            <div class="metric-sub">CAGR: {quant_metrics.get('cagr_pct', 'N/A')}%</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">ANNUALIZED VOLATILITY</div>
            <div class="metric-value">{quant_metrics.get('annualized_volatility_pct', 'N/A')}%</div>
            <div class="metric-sub">Risk Factor</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">MAX DRAWDOWN</div>
            <div class="metric-value negative">{quant_metrics.get('max_drawdown_pct', 'N/A')}%</div>
            <div class="metric-sub">Peak-to-Trough</div>
        </div>
        """, unsafe_allow_html=True)

    if not hist_df.empty:
        st.subheader("📊 1-Year Price & Volume Profile")
        fig = go.Figure(data=[go.Candlestick(
            x=hist_df.index,
            open=hist_df['Open'],
            high=hist_df['High'],
            low=hist_df['Low'],
            close=hist_df['Close'],
            name=symbol
        )])
        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("📑 Final Equity Research Report")
    report_text = response.text
    st.markdown(report_text)

    pdf_file = generate_pdf_report(report_text)
    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📥 Download Professional PDF Pitchbook",
            data=f,
            file_name=f"{symbol}_Research_Report.pdf",
            mime="application/pdf",
            type="primary"
        )
