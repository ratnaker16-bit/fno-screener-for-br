import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz

st.set_page_config(page_title="F&O RSI Screener", layout="wide")

# ------------------------------------------------------------------
# F&O STOCKS LIST (NSE) - ज़रूरत हो तो यहाँ add/remove करें
# ------------------------------------------------------------------
FNO_STOCKS = [
    "AARTIIND", "ABB", "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ACC", "ADANIENT", "ADANIPORTS",
    "ALKEM", "AMBUJACEM", "ANGELONE", "APOLLOHOSP", "APOLLOTYRE", "ASHOKLEY", "ASIANPAINT",
    "ASTRAL", "ATGL", "AUBANK", "AUROPHARMA", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV",
    "BAJFINANCE", "BALKRISIND", "BANDHANBNK", "BANKBARODA", "BANKINDIA", "BDL", "BEL",
    "BHARATFORG", "BHARTIARTL", "BHEL", "BIOCON", "BOSCHLTD", "BPCL", "BRITANNIA", "BSE",
    "BSOFT", "CAMS", "CANBK", "CDSL", "CESC", "CGPOWER", "CHAMBLFERT", "CHOLAFIN", "CIPLA",
    "COALINDIA", "COFORGE", "COLPAL", "CONCOR", "CROMPTON", "CUMMINSIND", "CYIENT", "DABUR",
    "DALBHARAT", "DEEPAKNTR", "DELHIVERY", "DIVISLAB", "DIXON", "DLF", "DMART", "DRREDDY",
    "EICHERMOT", "ESCORTS", "EXIDEIND", "FEDERALBNK", "GAIL", "GLENMARK", "GMRAIRPORT",
    "GODREJCP", "GODREJPROP", "GRANULES", "GRASIM", "HAL", "HAVELLS", "HCLTECH", "HDFCAMC",
    "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HFCL", "HINDALCO", "HINDCOPPER", "HINDPETRO",
    "HINDUNILVR", "HUDCO", "ICICIBANK", "ICICIGI", "ICICIPRULI", "IDEA", "IDFCFIRSTB", "IEX",
    "IGL", "IIFL", "INDHOTEL", "INDIANB", "INDIGO", "INDUSINDBK", "INDUSTOWER", "INFY",
    "INOXWIND", "IOC", "IRB", "IRCTC", "IREDA", "IRFC", "ITC", "JINDALSTEL", "JIOFIN",
    "JSWENERGY", "JSWSTEEL", "JUBLFOOD", "KALYANKJIL", "KAYNES", "KEI", "KFINTECH",
    "KOTAKBANK", "KPITTECH", "LAURUSLABS", "LICHSGFIN", "LICI", "LODHA", "LT", "LTF",
    "LTIM", "LUPIN", "M&M", "M&MFIN", "MANAPPURAM", "MANKIND", "MARICO", "MARUTI",
    "MAXHEALTH", "MAZDOCK", "MCX", "MFSL", "MGL", "MOTHERSON", "MPHASIS", "MUTHOOTFIN",
    "NATIONALUM", "NAUKRI", "NBCC", "NCC", "NESTLEIND", "NHPC", "NMDC", "NTPC", "NYKAA",
    "OBEROIRLTY", "OFSS", "OIL", "ONGC", "PAGEIND", "PATANJALI", "PAYTM", "PERSISTENT",
    "PETRONET", "PFC", "PGEL", "PHOENIXLTD", "PIDILITIND", "PIIND", "PNB", "PNBHOUSING",
    "POLICYBZR", "POLYCAB", "POONAWALLA", "POWERGRID", "PPLPHARMA", "PRESTIGE", "RBLBANK",
    "RECLTD", "RELIANCE", "RVNL", "SAIL", "SBICARD", "SBILIFE", "SBIN", "SHREECEM",
    "SHRIRAMFIN", "SIEMENS", "SOLARINDS", "SONACOMS", "SRF", "SUNPHARMA", "SUPREMEIND",
    "SUZLON", "SYNGENE", "TATACHEM", "TATACONSUM", "TATAELXSI", "TATAMOTORS", "TATAPOWER",
    "TATASTEEL", "TATATECH", "TCS", "TECHM", "TIINDIA", "TITAGARH", "TITAN", "TORNTPHARM",
    "TORNTPOWER", "TRENT", "TVSMOTOR", "ULTRACEMCO", "UNIONBANK", "UNITDSPR", "UNOMINDA",
    "UPL", "VBL", "VEDL", "VOLTAS", "WIPRO", "YESBANK", "ZYDUSLIFE",
]

IST = pytz.timezone("Asia/Kolkata")


# ------------------------------------------------------------------
# RSI (Wilder, 14)
# ------------------------------------------------------------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-12)
    return 100 - (100 / (1 + rs))


# ------------------------------------------------------------------
# VWAP (रोज़ का, हर दिन 9:15 से reset)
# ------------------------------------------------------------------
def vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].fillna(0)
    idx = df.index.tz_convert(IST) if df.index.tz is not None else df.index
    day = pd.Series(idx.date, index=df.index)
    pv = (typical * vol).groupby(day).cumsum()
    cv = vol.groupby(day).cumsum()
    return pv / cv.replace(0, float("nan"))


# ------------------------------------------------------------------
# Data download (batch)
# ------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def download(symbols: tuple, interval: str):
    tickers = [s + ".NS" for s in symbols]
    data = yf.download(
        tickers,
        period="5d",
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    return data


def get_ohlcv(data, symbol):
    t = symbol + ".NS"
    try:
        if t not in data.columns.get_level_values(0):
            return None
        df = data[t][["High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    except Exception:
        return None
    return df if len(df) > 20 else None


def fmt_time(ts):
    try:
        return ts.tz_convert(IST).strftime("%H:%M")
    except Exception:
        return str(ts)


# ------------------------------------------------------------------
# SCAN
# STEP 1: VWAP से side तय  -> Price > VWAP = सिर्फ CE | Price < VWAP = सिर्फ PE
# STEP 2: उसी side की RSI cross condition देखो
# ------------------------------------------------------------------
def scan(symbols, interval, lookback, drop_running, use_vwap):
    data = download(tuple(symbols), interval)
    results = {"CE_40": [], "CE_60": [], "PE_60": [], "PE_40": []}

    for sym in symbols:
        df = get_ohlcv(data, sym)
        if df is None:
            continue
        if drop_running:
            df = df.iloc[:-1]  # चल रही (अधूरी) candle हटाओ
        if len(df) < 20:
            continue

        c = df["Close"]
        r = rsi(c).dropna()
        vw = vwap(df)
        if len(r) < lookback + 1:
            continue

        last_price = float(c.iloc[-1])
        last_vwap = vw.iloc[-1]
        if pd.isna(last_vwap):
            continue
        last_vwap = float(last_vwap)
        last_rsi = float(r.iloc[-1])
        candle_time = fmt_time(c.index[-1])

        # ---------- STEP 1: VWAP से side ----------
        if use_vwap:
            if last_price > last_vwap:
                side = "CE"
            elif last_price < last_vwap:
                side = "PE"
            else:
                continue
        else:
            side = "BOTH"

        # ---------- STEP 2: उसी side की RSI condition ----------
        for i in range(1, lookback + 1):
            ts = r.index[-i]
            prev, cur = float(r.iloc[-i - 1]), float(r.iloc[-i])

            def row():
                return {
                    "Stock": sym,
                    "Price": round(last_price, 2),
                    "VWAP": round(last_vwap, 2),
                    "RSI (now)": round(last_rsi, 1),
                    "Prev RSI": round(prev, 1),
                    "Cross RSI": round(cur, 1),
                    "Cross candle": fmt_time(ts),
                    "Last candle": candle_time,
                }

            if side in ("CE", "BOTH"):
                if prev < 40 <= cur:
                    results["CE_40"].append(row())
                if prev < 60 <= cur:
                    results["CE_60"].append(row())

            if side in ("PE", "BOTH"):
                if prev > 60 >= cur:
                    results["PE_60"].append(row())
                if prev > 40 >= cur:
                    results["PE_40"].append(row())

    out = {}
    for k, v in results.items():
        d = pd.DataFrame(v)
        if not d.empty:
            d = d.sort_values("Cross candle", ascending=False).drop_duplicates("Stock")
        out[k] = d
    return out


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("📈 F&O VWAP + RSI(14) Screener")
now = datetime.now(IST)
st.caption(f"अभी: {now.strftime('%d-%b-%Y %H:%M:%S')} IST  |  Data: Yahoo Finance (≈1 मिनट cache, कुछ delay संभव)")

with st.sidebar:
    st.header("Settings")
    lookback = st.slider("पिछली कितनी candles में cross देखें", 1, 5, 1)
    drop_running = st.checkbox("चल रही candle ignore करें (सिर्फ closed candle)", value=True)
    use_vwap = st.checkbox("VWAP filter (ऊपर = CE, नीचे = PE)", value=True)
    st.write(f"कुल F&O stocks: **{len(FNO_STOCKS)}**")
    if st.button("🔄 Refresh / Scan Now"):
        st.cache_data.clear()
        st.rerun()

TF = [("2m", "2 Minute"), ("5m", "5 Minute")]
tabs = st.tabs([f"⏱ {name}" for _, name in TF])

for tab, (interval, name) in zip(tabs, TF):
    with tab:
        with st.spinner(f"{name} chart scan हो रहा है..."):
            res = scan(FNO_STOCKS, interval, lookback, drop_running, use_vwap)

        st.subheader(f"🟢 CE Entry — {name} (Price > VWAP)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**RSI 40 को नीचे से cross** ({len(res['CE_40'])})")
            st.dataframe(res["CE_40"], use_container_width=True, hide_index=True)
        with c2:
            st.markdown(f"**RSI 60 को नीचे से cross** ({len(res['CE_60'])})")
            st.dataframe(res["CE_60"], use_container_width=True, hide_index=True)

        st.subheader(f"🔴 PE Entry — {name} (Price < VWAP)")
        p1, p2 = st.columns(2)
        with p1:
            st.markdown(f"**RSI 60 को ऊपर से cross** ({len(res['PE_60'])})")
            st.dataframe(res["PE_60"], use_container_width=True, hide_index=True)
        with p2:
            st.markdown(f"**RSI 40 को ऊपर से cross** ({len(res['PE_40'])})")
            st.dataframe(res["PE_40"], use_container_width=True, hide_index=True)

st.caption("⚠️ सिर्फ screening tool है, trading advice नहीं। Entry से पहले chart खुद confirm करें।")
