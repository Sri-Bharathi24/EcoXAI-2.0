"""
EcoXAI 2.0 - Explainable IoT Water Intelligence for AI Data Centres
Developed by SRI BHARATHI VN, III B.Sc. Computer Science with Cognitive Systems

Every number on screen comes from a Random Forest trained on your results CSV
(or a clearly labelled simulated dataset if no CSV is found). SHAP explains each prediction.
"""

import glob
import io
import os

import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict, train_test_split

# ------------------------------------------------------------------
# SETTINGS you may want to edit
# ------------------------------------------------------------------
WATER_UNIT = "L"
TEMP_ALERT = 35
REFRESH_SECONDS = 3
HISTORY_LENGTH = 40
APP_DIR = os.path.dirname(os.path.abspath(__file__))

SERVERS, TEMP, COOLING, MINING = "Active Servers", "Temperature", "Cooling Hours", "Crypto Mining"
FEATURES = [SERVERS, TEMP, COOLING, MINING]
TARGET = "Water"
COLORS = {SERVERS: "#65BFFF", TEMP: "#27D8E8", COOLING: "#35E0C1", MINING: "#8E9BFF"}

ACTIONS = {
    TEMP: "Check cooling system efficiency",
    SERVERS: "Review idle or lightly used servers",
    COOLING: "Review the cooling schedule",
    MINING: "Review the crypto-mining workload",
}
FIRST_STEP = {
    TEMP: "Check cooling set-points and airflow first.",
    SERVERS: "Review server load. Can idle servers be consolidated or switched off?",
    COOLING: "Review the cooling schedule. Long cooling hours add water use.",
    MINING: "Review the crypto-mining workload. Reduce it or move it to cooler hours.",
}

# Your results data, built into the code so no separate CSV file is needed.
# If you later add a bigger CSV file next to app.py, the app uses that file instead.
EMBEDDED_CSV = """Number of Servers,Temperature (C),Cooling Hours,Crypto Mining,Water Used (L),Crypto Status,Risk Level
100,25,2,0,500,No Mining,Low
200,35,5,1,1200,Mining Active,High
150,30,3,0,800,No Mining,Medium
300,40,7,1,2000,Mining Active,Critical
250,38,6,1,1700,Mining Active,High
180,28,3,0,650,No Mining,Low
220,33,4,1,1100,Mining Active,High
270,39,6,1,1850,Mining Active,Critical
130,26,2,0,520,No Mining,Low
190,31,4,0,900,No Mining,Medium
"""

st.set_page_config(page_title="EcoXAI 2.0", page_icon="💧", layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------
# STYLE: dark ocean dashboard
# ------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Caveat:wght@600&display=swap');
html, body, [class*="css"], .stApp { font-family:'Manrope',sans-serif; }
.stApp { background:linear-gradient(180deg,#04101F 0%,#061A33 100%); }
[data-testid="stHeader"] { background:rgba(4,16,31,.85); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#071A30,#04101F); border-right:1px solid #123B5E; }
[data-testid="stSidebar"] [role="radiogroup"] { gap:6px; }
[data-testid="stSidebar"] [role="radiogroup"] label { background:transparent; border-radius:10px; padding:9px 12px; width:100%; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background:#0F4468; }
.brand { text-align:center; padding:6px 0 14px 0; }
.brand h2 { margin:6px 0 0 0; color:#F3F8FF; font-size:1.75rem; font-weight:800; }
.brand h2 span { color:#27D8E8; }
.brand p { color:#8FD3F5; font-style:italic; margin:4px 0 0 0; font-size:.9rem; line-height:1.35; }
.sidefoot { margin-top:34px; padding:14px 6px; border-top:1px solid #123B5E; color:#A8BED2; font-size:.85rem; }
.sidefoot b { color:#F3F8FF; display:block; font-size:1rem; }
.hero { position:relative; overflow:hidden; border:1px solid #1A6FA8; border-radius:18px; padding:28px 34px; margin-bottom:18px;
        background:linear-gradient(100deg,#05203D 0%,#0A4A82 55%,#1B86DA 100%); display:flex; align-items:center; gap:26px; }
.hero h1 { margin:0; color:#F3F8FF; font-size:3rem; font-weight:800; line-height:1.05; }
.hero h1 span { color:#27D8E8; }
.hero .l2 { color:#F3F8FF; font-size:1.55rem; font-weight:700; margin:6px 0 4px 0; }
.hero .l3 { color:#CFE9FA; font-size:1.02rem; }
.hero .script { position:absolute; right:44px; top:38%; transform:rotate(-6deg); font-family:'Caveat',cursive; font-size:1.9rem; color:#DDF3FF; text-align:right; line-height:1.15; }
.hero svg.wave { position:absolute; left:0; right:0; bottom:0; width:100%; height:62px; }
.sec { color:#F3F8FF; font-size:1.35rem; font-weight:800; margin:4px 0 6px 0; }
.pill { display:inline-block; margin-left:10px; font-size:.8rem; color:#A8BED2; font-weight:600; }
.pill i { display:inline-block; width:9px; height:9px; border-radius:50%; background:#35E0C1; margin-right:6px; }
.kc { background:linear-gradient(145deg,#0B2A4A,#08203A); border:1px solid #1A5A86; border-radius:16px; padding:16px 18px; height:100%; }
.kc .lbl { color:#BBD5E9; font-size:.88rem; font-weight:600; }
.kc .val { color:#F3F8FF; font-size:1.75rem; font-weight:800; line-height:1.3; margin-top:4px; }
.kc.main .val { color:#FFD166; }
.kc .sub { color:#8FA9BF; font-size:.78rem; margin-top:2px; }
.pn { background:linear-gradient(145deg,#0A2644,#071C33); border:1px solid #1A5A86; border-radius:16px; padding:16px 18px; margin-bottom:12px; }
.pn h4 { color:#F3F8FF; margin:0 0 10px 0; font-size:1.05rem; font-weight:800; }
.ins { display:flex; gap:10px; align-items:flex-start; padding:9px 0; border-bottom:1px solid #123B5E; }
.ins:last-child { border-bottom:none; }
.ins .dt { width:14px; height:14px; border-radius:50%; margin-top:4px; flex:none; }
.ins b { color:#F3F8FF; display:block; font-size:.92rem; } .ins span { color:#A8BED2; font-size:.82rem; }
.act { border:1px solid #1A5A86; background:#0A2A48; border-radius:10px; padding:10px 12px; margin-bottom:8px; color:#E3F1FB; font-size:.9rem; }
.donutwrap { display:flex; align-items:center; gap:22px; }
.donut { width:150px; height:150px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex:none; }
.hole { width:96px; height:96px; border-radius:50%; background:#08203A; text-align:center; display:flex; flex-direction:column; justify-content:center; color:#CFE9FA; font-size:.78rem; }
.hole b { color:#F3F8FF; font-size:1.2rem; }
.lg { display:flex; align-items:center; gap:8px; color:#D8E9F6; font-size:.88rem; margin-bottom:6px; }
.lg b { margin-left:auto; color:#F3F8FF; padding-left:16px; }
.dot { width:11px; height:11px; border-radius:50%; display:inline-block; }
.strip { display:flex; gap:14px; align-items:stretch; background:linear-gradient(145deg,#0A2644,#071C33); border:1px solid #1A5A86; border-radius:16px; padding:16px 18px; margin-top:6px; }
.strip .txt { flex:1.4; } .strip .txt h4 { color:#F3F8FF; margin:0 0 4px 0; font-size:1.25rem; font-weight:800; }
.strip .txt p { color:#A8BED2; font-size:.85rem; margin:0 0 6px 0; }
.strip .txt .go { color:#F3F8FF; font-weight:700; font-size:.92rem; }
.tile { flex:1; border-radius:12px; padding:10px 12px; border:1px solid; }
.tile .t1 { font-weight:700; font-size:.88rem; } .tile .t2 { font-size:1.15rem; font-weight:800; color:#F3F8FF; } .tile .t3 { font-size:.74rem; color:#CFE0EE; }
.tile.red { border-color:#B3414C; background:#2D1A24; } .tile.red .t1 { color:#FF8A93; }
.tile.amber { border-color:#9A6A20; background:#2A2418; } .tile.amber .t1 { color:#FFC26B; }
.tile.blue { border-color:#1A6FA8; background:#0A2A48; } .tile.blue .t1 { color:#65BFFF; }
.tile.green { border-color:#1F7A63; background:#0B2E31; } .tile.green .t1 { color:#55D6A4; }
.badge { display:inline-block; padding:2px 11px; border-radius:20px; font-weight:800; font-size:.8rem; color:#04101F; margin-right:8px; }
.badge.normal { background:#55D6A4; } .badge.warning { background:#FF9F43; } .badge.critical { background:#FF5D68; }
.footer { display:flex; justify-content:space-between; align-items:center; border-top:1px solid #123B5E; margin-top:26px; padding:16px 4px 4px 4px; color:#8FD3F5; font-size:.8rem; }
.footer .sc { font-family:'Caveat',cursive; font-size:1.4rem; color:#9FDDF7; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-color:#1A5A86 !important; border-radius:16px !important; background:linear-gradient(145deg,#0A2644,#071C33); }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# DATA: your CSV if found, otherwise a clearly labelled simulated set
# ------------------------------------------------------------------
def _to_binary(series):
    if pd.api.types.is_numeric_dtype(series):
        return (series > 0).astype(int)
    yes = ["yes", "y", "true", "1", "active", "on", "mining"]
    return series.astype(str).str.strip().str.lower().isin(yes).astype(int)


def _find(df, hints, used, numeric_only=True):
    for col in df.columns:
        if col in used:
            continue
        if numeric_only and not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if any(h in col.lower() for h in hints):
            return col
    return None


def _from_csv(path):
    try:
        raw = pd.read_csv(path)
    except Exception:
        return None
    used, found = set(), {}
    for key, hints, numeric in [
        (TARGET, ["water"], True),
        (SERVERS, ["server"], True),
        (TEMP, ["temp"], True),
        (COOLING, ["cool"], True),
        (MINING, ["mining", "crypto"], False),
    ]:
        col = _find(raw, hints, used, numeric)
        if col is None:
            return None
        used.add(col)
        found[key] = col
    out = pd.DataFrame({k: raw[c] for k, c in found.items()})
    out[MINING] = _to_binary(out[MINING])
    out = out.dropna()
    return out[FEATURES + [TARGET]] if len(out) >= 8 else None


def _simulated(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    servers = rng.integers(50, 500, n)
    temp = rng.uniform(18, 45, n)
    cooling = rng.uniform(1, 24, n)
    mining = rng.integers(0, 2, n)
    water = (150 + 1.5 * servers + 8 * cooling + 3 * np.maximum(temp - 28, 0) ** 2) * (1 + 0.9 * mining)
    water = water + rng.normal(0, 40, n)
    return pd.DataFrame({SERVERS: servers, TEMP: temp, COOLING: cooling, MINING: mining, TARGET: water})


def load_data():
    csvs = glob.glob(os.path.join(APP_DIR, "*.csv"))
    csvs.sort(key=lambda p: ("ecoxai" not in p.lower() and "result" not in p.lower(), p))
    for path in csvs:
        data = _from_csv(path)
        if data is not None:
            return data, f"{os.path.basename(path)} ({len(data):,} rows)", False
    embedded = _from_csv(io.StringIO(EMBEDDED_CSV))
    if embedded is not None:
        return embedded, f"results data built into the code ({len(embedded)} rows)", False
    return _simulated(), "built-in simulated dataset", True


@st.cache_resource(show_spinner="Training the Random Forest model...")
def build_model():
    data, source, is_fallback = load_data()
    if len(data) < 50:
        # Very small dataset: score with leave-one-out so each row is predicted by a model that never saw it.
        model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        predicted = cross_val_predict(model, data[FEATURES], data[TARGET], cv=LeaveOneOut())
        actual = data[TARGET]
        metrics = {"r2": r2_score(actual, predicted), "mae": mean_absolute_error(actual, predicted),
                   "n_rows": len(data), "method": "leave-one-out", "small": True}
        model.fit(data[FEATURES], data[TARGET])
    else:
        x_train, x_test, y_train, y_test = train_test_split(data[FEATURES], data[TARGET], test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=150, min_samples_leaf=2, random_state=42, n_jobs=-1)
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        metrics = {"r2": r2_score(y_test, predicted), "mae": mean_absolute_error(y_test, predicted),
                   "n_rows": len(data), "method": "80/20 hold-out", "small": False}
    return model, shap.TreeExplainer(model), data, source, is_fallback, metrics


model, explainer, data, source, is_fallback, metrics = build_model()
LOW, HIGH = data[FEATURES].min(), data[FEATURES].max()
MEDIAN, P75 = data[FEATURES].median(), data[FEATURES].quantile(0.75)
AVG_WATER = float(data[TARGET].mean())


@st.cache_resource
def global_shap():
    values = explainer.shap_values(data[FEATURES])
    return pd.Series(np.abs(np.array(values)).mean(axis=0), index=FEATURES)


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def predict(row):
    return float(model.predict(pd.DataFrame([row], columns=FEATURES))[0])


def shap_values(row):
    values = explainer.shap_values(pd.DataFrame([row], columns=FEATURES))
    return np.array(values).reshape(-1)[: len(FEATURES)]


def show(feature, value):
    if feature == MINING:
        return "On" if value else "Off"
    if feature == TEMP:
        return f"{value:.1f} °C"
    if feature == SERVERS:
        return f"{value:.0f}"
    return f"{value:.1f}"


def status_of(row):
    hot, mining = row[TEMP] > TEMP_ALERT, bool(row[MINING])
    if hot and mining:
        return "critical", "Critical"
    if hot or mining:
        return "warning", "Warning"
    return "normal", "Normal"


def kcard(label, value, sub="", main=False):
    return f'<div class="kc{" main" if main else ""}"><div class="lbl">{label}</div><div class="val">{value}</div><div class="sub">{sub}</div></div>'


def donut_html(shares):
    stops, acc = [], 0.0
    for _, share, color in shares:
        stops.append(f"{color} {acc * 100:.1f}% {(acc + share) * 100:.1f}%")
        acc += share
    top = max(shares, key=lambda s: s[1])
    legend = "".join(f'<div class="lg"><span class="dot" style="background:{c}"></span>{n}<b>{s * 100:.0f}%</b></div>' for n, s, c in shares)
    return (f'<div class="donutwrap"><div class="donut" style="background:conic-gradient({", ".join(stops)})">'
            f'<div class="hole"><div>{top[0]}</div><b>{top[1] * 100:.0f}%</b></div></div><div>{legend}</div></div>')


def insights_html(row):
    items = []
    if row[TEMP] > TEMP_ALERT:
        items.append(("#FF5D68", "Temperature is high", f"Above {TEMP_ALERT} °C may increase cooling water use."))
    elif row[TEMP] > TEMP_ALERT - 3:
        items.append(("#FFB454", "Temperature is slightly high", "May increase cooling water usage."))
    else:
        items.append(("#55D6A4", "Temperature is within range", "No temperature alert."))
    if row[MINING]:
        items.append(("#FFB454", "Crypto mining is active", "Mining raises the predicted water use."))
    else:
        items.append(("#55D6A4", "Mining load is normal", "No immediate action needed."))
    if row[SERVERS] > P75[SERVERS]:
        items.append(("#FFB454", "Server count is above typical", "More servers can increase cooling demand."))
    else:
        items.append(("#65BFFF", "Server count is within range", "Keep monitoring."))
    rows = "".join(f'<div class="ins"><span class="dt" style="background:{c}"></span><div><b>{t}</b><span>{d}</span></div></div>' for c, t, d in items)
    return f'<div class="pn"><h4>Quick insights</h4>{rows}</div>'


def actions_html(sv):
    rising = [FEATURES[i] for i in np.argsort(-sv) if sv[i] > 0]
    steps = [ACTIONS[f] for f in rising[:3]]
    for filler in ["Keep monitoring the live readings", "Compare with the previous reading", "Review the what-if tab"]:
        if len(steps) < 3:
            steps.append(filler)
    return '<div class="pn"><h4>Suggested actions</h4>' + "".join(f'<div class="act">{s}</div>' for s in steps) + "</div>"


def strip_html(row, sv):
    level, label = status_of(row)
    rising = [i for i in np.argsort(-sv) if sv[i] > 0]
    go = FIRST_STEP[FEATURES[rising[0]]] if rising else "No input is pushing water use up right now. Keep monitoring."

    def tone(feature):
        v = row[feature]
        if feature == TEMP:
            return "red" if v > TEMP_ALERT else ("amber" if v > TEMP_ALERT - 3 else "green")
        if feature == MINING:
            return "red" if v else "green"
        return "amber" if v > P75[feature] else "blue"

    tiles = ""
    for i, f in enumerate(FEATURES):
        note = f"{'Raising' if sv[i] > 0 else 'Lowering'} water use by about {abs(sv[i]):,.0f} {WATER_UNIT}"
        tiles += f'<div class="tile {tone(f)}"><div class="t1">{f}</div><div class="t2">{show(f, row[f])}</div><div class="t3">{note}</div></div>'
    return (f'<div class="strip"><div class="txt"><h4>Fix first</h4><p><span class="badge {level}">{label}</span>'
            f'Model estimate, not a measured leak.</p><div class="go">{go}</div></div>{tiles}</div>')


def next_reading(prev, rng):
    row = dict(prev)
    row[SERVERS] = float(np.clip(prev[SERVERS] + rng.integers(-10, 11), LOW[SERVERS], HIGH[SERVERS]))
    row[TEMP] = float(np.clip(prev[TEMP] + rng.uniform(-1.2, 1.5), LOW[TEMP], HIGH[TEMP]))
    row[COOLING] = float(np.clip(prev[COOLING] + rng.uniform(-0.8, 0.8), LOW[COOLING], HIGH[COOLING]))
    if rng.random() < 0.12:
        row[MINING] = 1 - int(prev[MINING])
    return row


# ------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------
if "history" not in st.session_state or any(f not in st.session_state.history[-1] for f in FEATURES):
    # (also resets stale readings left over from an older version of the app)
    start = {f: float(MEDIAN[f]) for f in FEATURES}
    start[MINING] = 0
    start["Predicted water"] = predict(start)
    start["Reading"] = 1
    st.session_state.history = [start]
if "running" not in st.session_state:
    st.session_state.running = False

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
DROP = ('<svg width="64" height="78" viewBox="0 0 64 80"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#7BE3FF"/><stop offset="1" stop-color="#1E7BE0"/></linearGradient></defs>'
        '<path d="M32 3C32 3 8 30 8 50a24 24 0 0 0 48 0C56 30 32 3 32 3Z" fill="url(#g)"/>'
        '<path d="M20 52a12 12 0 0 0 12 12" stroke="#fff" stroke-opacity=".75" stroke-width="3" fill="none" stroke-linecap="round"/></svg>')

PAGES = ["🏠 Home", "📊 Data Explorer", "🧠 XAI Explanations", "🔮 What-If", "✅ Validation", "🧩 Architecture"]
with st.sidebar:
    st.markdown('<div class="brand"><h2>EcoXAI <span>2.0</span></h2><p>Saving Water.<br>Explaining Why.<br>Intelligently.</p></div>', unsafe_allow_html=True)
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown('<div class="sidefoot">Developed by<b>SRI BHARATHI VN</b>III B.Sc. Computer Science<br>with Cognitive Systems</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# HERO
# ------------------------------------------------------------------
st.markdown(
    '<div class="hero"><div><h1>EcoXAI <span>2.0</span></h1>'
    '<div class="l2">Saving Water. Explaining Why. Intelligently.</div>'
    '<div class="l3">Explainable IoT Water Intelligence for AI Data Centres</div></div></div>',
    unsafe_allow_html=True,
)

if is_fallback:
    st.warning("Your results CSV was not found next to app.py, so this app is using a built-in SIMULATED dataset.")
else:
    st.success(f"Model trained on {source}. Sensor readings in the live stream are simulated.")


# ------------------------------------------------------------------
# PAGE: HOME
# ------------------------------------------------------------------
if page == PAGES[0]:
    head, tog, rst = st.columns([3, 2.2, 1])
    head.markdown('<div class="sec">Live resource indicators<span class="pill"><i></i>Simulated data</span></div>', unsafe_allow_html=True)
    tog.toggle("Start live monitoring", key="running")
    if rst.button("Reset stream"):
        st.session_state.history = st.session_state.history[:1]

    run_every = REFRESH_SECONDS if st.session_state.running else None

    @st.fragment(run_every=run_every)
    def home_panel():
        history = st.session_state.history
        if st.session_state.running:
            row = next_reading({f: history[-1][f] for f in FEATURES}, np.random.default_rng())
            row["Predicted water"] = predict(row)
            row["Reading"] = history[-1]["Reading"] + 1
            history.append(row)
            del history[:-HISTORY_LENGTH]
        now = history[-1]
        row_now = {f: now[f] for f in FEATURES}
        sv = shap_values(row_now)
        level, label = status_of(row_now)
        top = int(np.argmax(np.abs(sv)))
        vs_avg = (now["Predicted water"] - AVG_WATER) / AVG_WATER * 100 if AVG_WATER else 0.0

        main_col, side_col = st.columns([2.5, 1.1])
        with main_col:
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(kcard("💧 Predicted water use", f"{now['Predicted water']:,.0f} {WATER_UNIT}", "Random Forest estimate", main=True), unsafe_allow_html=True)
            c2.markdown(kcard("🚦 Status", label, f"Warning if temperature is above {TEMP_ALERT} °C or mining is on"), unsafe_allow_html=True)
            c3.markdown(kcard("🧠 Top driver", FEATURES[top], f"About {abs(sv[top]):,.0f} {WATER_UNIT} {'up' if sv[top] > 0 else 'down'} (SHAP)"), unsafe_allow_html=True)
            c4.markdown(kcard("📈 Vs dataset average", f"{vs_avg:+.0f}%", f"Average water use: {AVG_WATER:,.0f} {WATER_UNIT}"), unsafe_allow_html=True)

            t_col, d_col = st.columns([1.35, 1])
            with t_col:
                with st.container(border=True):
                    st.markdown("**Predicted water use over time**")
                    st.line_chart(pd.DataFrame(history).set_index("Reading")[["Predicted water"]], height=235, color="#27D8E8")
            with d_col:
                total = float(np.abs(sv).sum())
                shares = [(f, (abs(sv[i]) / total) if total else 0.25, COLORS[f]) for i, f in enumerate(FEATURES)]
                st.markdown('<div class="pn"><h4>What is driving water use</h4>' + donut_html(shares) + '<div style="color:#8FA9BF;font-size:.75rem;margin-top:8px">Share of SHAP impact for this reading</div></div>', unsafe_allow_html=True)
        with side_col:
            st.markdown(insights_html(row_now), unsafe_allow_html=True)
            st.markdown(actions_html(sv), unsafe_allow_html=True)

        st.markdown(strip_html(row_now, sv), unsafe_allow_html=True)

    home_panel()

# ------------------------------------------------------------------
# PAGE: DATA EXPLORER
# ------------------------------------------------------------------
elif page == PAGES[1]:
    st.markdown('<div class="sec">Data explorer</div>', unsafe_allow_html=True)
    st.caption(f"Data used: {source}")
    st.dataframe(data.round(2), height=300)
    st.markdown("**Water use vs temperature**")
    st.scatter_chart(data, x=TEMP, y=TARGET, color="#27D8E8", height=320)
    st.markdown("**Summary**")
    st.dataframe(data.describe().round(2))

# ------------------------------------------------------------------
# PAGE: XAI EXPLANATIONS
# ------------------------------------------------------------------
elif page == PAGES[2]:
    st.markdown('<div class="sec">XAI explanations</div>', unsafe_allow_html=True)
    st.markdown("SHAP shows how much each input pushes the predicted water use up or down.")
    st.markdown("**Overall: which inputs matter most across the dataset**")
    st.bar_chart(global_shap().sort_values(), height=240, color="#27D8E8", horizontal=True)
    latest = st.session_state.history[-1]
    row_now = {f: latest[f] for f in FEATURES}
    sv = shap_values(row_now)
    st.markdown(f"**Latest reading explained** (predicted {latest['Predicted water']:,.0f} {WATER_UNIT})")
    st.bar_chart(pd.Series(sv, index=FEATURES), height=240, color="#65BFFF", horizontal=True)
    st.caption(f"Positive bars push water use up, negative bars push it down (in {WATER_UNIT}). Start live monitoring on the Home page to see this change.")

# ------------------------------------------------------------------
# PAGE: WHAT-IF
# ------------------------------------------------------------------
elif page == PAGES[3]:
    st.markdown('<div class="sec">What-if comparison</div>', unsafe_allow_html=True)
    st.markdown("Change the inputs to compare a current setup with an alternative. Both results come from the trained model.")

    def scenario(title, key, defaults):
        st.markdown(f"**{title}**")
        return {
            SERVERS: float(st.slider(SERVERS, int(LOW[SERVERS]), int(HIGH[SERVERS]), int(defaults[SERVERS]), key=f"{key}_s")),
            TEMP: float(st.slider("Temperature (°C)", float(LOW[TEMP]), float(HIGH[TEMP]), float(defaults[TEMP]), key=f"{key}_t")),
            COOLING: float(st.slider(COOLING, float(LOW[COOLING]), float(HIGH[COOLING]), float(defaults[COOLING]), key=f"{key}_c")),
            MINING: int(st.toggle("Crypto mining active", value=bool(defaults[MINING]), key=f"{key}_m")),
        }

    base = {f: float(MEDIAN[f]) for f in FEATURES}
    base[MINING] = 1
    col_a, col_b = st.columns(2)
    with col_a:
        current = scenario("Current setup", "cur", base)
    with col_b:
        alt_default = dict(base)
        alt_default[MINING] = 0
        alternative = scenario("Alternative setup", "alt", alt_default)

    p_cur, p_alt = predict(current), predict(alternative)
    change = (p_alt - p_cur) / p_cur * 100 if p_cur else 0.0
    m1, m2, m3 = st.columns(3)
    m1.metric("Current", f"{p_cur:,.0f} {WATER_UNIT}")
    m2.metric("Alternative", f"{p_alt:,.0f} {WATER_UNIT}", f"{change:+.1f}%", delta_color="inverse")
    m3.metric("Difference", f"{p_cur - p_alt:+,.0f} {WATER_UNIT}")
    st.bar_chart(pd.DataFrame({"Predicted water": [p_cur, p_alt]}, index=["Current", "Alternative"]), height=250, color="#27D8E8")
    st.caption("This compares model predictions. It does not prove real savings, and it does not check performance limits of a real facility.")

# ------------------------------------------------------------------
# PAGE: VALIDATION
# ------------------------------------------------------------------
elif page == PAGES[4]:
    st.markdown('<div class="sec">Validation</div>', unsafe_allow_html=True)
    v1, v2, v3 = st.columns(3)
    v1.metric("R² (unseen data)", f"{metrics['r2']:.3f}")
    v2.metric("Average error (MAE)", f"{metrics['mae']:,.1f} {WATER_UNIT}")
    v3.metric("Rows in dataset", f"{metrics['n_rows']:,}")
    st.caption(f"Scoring method: {metrics['method']}. Every score is computed on rows the model did not train on.")
    if metrics["small"]:
        st.warning(
            f"The dataset has only {metrics['n_rows']} rows. Scores from so little data are only indicative. "
            "Treat this as a working prototype and test it on a larger dataset before drawing conclusions."
        )
    st.markdown("**Which inputs matter most overall?**")
    st.bar_chart(pd.Series(model.feature_importances_, index=FEATURES).sort_values(), height=230, color="#65BFFF", horizontal=True)
    st.markdown("**Data and limits (please read)**")
    st.markdown(
        f"""
- **Data used:** {source}.
- **Inputs:** active servers, temperature, cooling hours, crypto-mining activity.
- **The data is simulated IoT sensor data.** Results show how the framework behaves, not measured water use in a real data centre.
- **Live stream:** readings come from a simulator. No physical sensor is connected yet. ESP32 integration is the next step.
- **Alert rule:** Warning if temperature is above {TEMP_ALERT} °C or mining is active. Critical if both.
- **Model estimates are not measured leaks.** Always check real equipment before acting.
"""
    )

# ------------------------------------------------------------------
# PAGE: ARCHITECTURE
# ------------------------------------------------------------------
else:
    st.markdown('<div class="sec">Architecture</div>', unsafe_allow_html=True)
    st.graphviz_chart(
        """
digraph G {
  rankdir=LR;
  node [shape=box, style="rounded,filled", fillcolor="#0B2340", color="#27D8E8", fontcolor="#F3F8FF", fontname="Helvetica"];
  edge [color="#65BFFF"];
  A [label="IoT sensors\\n(simulated now,\\nESP32 next)"];
  B [label="Data check\\n(ranges, missing values)"];
  C [label="Random Forest\\nwater prediction"];
  D [label="SHAP\\nwhy this prediction"];
  E [label="Alert engine\\nFix first"];
  F [label="Streamlit\\ndashboard"];
  A -> B -> C -> D -> E -> F;
}
"""
    )
    st.markdown("Most dashboards stop after showing the number. EcoXAI adds the next two steps: it explains which input caused the water use, and it tells the operator what to check first.")

st.markdown(
    '<div class="footer"><span>Saving Water. Explaining Why. Intelligently.</span>'
    '<span>Developed by <b style="color:#27D8E8">SRI BHARATHI VN</b>, III B.Sc. Computer Science with Cognitive Systems</span></div>',
    unsafe_allow_html=True,
)
