"""
EcoXAI 2.0 - Explainable IoT Water Intelligence for AI Data Centres
Developed by SRI BHARATHI VN, III B.Sc. Computer Science with Cognitive Systems

How this app gets its numbers (nothing is hard-coded):
  1. It looks for your results CSV in the same folder as this file.
  2. It trains a Random Forest on it and checks it on held-out test data.
  3. SHAP explains every prediction.
  4. If no CSV is found it uses a built-in SIMULATED dataset and says so on screen.
"""

import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# SETTINGS you may want to edit
# ------------------------------------------------------------------
WATER_UNIT = "L"          # unit of the water column in your dataset
TEMP_ALERT = 35           # temperature (deg C) above which we raise a warning
REFRESH_SECONDS = 3       # how often the simulated sensor sends a new reading
HISTORY_LENGTH = 40       # readings kept on the monitoring chart
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Names used inside the app (your CSV column names are auto-detected)
SERVERS, TEMP, COOLING, MINING = "Active Servers", "Temperature", "Cooling Duration", "Mining Active"
FEATURES = [SERVERS, TEMP, COOLING, MINING]
TARGET = "Water"

ACTIONS = {
    TEMP: "Check cooling set-points and airflow first.",
    SERVERS: "Review server load. Can idle servers be consolidated or switched off?",
    COOLING: "Review the cooling schedule. Long cooling duration adds water use.",
    MINING: "Review the crypto-mining workload. Reduce it or move it to cooler hours.",
}

# ------------------------------------------------------------------
# PAGE + STYLE (dark navy and cyan control room)
# ------------------------------------------------------------------
st.set_page_config(page_title="EcoXAI 2.0", page_icon="💧", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Manrope', sans-serif; }
.stApp { background:#061526; }
[data-testid="stHeader"] { background:rgba(6,21,38,.92); }
.banner { background:linear-gradient(90deg,#04142B 0%,#06335A 60%,#05243F 100%); border:1px solid #27D8E8; border-radius:16px; padding:28px 32px; margin-bottom:14px; }
.banner h1 { color:#F3F8FF; margin:0; font-size:2.3rem; font-weight:800; }
.banner p { color:#9FE9F2; margin:6px 0 0 0; font-size:1.02rem; }
.card { background:#0B2340; border:1px solid #19466B; border-left:6px solid #27D8E8; border-radius:12px; padding:14px 16px; height:100%; }
.card.main { border-left-color:#FFD166; background:#0D2B4C; }
.card .lbl { color:#A8BED2; font-size:.85rem; font-weight:600; }
.card .val { color:#F3F8FF; font-size:1.7rem; font-weight:800; line-height:1.25; }
.card.main .val { font-size:2.3rem; color:#FFD166; }
.card .sub { color:#8FA9BF; font-size:.78rem; }
.fix { border-radius:14px; padding:18px 20px; border:1px solid #19466B; background:#0B2340; }
.fix.normal { border-left:8px solid #55D6A4; }
.fix.warning { border-left:8px solid #FF9F43; background:#2A2418; }
.fix.critical { border-left:8px solid #FF5D68; background:#2D1A24; }
.badge { display:inline-block; padding:3px 12px; border-radius:20px; font-weight:800; font-size:.85rem; color:#061526; }
.badge.normal { background:#55D6A4; } .badge.warning { background:#FF9F43; } .badge.critical { background:#FF5D68; }
.fix h4 { margin:10px 0 6px 0; color:#F3F8FF; }
.fix p, .fix li { color:#F3F8FF; margin-bottom:4px; }
.footer { border-top:1px solid #19466B; margin-top:34px; padding:18px 0 6px 0; text-align:center; color:#A8BED2; }
.footer b { color:#27D8E8; }
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
    used = set()
    found = {}
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
    return out[FEATURES + [TARGET]] if len(out) >= 50 else None


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
    return _simulated(), "built-in simulated dataset", True


@st.cache_resource(show_spinner="Training the Random Forest model...")
def build_model():
    data, source, is_fallback = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=150, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    metrics = {
        "r2": r2_score(y_test, predicted),
        "mae": mean_absolute_error(y_test, predicted),
        "n_train": len(x_train),
        "n_test": len(x_test),
    }
    return model, shap.TreeExplainer(model), data, source, is_fallback, metrics


model, explainer, data, source, is_fallback, metrics = build_model()
LOW, HIGH = data[FEATURES].min(), data[FEATURES].max()


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
    hot = row[TEMP] > TEMP_ALERT
    mining = bool(row[MINING])
    if hot and mining:
        return "critical", "Critical"
    if hot or mining:
        return "warning", "Warning"
    return "normal", "Normal"


def card(label, value, sub="", main=False):
    cls = "card main" if main else "card"
    return f'<div class="{cls}"><div class="lbl">{label}</div><div class="val">{value}</div><div class="sub">{sub}</div></div>'


def fix_first_html(row, sv):
    level, label = status_of(row)
    order = np.argsort(-np.abs(sv))[:2]
    reasons = ""
    for i in order:
        direction = "pushing water use up" if sv[i] > 0 else "keeping water use down"
        reasons += f"<li><b>{FEATURES[i]}</b> ({show(FEATURES[i], row[FEATURES[i]])}) is {direction} by about {abs(sv[i]):,.0f} {WATER_UNIT}.</li>"
    rising = [i for i in np.argsort(-sv) if sv[i] > 0]
    action = ACTIONS[FEATURES[rising[0]]] if rising else "No factor is pushing water use up right now. Keep monitoring."
    why = []
    if row[TEMP] > TEMP_ALERT:
        why.append(f"temperature is above {TEMP_ALERT} °C")
    if row[MINING]:
        why.append("crypto mining is active")
    trigger = "Alert reason: " + " and ".join(why) + "." if why else "No alert rule is triggered."
    return (
        f'<div class="fix {level}"><span class="badge {level}">{label}</span>'
        f"<h4>Fix first</h4><p style='margin:0 0 8px 0'><b>{action}</b></p>"
        f"<div style='color:#A8BED2;font-size:.85rem'>Why the model says this:</div><ul>{reasons}</ul>"
        f"<div style='color:#8FA9BF;font-size:.8rem'>{trigger} These are model estimates, not measured leaks.</div></div>"
    )


def next_reading(prev, rng):
    row = dict(prev)
    row[SERVERS] = float(np.clip(prev[SERVERS] + rng.integers(-10, 11), LOW[SERVERS], HIGH[SERVERS]))
    row[TEMP] = float(np.clip(prev[TEMP] + rng.uniform(-1.2, 1.5), LOW[TEMP], HIGH[TEMP]))
    row[COOLING] = float(np.clip(prev[COOLING] + rng.uniform(-0.8, 0.8), LOW[COOLING], HIGH[COOLING]))
    if rng.random() < 0.12:
        row[MINING] = 1 - int(prev[MINING])
    return row


# ------------------------------------------------------------------
# SESSION STATE (live stream memory)
# ------------------------------------------------------------------
if "history" not in st.session_state:
    start = {f: float(data[f].median()) for f in FEATURES}
    start[MINING] = 0
    start["Predicted water"] = predict(start)
    start["Reading"] = 1
    st.session_state.history = [start]
if "running" not in st.session_state:
    st.session_state.running = False


# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.markdown(
    """
<div class="banner">
<h1>EcoXAI 2.0</h1>
<p>Explainable IoT water intelligence for AI data centres and crypto mining. Saving water. Explaining why. Intelligently.</p>
</div>
""",
    unsafe_allow_html=True,
)

if is_fallback:
    st.warning(
        "Your results CSV was not found next to app.py, so this app is using a built-in SIMULATED dataset. "
        "Copy your CSV into the same folder to use your own data."
    )
else:
    st.success(f"Model trained on {source}. Sensor readings in the live stream are simulated.")

tab_live, tab_what, tab_valid, tab_arch = st.tabs(["Live monitoring", "What-if", "Validation", "Architecture"])


# ------------------------------------------------------------------
# TAB 1: LIVE MONITORING
# ------------------------------------------------------------------
with tab_live:
    left, right = st.columns([2, 1])
    left.toggle("Start live monitoring (simulated IoT sensor stream)", key="running")
    if right.button("Reset stream"):
        st.session_state.history = st.session_state.history[:1]

    run_every = REFRESH_SECONDS if st.session_state.running else None

    @st.fragment(run_every=run_every)
    def live_panel():
        history = st.session_state.history
        if st.session_state.running:
            row = next_reading({f: history[-1][f] for f in FEATURES}, np.random.default_rng())
            row["Predicted water"] = predict(row)
            row["Reading"] = history[-1]["Reading"] + 1
            history.append(row)
            del history[:-HISTORY_LENGTH]
        now = history[-1]
        cards = st.columns([1.6, 1, 1, 1, 1])
        cards[0].markdown(card("Predicted water use", f"{now['Predicted water']:,.0f} {WATER_UNIT}", "Random Forest estimate", main=True), unsafe_allow_html=True)
        cards[1].markdown(card(SERVERS, show(SERVERS, now[SERVERS])), unsafe_allow_html=True)
        cards[2].markdown(card(TEMP, show(TEMP, now[TEMP])), unsafe_allow_html=True)
        cards[3].markdown(card(COOLING, show(COOLING, now[COOLING])), unsafe_allow_html=True)
        cards[4].markdown(card("Crypto mining", show(MINING, now[MINING])), unsafe_allow_html=True)

        chart_col, fix_col = st.columns([3, 2])
        with chart_col:
            st.markdown("**Predicted water use over time**")
            frame = pd.DataFrame(history).set_index("Reading")[["Predicted water"]]
            st.line_chart(frame, height=300, color="#27D8E8")
            st.caption("Each point is one simulated sensor reading passed through the trained model.")
        with fix_col:
            row_now = {f: now[f] for f in FEATURES}
            st.markdown(fix_first_html(row_now, shap_values(row_now)), unsafe_allow_html=True)

    live_panel()


# ------------------------------------------------------------------
# TAB 2: WHAT-IF
# ------------------------------------------------------------------
with tab_what:
    st.markdown("Change the inputs to compare a current setup with an alternative. Both results come from the trained model.")

    def scenario(title, key, defaults):
        st.markdown(f"**{title}**")
        return {
            SERVERS: float(st.slider(SERVERS, int(LOW[SERVERS]), int(HIGH[SERVERS]), int(defaults[SERVERS]), key=f"{key}_s")),
            TEMP: float(st.slider("Temperature (°C)", float(LOW[TEMP]), float(HIGH[TEMP]), float(defaults[TEMP]), key=f"{key}_t")),
            COOLING: float(st.slider(COOLING, float(LOW[COOLING]), float(HIGH[COOLING]), float(defaults[COOLING]), key=f"{key}_c")),
            MINING: int(st.toggle("Crypto mining active", value=bool(defaults[MINING]), key=f"{key}_m")),
        }

    base_defaults = {f: float(data[f].median()) for f in FEATURES}
    base_defaults[MINING] = 1
    col_a, col_b = st.columns(2)
    with col_a:
        current = scenario("Current setup", "cur", base_defaults)
    with col_b:
        alt_defaults = dict(base_defaults)
        alt_defaults[MINING] = 0
        alternative = scenario("Alternative setup", "alt", alt_defaults)

    p_cur, p_alt = predict(current), predict(alternative)
    change = (p_alt - p_cur) / p_cur * 100 if p_cur else 0.0
    m1, m2, m3 = st.columns(3)
    m1.metric("Current", f"{p_cur:,.0f} {WATER_UNIT}")
    m2.metric("Alternative", f"{p_alt:,.0f} {WATER_UNIT}", f"{change:+.1f}%", delta_color="inverse")
    m3.metric("Difference", f"{p_cur - p_alt:+,.0f} {WATER_UNIT}")
    st.bar_chart(pd.DataFrame({"Predicted water": [p_cur, p_alt]}, index=["Current", "Alternative"]), height=260, color="#27D8E8")
    st.caption("This compares model predictions. It does not prove real savings, and it does not check performance limits of a real facility.")


# ------------------------------------------------------------------
# TAB 3: VALIDATION
# ------------------------------------------------------------------
with tab_valid:
    st.markdown("**How well does the model work on data it has not seen?**")
    v1, v2, v3 = st.columns(3)
    v1.metric("R² on test data", f"{metrics['r2']:.3f}")
    v2.metric(f"Average error (MAE)", f"{metrics['mae']:,.1f} {WATER_UNIT}")
    v3.metric("Train / test rows", f"{metrics['n_train']:,} / {metrics['n_test']:,}")
    st.caption("The model is trained on 80% of the data and scored on the other 20%.")

    st.markdown("**Which inputs matter most overall?**")
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    st.bar_chart(importance, height=240, color="#65BFFF", horizontal=True)

    st.markdown("**Data and limits (please read)**")
    st.markdown(
        f"""
- **Data used:** {source}.
- **Inputs:** active servers, temperature, cooling duration, crypto-mining activity.
- **The data is simulated IoT sensor data.** Results show how the framework behaves. They are not measured water use from a real data centre.
- **Live stream:** readings are generated by a simulator. No physical sensor is connected yet. ESP32 integration is the next step.
- **Alert rule:** Warning if temperature is above {TEMP_ALERT} °C or mining is active. Critical if both.
- **Model estimates are not measured leaks.** Always check real equipment before acting.
"""
    )


# ------------------------------------------------------------------
# TAB 4: ARCHITECTURE
# ------------------------------------------------------------------
with tab_arch:
    st.markdown("**How data flows through EcoXAI 2.0**")
    st.graphviz_chart(
        """
digraph G {
  rankdir=LR;
  node [shape=box, style="rounded,filled", fillcolor="#0B2340", color="#27D8E8", fontcolor="#F3F8FF", fontname="Helvetica"];
  A [label="IoT sensors\\n(simulated now,\\nESP32 next)"];
  B [label="Data check\\n(ranges, missing values)"];
  C [label="Random Forest\\nwater prediction"];
  D [label="SHAP\\nwhy this prediction"];
  E [label="Alert engine\\nFix first"];
  F [label="Streamlit\\ndashboard"];
  edge [color="#65BFFF"];
  A -> B -> C -> D -> E -> F;
}
"""
    )
    st.markdown(
        "Most dashboards stop after showing the number. EcoXAI adds the next two steps: "
        "it explains which input caused the water use, and it tells the operator what to check first."
    )

st.markdown(
    """
<div class="footer">
<div><b>Developed by SRI BHARATHI VN</b></div>
<div>III B.Sc. Computer Science with Cognitive Systems</div>
</div>
""",
    unsafe_allow_html=True,
)
