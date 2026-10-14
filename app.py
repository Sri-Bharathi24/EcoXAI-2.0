"""EcoXAI 2.0: Explainable IoT Water Intelligence for AI Data Centres.

Run:  pip install streamlit pandas numpy plotly
      streamlit run app.py

Optional: put a trained model next to this file as `ecoxai_model.pkl`
(joblib, multi-output regressor taking [servers, temperature, cooling_hours, mining]
and returning [energy_kWh, cooling_water_L, carbon_kg]). The app uses it
automatically; otherwise it falls back to clearly-labelled demo equations.
"""
import inspect
import os
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="EcoXAI 2.0", page_icon="💧", layout="wide")

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
.stApp { background: #f4f8f5; }
.hero { background: linear-gradient(135deg,#173b32,#2f7d62); color:#fff;
        padding: 28px 32px; border-radius: 18px; margin-bottom: 18px; }
.hero h1 { margin:0; font-size: 2.2rem; color:#fff; }
.hero p { margin: 6px 0 0 0; opacity:.9; font-size:1.05rem; }
.section-title { font-size:1.25rem; font-weight:700; color:#173b32; margin: 18px 0 2px 0; }
.section-sub { color:#687a72; font-size:.92rem; margin-bottom:10px; }
.card { background:#fff; border:1px solid #dbe7e0; border-radius:14px;
        padding:14px 16px; min-height:104px; box-shadow:0 1px 3px rgba(23,59,50,.06); }
.card-label { color:#687a72; font-size:.82rem; text-transform:uppercase; letter-spacing:.04em; }
.card-value { color:#173b32; font-size:1.7rem; font-weight:700; margin-top:4px; }
.card-note { color:#8aa096; font-size:.78rem; margin-top:2px; }
.notice { background:#fff8e6; border-left:4px solid #e0a800; padding:10px 14px;
          border-radius:8px; color:#5c4a00; font-size:.9rem; margin-top:14px; }
.good { background:#e8f5ee; border-left:4px solid #2f7d62; padding:10px 14px;
        border-radius:8px; color:#173b32; margin-top:14px; }
.fix-high { background:#fdecec; border-left:4px solid #d64545; padding:9px 13px; border-radius:8px; margin:6px 0; color:#5a1a1a; }
.fix-med  { background:#fff4e0; border-left:4px solid #e08a00; padding:9px 13px; border-radius:8px; margin:6px 0; color:#5a3a00; }
.fix-ok   { background:#e8f5ee; border-left:4px solid #2f7d62; padding:9px 13px; border-radius:8px; margin:6px 0; color:#173b32; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- helpers
def _accepts(func, name):
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def show_chart(fig):
    """Works on both older and newer Streamlit (use_container_width was deprecated)."""
    if _accepts(st.plotly_chart, "width"):
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def show_table(df):
    if _accepts(st.dataframe, "width"):
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------- model layer
FEATURES = ["servers", "temperature", "cooling_hours", "mining"]
EMISSION_FACTOR = 0.71  # kg CO2e per kWh: PLACEHOLDER, replace with a verified grid factor

MODEL = None
try:
    import joblib

    if os.path.exists("ecoxai_model.pkl"):
        MODEL = joblib.load("ecoxai_model.pkl")
except Exception:
    MODEL = None


def demo_predict(servers, temperature, cooling_hours, mining):
    """Illustrative equations only. NOT a trained model and NOT a measurement."""
    energy = servers * 0.4 * 24 * (1.35 if mining else 1.0)  # kWh/day
    heat_factor = 1 + 0.03 * (temperature - 20)
    cooling_factor = 0.6 + 0.4 * (cooling_hours / 24)
    water = energy * 1.8 * heat_factor * cooling_factor  # litres/day
    carbon = energy * EMISSION_FACTOR
    return {"Energy": round(energy, 1), "Cooling water": round(water, 1), "Carbon": round(carbon, 1)}


def predict(servers, temperature, cooling_hours, mining):
    if MODEL is not None:
        try:
            x = pd.DataFrame([[servers, temperature, cooling_hours, int(mining)]], columns=FEATURES)
            out = np.asarray(MODEL.predict(x)).ravel()
            return {"Energy": round(float(out[0]), 1),
                    "Cooling water": round(float(out[1]), 1),
                    "Carbon": round(float(out[2]), 1)}
        except Exception:
            pass
    return demo_predict(servers, temperature, cooling_hours, mining)


def demo_feature_scores(servers, temperature, cooling_hours, mining):
    """Demo importance scores (NOT SHAP). Replace with shap.TreeExplainer values."""
    raw = {
        "Active servers": 0.35 * servers / 100,
        "Temperature": 0.30 * (temperature - 15) / 35,
        "Cooling duration": 0.20 * cooling_hours / 24,
        "Mining activity": 0.15 * (1 if mining else 0),
    }
    total = sum(raw.values()) or 1
    return {k: v / total * 100 for k, v in raw.items()}


def fix_first(servers, temperature, cooling_hours, mining):
    """Threshold rules. Replace thresholds with verified operating limits."""
    items = []
    if temperature >= 35:
        items.append(("high", f"Temperature is {temperature:.1f} °C. Inspect cooling capacity and airflow first."))
    elif temperature >= 30:
        items.append(("med", f"Temperature is {temperature:.1f} °C and rising. Watch cooling performance."))
    if cooling_hours >= 18:
        items.append(("med", f"Cooling runs {cooling_hours:.1f} h/day. Check whether it is compensating for a hot spot."))
    if mining:
        items.append(("med", "Mining workload is active. It adds sustained heat and energy load."))
    if servers >= 70:
        items.append(("med", f"{servers} active servers. Check rack density and load balancing."))
    if not items:
        items.append(("ok", "No condition crosses the demo thresholds. Keep monitoring."))
    return items


# ---------------------------------------------------------------- header
st.markdown(
    '<div class="hero"><h1>💧 EcoXAI 2.0</h1>'
    "<p>Explainable IoT Water Intelligence for AI Data Centres: estimate, explain, then decide what to check first.</p></div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Operating conditions")
    servers = st.slider("Active servers", 1, 100, 40)
    temperature = st.slider("Temperature (°C)", 15.0, 50.0, 28.0, 0.5)
    cooling_hours = st.slider("Cooling duration (h/day)", 1.0, 24.0, 12.0, 0.5)
    mining_active = st.toggle("Mining active", value=False)
    live = st.toggle("Simulated live feed (every 3 s)", value=False)
    st.caption("Model source: " + ("trained model file" if MODEL is not None else "demo equations (no trained model loaded)"))


# ---------------------------------------------------------------- dashboard
def show_dashboard(s, t, c, mining, is_live):
    pred = predict(s, t, c, mining)

    badge = "🟢 Live (simulated)" if is_live else "⚪ Snapshot"
    st.markdown(f'<div class="section-title">Resource indicators <span style="font-size:.8rem;color:#687a72">{badge}</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Daily estimates for the selected operating conditions.</div>', unsafe_allow_html=True)

    kpis = [
        ("Energy", f"{pred['Energy']:,.0f} kWh", "per day"),
        ("Cooling water", f"{pred['Cooling water']:,.0f} L", "per day"),
        ("Carbon", f"{pred['Carbon']:,.0f} kg CO₂e", "placeholder emission factor"),
    ]
    for col, (label, value, note) in zip(st.columns(3), kpis):
        with col:
            st.markdown(f'<div class="card"><div class="card-label">{label}</div><div class="card-value">{value}</div><div class="card-note">{note}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Fix first</div>', unsafe_allow_html=True)
    css = {"high": "fix-high", "med": "fix-med", "ok": "fix-ok"}
    for level, msg in fix_first(s, t, c, mining):
        st.markdown(f'<div class="{css[level]}">{msg}</div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-title">Water vs temperature</div>', unsafe_allow_html=True)
        temps = np.arange(15, 50.5, 0.5)
        water = [predict(s, float(x), c, mining)["Cooling water"] for x in temps]
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=temps, y=water, mode="lines", line=dict(color="#2f7d62", width=3), name="Cooling water"))
        fig1.add_trace(go.Scatter(x=[t], y=[pred["Cooling water"]], mode="markers",
                                  marker=dict(size=13, color="#d64545"), name="Current"))
        fig1.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title="Temperature (°C)", yaxis_title="Water (L/day)",
                           plot_bgcolor="#fff", paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
        show_chart(fig1)

    with right:
        st.markdown('<div class="section-title">What drives the estimate</div>', unsafe_allow_html=True)
        scores = demo_feature_scores(s, t, c, mining)
        names = list(scores.keys())[::-1]
        vals = [scores[n] for n in names]
        fig2 = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color="#2f7d62",
                                text=[f"{v:.0f}%" for v in vals], textposition="outside"))
        fig2.update_layout(height=340, margin=dict(l=10, r=40, t=10, b=10),
                           xaxis_title="Demo contribution (%). Not SHAP yet.",
                           plot_bgcolor="#fff", paper_bgcolor="rgba(0,0,0,0)")
        show_chart(fig2)

    st.markdown('<div class="section-title">Operating conditions</div>', unsafe_allow_html=True)
    details = [("Active servers", str(s)), ("Temperature", f"{t:.1f} °C"),
               ("Cooling duration", f"{c:.1f} h/day"), ("Mining activity", "Active" if mining else "Inactive")]
    for col, (label, value) in zip(st.columns(4), details):
        with col:
            st.markdown(f'<div class="card"><div class="card-label">{label}</div><div class="card-value">{value}</div><div class="card-note">Simulated input</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="notice"><b>Transparency:</b> Unless a trained model file is loaded, values come from illustrative equations and demo feature scores. They are not measurements, validated predictions, or SHAP explanations.</div>', unsafe_allow_html=True)


if live and hasattr(st, "fragment"):
    @st.fragment(run_every="3s")
    def live_panel():
        # Small random variation for demonstration; not sensor data.
        s = int(np.clip(servers + random.choice([-2, -1, 0, 1, 2]), 1, 100))
        t = float(np.clip(temperature + random.choice([-0.5, 0, 0.5, 1]), 15, 50))
        c = float(np.clip(cooling_hours + random.choice([-0.5, 0, 0.5]), 1, 24))
        show_dashboard(s, t, c, mining_active, True)

    live_panel()
else:
    if live:
        st.info("Your Streamlit version has no st.fragment, so this is a static snapshot. Upgrade Streamlit for timed refresh.")
    show_dashboard(servers, temperature, cooling_hours, mining_active, False)

# ---------------------------------------------------------------- tabs
st.markdown("")
tab1, tab2, tab3, tab4 = st.tabs(["What-if lab", "Architecture", "Validation & limits", "Project story"])

with tab1:
    st.markdown('<div class="section-title">Compare two scenarios</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Change inputs and compare the output. Replace the demo calculation with your trained model before judging.</div>', unsafe_allow_html=True)
    baseline = predict(servers, temperature, cooling_hours, mining_active)
    x, y = st.columns(2)
    with x:
        st.markdown("**Baseline**")
        st.write(f"{servers} servers · {temperature:.1f}°C · {cooling_hours:.1f} cooling hours/day · Mining {'on' if mining_active else 'off'}")
    with y:
        st.markdown("**Alternative conditions**")
        alt_servers = st.slider("Servers", 1, 100, int(servers), key="alt_servers")
        alt_temp = st.slider("Temperature (°C)", 15.0, 50.0, float(max(15.0, min(50.0, temperature - 3))), 0.5, key="alt_temp")
        alt_cooling = st.slider("Cooling duration (h/day)", 1.0, 24.0, float(max(1.0, min(24.0, cooling_hours - 1))), 0.5, key="alt_cooling")
        alt_mining = st.toggle("Mining active", value=mining_active, key="alt_mining")
    alternative = predict(alt_servers, alt_temp, alt_cooling, alt_mining)
    comp = pd.DataFrame([
        {"Indicator": "Energy (kWh)", "Baseline": baseline["Energy"], "Alternative": alternative["Energy"]},
        {"Indicator": "Cooling water (L)", "Baseline": baseline["Cooling water"], "Alternative": alternative["Cooling water"]},
        {"Indicator": "Carbon (kg CO₂e)", "Baseline": baseline["Carbon"], "Alternative": alternative["Carbon"]},
    ])
    show_table(comp)
    delta = baseline["Cooling water"] - alternative["Cooling water"]
    st.info(f"Difference in cooling-water indicator: {delta:+.2f} L. Not evidence of real-world savings unless the model is validated.")

with tab2:
    st.markdown('<div class="section-title">Proposed system architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Prediction and explanation sit between data inputs and operator guidance.</div>', unsafe_allow_html=True)
    steps = [
        ("01 · Data inputs", "Dataset initially; verified temperature / power sensors in a future IoT deployment"),
        ("02 · Preprocessing", "Check missing values, units and feature order"),
        ("03 · Random Forest", "Predict supported resource indicators using the trained model"),
        ("04 · SHAP explanation", "Explain feature contributions to each prediction"),
        ("05 · Decision support", "Threshold rules, recommendations and what-if comparison"),
        ("06 · Dashboard", "Readable indicators, explanations and limitations"),
    ]
    for title, desc in steps:
        st.markdown(f'<div class="card" style="min-height:0;margin:8px 0;"><b style="color:#173b32">{title}</b><div style="color:#687a72;font-size:.9rem;margin-top:4px">{desc}</div></div>', unsafe_allow_html=True)
    st.caption("Do not claim physical sensor deployment unless hardware has been connected and tested.")

with tab3:
    st.markdown('<div class="section-title">Validation & limitations</div>', unsafe_allow_html=True)
    st.markdown("""
**Current status:** Without `ecoxai_model.pkl` this app is a UI prototype using demo equations.

Before presenting validation, add:
- Held-out test-set R² and MAE for each model target.
- Dataset source, row count, features and units.
- Train/test split method and preprocessing details.
- Actual SHAP values from the same trained model.

**Limitations**
- Monitoring values are simulated.
- Demo calculations are not measurements or validated predictions.
- Historical or synthetic data may not generalise to real facilities.
- Carbon estimates need a verified electricity-emissions factor.
- IoT sensor integration is future work unless already implemented and tested.
""")
    st.info("Replace this section with genuine metrics from your held-out test set. Do not invent scores.")

with tab4:
    st.markdown('<div class="section-title">EcoXAI 2.0 in one minute</div>', unsafe_allow_html=True)
    st.markdown("""
**Problem:** Data-centre and mining workloads consume energy and require cooling. Operators need understandable ways to examine resource impact.

**Solution:** EcoXAI combines resource prediction with explainability, helping users explore both an estimate and the input features that influenced it.

**Difference:** The focus is interpretable sustainability decision support, with a pathway toward IoT sensor integration.

**Next step:** Connect validated trained models and verified sensor data, then evaluate with real operational data.
""")
    st.markdown('<div class="good"><b>Presentation line:</b> “EcoXAI aims to make sustainability predictions understandable, so operators can investigate the factors behind resource impact and decide what to check first.”</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("EcoXAI 2.0: Explainable IoT Water Intelligence for AI Data Centres")
st.markdown("<div style='text-align:center;color:#687a72;font-size:0.85rem;margin-top:8px;'>Developed by <b>SRI BHARATHI VN</b><br/>III B.Sc. Computer Science with Cognitive Systems</div>", unsafe_allow_html=True)
