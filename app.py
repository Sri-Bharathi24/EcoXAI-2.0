
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="EcoXAI 2.0 | Water Intelligence",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- COLOURS & STYLE --------------------
NAVY = "#071626"
PANEL = "#10253A"
SKY = "#79B9FF"
AQUA = "#42D9D0"
YELLOW = "#FFD166"
ORANGE = "#FFB454"
MINT = "#55D6A4"
WHITE = "#F4F8FC"
MUTED = "#A8BACB"
BORDER = "#25435F"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {{
    background: {NAVY};
    color: {WHITE};
    font-family: 'Inter', sans-serif;
}}
[data-testid="stHeader"] {{
    background: {NAVY};
}}
[data-testid="stSidebar"] {{
    background: #0B2035;
    border-right: 1px solid #1C4765;
}}
[data-testid="stSidebar"] * {{
    color: {WHITE};
}}
h1, h2, h3, p, label {{
    color: {WHITE};
}}
.stCaption, small {{
    color: {MUTED} !important;
}}
div[data-testid="stMetric"] {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 15px;
    padding: 17px;
}}
div[data-testid="stMetricLabel"] p {{
    color: {MUTED} !important;
}}
div[data-testid="stMetricValue"] {{
    color: {YELLOW} !important;
    font-weight: 800;
}}
div[data-testid="stMetricDelta"] {{
    color: {AQUA} !important;
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background: {NAVY};
}}
.stTabs [data-baseweb="tab"] {{
    background: {PANEL};
    color: {WHITE};
    border-radius: 8px 8px 0 0;
    padding: 10px 14px;
}}
.stTabs [aria-selected="true"] {{
    background: #164A68 !important;
    color: #FFFFFF !important;
    border-bottom: 2px solid {AQUA};
}}
.stButton > button {{
    background: linear-gradient(100deg, #087EBA, #087F8C);
    color: white;
    border: 1px solid #42D9D0;
    border-radius: 10px;
    font-weight: 700;
    min-height: 43px;
}}
.stButton > button:hover {{
    border-color: {YELLOW};
    color: white;
}}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {{
    background: {PANEL};
    border-color: {BORDER};
}}
.hero {{
    background: linear-gradient(120deg, #0A2340, #104A73 60%, #176D8C);
    border: 1px solid {AQUA};
    border-radius: 22px;
    padding: 32px 30px;
    margin-bottom: 22px;
    box-shadow: 0 8px 35px rgba(0, 190, 220, 0.10);
}}
.hero-title {{
    font-size: 2.7rem;
    font-weight: 800;
    color: white;
    margin: 0;
}}
.hero-tagline {{
    color: #E4F0F8;
    font-size: 1.05rem;
    margin-top: 10px;
}}
.hero-label {{
    color: #9BECE7;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 0.76rem;
    font-weight: 700;
}}
.section-title {{
    color: {WHITE};
    font-size: 1.35rem;
    font-weight: 800;
    margin: 12px 0 5px 0;
}}
.info-card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 18px;
    height: 100%;
}}
.info-title {{
    color: {SKY};
    font-weight: 700;
    font-size: 1.02rem;
}}
.info-copy {{
    color: {MUTED};
    font-size: 0.92rem;
    line-height: 1.6;
}}
.alert-box {{
    background: #35291B;
    border: 1px solid #75552A;
    border-left: 5px solid {ORANGE};
    border-radius: 10px;
    padding: 14px 16px;
    color: #FFE3B0;
    margin: 10px 0;
}}
.good-box {{
    background: #102F32;
    border: 1px solid #225A58;
    border-left: 5px solid {MINT};
    border-radius: 10px;
    padding: 14px 16px;
    color: #C9F7E4;
    margin: 10px 0;
}}
.note {{
    color: {MUTED};
    font-size: 0.82rem;
    line-height: 1.5;
}}
.footer {{
    border-top: 1px solid {BORDER};
    text-align: center;
    margin-top: 35px;
    padding: 22px 8px 10px 8px;
    color: {MUTED};
}}
.footer-name {{
    color: {AQUA};
    font-weight: 800;
    font-size: 1rem;
}}
</style>
""", unsafe_allow_html=True)


# -------------------- SIDEBAR CONTROLS --------------------
with st.sidebar:
    st.markdown("## 💧 EcoXAI 2.0")
    st.caption("WATER INTELLIGENCE")
    st.divider()

    st.markdown("### Operating conditions")
    servers = st.slider("Active servers", 10, 500, 250)
    temperature = st.slider("Temperature (°C)", 15, 50, 28)
    cooling_hours = st.slider("Cooling hours per day", 1.0, 24.0, 6.5, 0.5)
    mining_active = st.toggle("Crypto-mining workload", value=False)

    st.divider()
    st.markdown("### Demo settings")
    simulated_live = st.toggle("Simulate changing readings", value=False)
    st.caption(
        "Demo only: these controls do not read physical IoT sensors."
    )

    run_prediction = st.button(
        "💧 Update example",
        use_container_width=True
    )


# -------------------- DEMO CALCULATIONS --------------------
# Illustrative formulas only. Replace with validated model predictions.
water_l = (
    150
    + servers * 7.5
    + max(temperature - 20, 0) * 12
    + cooling_hours * 18
    + (120 if mining_active else 0)
)
energy_kwh = (
    servers * 9.5
    + max(temperature - 20, 0) * 3
    + cooling_hours * 4
    + (180 if mining_active else 0)
)
carbon_kg = energy_kwh * 0.71  # Example factor, not a verified local factor.

if simulated_live:
    # Small visual variation to demonstrate how a live dashboard might behave.
    rng = np.random.default_rng()
    water_display = max(0, water_l + rng.normal(0, water_l * 0.025))
    energy_display = max(0, energy_kwh + rng.normal(0, energy_kwh * 0.02))
    carbon_display = energy_display * 0.71
else:
    water_display = water_l
    energy_display = energy_kwh
    carbon_display = carbon_kg


# -------------------- HERO --------------------
st.markdown("""
<div class="hero">
    <div class="hero-label">Sustainable Computing · Water Intelligence</div>
    <div class="hero-title">💧 EcoXAI 2.0</div>
    <div class="hero-tagline">
        Saving Water · Explaining Why · Making Smarter Decisions
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "**Understand water use. Discover the reasons. Know what to check next.**"
)
st.caption(
    "Demonstration dashboard · Values below come from illustrative formulas, "
    "not a trained model or physical sensors."
)


# -------------------- KEY INDICATORS --------------------
st.markdown('<div class="section-title">Your resource snapshot</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("💧 Cooling-water indicator", f"{water_display:,.0f} L")
    st.caption("Illustrative daily indicator")

with c2:
    st.metric("⚡ Energy indicator", f"{energy_display:,.0f} kWh")
    st.caption("Illustrative daily indicator")

with c3:
    st.metric("🌍 Carbon indicator", f"{carbon_display:,.0f} kg CO₂e")
    st.caption("Uses an example emissions factor")


# -------------------- SIMPLE ALERTS --------------------
st.markdown('<div class="section-title">What should you check first?</div>',
            unsafe_allow_html=True)

if temperature >= 35:
    st.markdown(
        f'<div class="alert-box">🌡️ <b>Temperature is high in this scenario.</b>'
        f'<br>Current example input: {temperature}°C. '
        'Check cooling conditions and compare against your facility’s '
        'approved operating limits.</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f'<div class="good-box">✓ Temperature input is {temperature}°C. '
        'This demo alert threshold has not been crossed. '
        'This does not confirm that a real facility is safe.</div>',
        unsafe_allow_html=True
    )

if mining_active:
    st.info(
        "Crypto-mining is enabled in this example. "
        "Its real impact depends on hardware, workload, cooling design, "
        "and measured operating data."
    )

st.caption(
    "Alerts are simple demo rules, not professional facility-safety advice."
)


# -------------------- TABS --------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🌊 Water Explorer",
    "🧠 Why it changes",
    "🔄 What-if comparison",
    "ℹ️ About & validation"
])


# -------------------- TAB 1: OVERVIEW --------------------
with tab1:
    left, right = st.columns([1.15, 0.85])

    with left:
        st.markdown("### Water indicator by temperature")

        temperatures = np.arange(15, 51, 1)
        water_curve = [
            150 + servers * 7.5
            + max(t - 20, 0) * 12
            + cooling_hours * 18
            + (120 if mining_active else 0)
            for t in temperatures
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=temperatures,
            y=water_curve,
            mode="lines",
            line=dict(color=AQUA, width=3),
            fill="tozeroy",
            fillcolor="rgba(66,217,208,0.10)",
            name="Example indicator"
        ))
        fig.add_vline(
            x=temperature,
            line_dash="dash",
            line_color=YELLOW,
            annotation_text="Selected temperature",
            annotation_font_color=YELLOW
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=PANEL,
            font=dict(color=WHITE),
            xaxis_title="Temperature (°C)",
            yaxis_title="Illustrative water indicator (L/day)",
            margin=dict(l=10, r=10, t=20, b=10),
            height=360
        )
        fig.update_xaxes(gridcolor=BORDER)
        fig.update_yaxes(gridcolor=BORDER)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### How to read this")
        st.markdown(f"""
        <div class="info-card">
            <div class="info-title">💧 Water</div>
            <p class="info-copy">
            This example changes when the selected server count, temperature,
            cooling duration, or mining setting changes.
            </p>
            <div class="info-title">🔎 Explanation</div>
            <p class="info-copy">
            A real XAI system should show which inputs influenced its model
            prediction. This demo chart shows formula behaviour only.
            </p>
            <div class="info-title">🎯 Next step</div>
            <p class="info-copy">
            Connect measured data and a tested model before using the output
            for real operating decisions.
            </p>
        </div>
        """, unsafe_allow_html=True)


# -------------------- TAB 2: WATER EXPLORER --------------------
with tab2:
    st.markdown("### Explore the selected conditions")
    st.write(
        "Change the controls in the left panel to see how this demo's "
        "illustrative water indicator responds."
    )

    fig2 = go.Figure(go.Bar(
        x=["Water", "Energy", "Carbon"],
        y=[water_display, energy_display, carbon_display],
        marker_color=[AQUA, SKY, YELLOW],
        text=[
            f"{water_display:,.0f}",
            f"{energy_display:,.0f}",
            f"{carbon_display:,.0f}"
        ],
        textposition="outside"
    ))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL,
        font=dict(color=WHITE),
        yaxis_title="Illustrative value (different units)",
        showlegend=False,
        margin=dict(l=10, r=10, t=25, b=10),
        height=360
    )
    fig2.update_xaxes(gridcolor=BORDER)
    fig2.update_yaxes(gridcolor=BORDER)
    st.plotly_chart(fig2, use_container_width=True)

    st.warning(
        "Water, energy, and carbon use different units. "
        "This chart is for visual exploration, not a direct comparison "
        "of their magnitudes."
    )


# -------------------- TAB 3: EXPLANATION --------------------
with tab3:
    st.markdown("### Why does the example change?")
    st.write(
        "These are the inputs used by the demonstration formula. "
        "They are not SHAP values and do not show scientifically measured "
        "causal effects."
    )

    drivers = pd.DataFrame({
        "Input": [
            "Active servers",
            "Temperature",
            "Cooling duration",
            "Mining setting"
        ],
        "Current value": [
            f"{servers} servers",
            f"{temperature} °C",
            f"{cooling_hours:.1f} hours/day",
            "On" if mining_active else "Off"
        ],
        "What it represents": [
            "How many servers are active in this scenario",
            "The selected temperature input",
            "The example cooling duration",
            "Whether the example mining workload is enabled"
        ]
    })
    st.dataframe(drivers, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="info-card">
        <div class="info-title">What makes Explainable AI useful?</div>
        <p class="info-copy">
        A trained model can produce a prediction, but operators also need
        to understand which inputs influenced it. EcoXAI 2.0 is designed
        to use SHAP to explain predictions after the real model is connected
        and tested.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "The current version does not calculate real SHAP explanations."
    )


# -------------------- TAB 4: WHAT-IF --------------------
with tab4:
    st.markdown("### What if operating conditions changed?")
    st.write(
        "Compare the current scenario with alternative inputs. "
        "Both results below use illustrative formulas."
    )

    a, b = st.columns(2)

    with a:
        st.markdown("#### Current scenario")
        st.write(f"Servers: **{servers}**")
        st.write(f"Temperature: **{temperature}°C**")
        st.write(f"Cooling: **{cooling_hours:.1f} hours/day**")
        st.write(f"Mining: **{'On' if mining_active else 'Off'}**")

    with b:
        st.markdown("#### Try alternative conditions")
        alt_servers = st.slider(
            "Alternative servers", 10, 500, min(servers, 500), key="alt_servers"
        )
        alt_temp = st.slider(
            "Alternative temperature (°C)", 15, 50,
            temperature, key="alt_temp"
        )
        alt_cooling = st.slider(
            "Alternative cooling hours/day", 1.0, 24.0,
            cooling_hours, 0.5, key="alt_cooling"
        )
        alt_mining = st.toggle(
            "Alternative mining workload", value=mining_active, key="alt_mining"
        )

    alt_water = (
        150 + alt_servers * 7.5
        + max(alt_temp - 20, 0) * 12
        + alt_cooling * 18
        + (120 if alt_mining else 0)
    )
    alt_energy = (
        alt_servers * 9.5
        + max(alt_temp - 20, 0) * 3
        + alt_cooling * 4
        + (180 if alt_mining else 0)
    )
    alt_carbon = alt_energy * 0.71

    comparison = pd.DataFrame({
        "Indicator": ["Cooling-water demo value", "Energy demo value", "Carbon demo value"],
        "Current scenario": [
            f"{water_display:,.1f} L",
            f"{energy_display:,.1f} kWh",
            f"{carbon_display:,.1f} kg CO₂e"
        ],
        "Alternative scenario": [
            f"{alt_water:,.1f} L",
            f"{alt_energy:,.1f} kWh",
            f"{alt_carbon:,.1f} kg CO₂e"
        ]
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    difference = water_display - alt_water
    if difference > 0:
        st.success(
            f"The alternative scenario gives a {difference:,.1f} L lower "
            "illustrative water indicator in this demo."
        )
    elif difference < 0:
        st.warning(
            f"The alternative scenario gives a {abs(difference):,.1f} L higher "
            "illustrative water indicator in this demo."
        )
    else:
        st.info("Both scenarios have the same illustrative water indicator.")

    st.caption(
        "This is not proof of real water savings. Validate with a trained model "
        "and reliable operational data."
    )


# -------------------- TAB 5: ABOUT & VALIDATION --------------------
with tab5:
    st.markdown("### What is EcoXAI 2.0?")
    st.write(
        "EcoXAI 2.0 is a project concept for making water-related resource "
        "estimates in AI data centres easier to understand. It combines a "
        "dashboard demonstration with a proposed explainable-AI and IoT design."
    )

    st.markdown("### Key innovation")
    st.markdown("""
    <div class="info-card">
        <div class="info-title">Estimate → Explain → Investigate</div>
        <p class="info-copy">
        The goal is not only to display a number, but also to explain which
        inputs influenced a prediction and help users decide what to check next.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Current implementation status")
    status = pd.DataFrame({
        "Feature": [
            "Dark ocean dashboard",
            "Interactive operating controls",
            "Example resource calculations",
            "Simulated live variation",
            "Trained prediction model",
            "Genuine SHAP explanations",
            "Physical IoT sensor readings",
            "Verified test metrics"
        ],
        "Status": [
            "Available",
            "Available",
            "Demo only",
            "Optional demo",
            "Not connected in this file",
            "Not connected in this file",
            "Not connected in this file",
            "Not calculated in this file"
        ]
    })
    st.dataframe(status, use_container_width=True, hide_index=True)

    st.markdown("### Planned system flow")
    st.code(
        "Dataset / IoT sensors\n"
        "        ↓\n"
        "Data validation and preparation\n"
        "        ↓\n"
        "Trained resource prediction model\n"
        "        ↓\n"
        "SHAP explanation\n"
        "        ↓\n"
        "Dashboard, alerts and what-if comparison",
        language="text"
    )

    st.markdown("### Important limitations")
    st.markdown("""
    - The displayed values come from illustrative formulas.
    - The changing feed is simulated, not a real-time sensor connection.
    - The driver table is not a genuine SHAP explanation.
    - The carbon figure uses an example emissions factor.
    - No real-world water savings are claimed.
    - A trained model, reliable data, test metrics, and actual sensor integration
      must be added and tested before making deployment claims.
    """)

    st.markdown("### Technology")
    st.write(
        "Python · Streamlit · Pandas · NumPy · Plotly · "
        "Scikit-learn / Random Forest and SHAP (planned integration)"
    )


# -------------------- FOOTER --------------------
st.markdown("""
<div class="footer">
    <div class="footer-name">Developed by SRI BHARATHI VN</div>
    <div>III B.Sc. Computer Science with Cognitive Systems</div>
    <div style="margin-top:8px;">
        EcoXAI 2.0 · Explainable IoT Water Intelligence for AI Data Centres
    </div>
    <div style="margin-top:8px;font-size:0.78rem;">
        Demo interface · Simulated values are not real sensor measurements.
    </div>
</div>
""", unsafe_allow_html=True)
