
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="EcoXAI 2.0",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- THEME ----------------
BG = "#061526"
PANEL = "#0B2340"
CYAN = "#27D8E8"
SKY = "#65BFFF"
AQUA = "#35E0C1"
YELLOW = "#FFD166"
ORANGE = "#FF9F43"
RED = "#FF5D68"
GREEN = "#55D6A4"
WHITE = "#F3F8FF"
MUTED = "#A8BED2"
BORDER = "#19466B"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

.stApp {{
    background: {BG};
    color: {WHITE};
    font-family: Inter, sans-serif;
}}
[data-testid="stHeader"] {{ background: rgba(6,21,38,.9); }}
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg,#0B2038,#071526);
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{ color: {WHITE}; }}
h1,h2,h3,h4,p,label {{ color: {WHITE}; }}
[data-testid="stMetric"] {{
    background: linear-gradient(145deg,{PANEL},#091B30);
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 16px;
}}
[data-testid="stMetricLabel"] p {{ color: {MUTED}!important; }}
[data-testid="stMetricValue"] {{ color: {YELLOW}!important; font-weight:800!important; }}
.stButton>button {{
    background: linear-gradient(100deg,#087FB5,#087F91);
    color: white;
    border: 1px solid {CYAN};
    border-radius: 10px;
    font-weight: 700;
}}
.hero {{
    background:
      linear-gradient(90deg,rgba(3,20,42,.96),rgba(5,45,78,.75),rgba(3,20,42,.45)),
      url("https://images.unsplash.com/photo-1558494949-ef010cbdcc31?auto=format&fit=crop&w=1800&q=85");
    background-size: cover;
    background-position: center;
    border: 1px solid {CYAN};
    border-radius: 18px;
    padding: 32px;
    margin: 8px 0 20px;
}}
.hero-title {{ font-size:2.7rem; font-weight:800; }}
.hero-sub {{ color:{CYAN}; letter-spacing:2px; font-weight:700; }}
.panel {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 18px;
}}
.footer {{
    border-top:1px solid {BORDER};
    margin-top:30px;
    padding:20px;
    text-align:center;
    color:{MUTED};
}}
</style>
""", unsafe_allow_html=True)

# ---------------- CITY DATA ----------------
# Demonstration values only, not real measured city data.
CITY_DATA = {
    "Chennai, India": {"water": 1240, "temperature": 32, "servers": 250},
    "Mumbai, India": {"water": 1680, "temperature": 35, "servers": 280},
    "Hyderabad, India": {"water": 1420, "temperature": 33, "servers": 265},
    "Bengaluru, India": {"water": 1320, "temperature": 29, "servers": 245},
    "Delhi, India": {"water": 1500, "temperature": 34, "servers": 270},
}

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 💧 EcoXAI 2.0")
    st.caption("SAVING WATER. EXPLAINING WHY.")
    st.divider()

    st.markdown("### ⚙️ Live Controls")

    city = st.selectbox(
        "📍 Data Centre City",
        options=list(CITY_DATA.keys()),
        key="city_selector"
    )

    selected = CITY_DATA[city]

    servers = st.slider(
        "🖥️ Number of Servers",
        10, 500, int(selected["servers"]),
        key=f"servers_{city}"
    )

    temperature = st.slider(
        "🌡️ Temperature (°C)",
        15, 50, int(selected["temperature"]),
        key=f"temperature_{city}"
    )

    cooling_hours = st.slider(
        "❄️ Cooling Hours / Day",
        1, 24, 6,
        key=f"cooling_{city}"
    )

    mining_active = st.toggle(
        "⛏️ Crypto Mining",
        value=False,
        key=f"mining_{city}"
    )

    st.divider()
    simulate_live = st.toggle("Simulate Live Readings", value=False)
    st.caption("Demo only — no physical sensors connected.")

    st.button("🔎 Run EcoXAI Analysis", use_container_width=True)

# ---------------- DEMO CALCULATIONS ----------------
# Illustrative formulas, not validated AI predictions.
base_water = selected["water"]

water = (
    base_water
    + (servers - selected["servers"]) * 2.5
    + max(temperature - selected["temperature"], 0) * 22
    + (cooling_hours - 6) * 8
    + (180 if mining_active else 0)
)

energy = (
    servers * 9.5
    + max(temperature - 20, 0) * 3
    + cooling_hours * 4
    + (180 if mining_active else 0)
)

if simulate_live:
    rng = np.random.default_rng()
    water = max(0, water + rng.normal(0, max(water * .02, 1)))
    energy = max(0, energy + rng.normal(0, max(energy * .02, 1)))

water = max(0, water)
energy = max(0, energy)
carbon = energy * 0.71

if temperature >= 40 or water >= 1800:
    risk = "Critical"
elif temperature >= 35 or water >= 1500:
    risk = "High"
elif temperature >= 30 or water >= 1200:
    risk = "Medium"
else:
    risk = "Low"

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
  <div class="hero-title">💧 EcoXAI <span style="color:#27D8E8">2.0</span></div>
  <p>Saving Water. Explaining Why. Intelligently.</p>
  <div class="hero-sub">AI • IoT • SUSTAINABLE DATA CENTRES</div>
</div>
""", unsafe_allow_html=True)

st.caption(
    f"Selected city: {city} | Demo values only, not live city measurements."
)

# ---------------- KPI CARDS ----------------
st.markdown("### 📊 Resource Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("💧 Water Usage", f"{water:,.0f} L")
c2.metric("⚡ Energy", f"{energy:,.0f} kWh")
c3.metric("🌍 Carbon", f"{carbon:,.0f} kg CO₂e")
c4.metric("🚦 Scenario Risk", risk)

if temperature >= 35:
    st.warning(
        f"🌡️ Selected scenario temperature is {temperature}°C. "
        "Review approved cooling limits for any real facility."
    )
else:
    st.success(
        f"Selected scenario temperature: {temperature}°C. "
        "This is not a real-world safety certification."
    )

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Explorer",
    "📈 Visual Findings",
    "🧠 XAI Explanations",
    "🔮 Live Prediction"
])

# ---------------- TAB 1 ----------------
with tab1:
    left, right = st.columns([1.4, 1])

    with left:
        st.markdown("### 💧 Water Indicator vs Temperature")

        temps = list(range(15, 51))
        curve = [
            max(0,
                base_water
                + (servers - selected["servers"]) * 2.5
                + max(t - selected["temperature"], 0) * 22
                + (cooling_hours - 6) * 8
                + (180 if mining_active else 0)
            )
            for t in temps
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=temps,
            y=curve,
            mode="lines",
            line=dict(color=CYAN, width=3),
            fill="tozeroy",
            fillcolor="rgba(39,216,232,.12)",
            name="Example water indicator"
        ))

        fig.add_vline(
            x=temperature,
            line_dash="dash",
            line_color=YELLOW,
            annotation_text="Selected temperature"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=PANEL,
            font=dict(color=WHITE),
            xaxis_title="Temperature (°C)",
            yaxis_title="Illustrative water indicator (L/day)",
            height=360
        )
        fig.update_xaxes(gridcolor=BORDER)
        fig.update_yaxes(gridcolor=BORDER)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### 📍 Selected Data Centre")
        st.markdown(f"""
        <div class="panel">
          <h3 style="color:{CYAN}">{city}</h3>
          <p style="color:{MUTED}">Demo location</p>
          <hr>
          <p style="color:{MUTED}">Active servers</p>
          <h2 style="color:{SKY}">{servers}</h2>
          <p style="color:{MUTED}">Temperature</p>
          <h2 style="color:{YELLOW}">{temperature}°C</h2>
          <p style="color:{MUTED}">Cooling hours/day</p>
          <h2 style="color:{AQUA}">{cooling_hours}</h2>
          <p style="color:{MUTED}">Crypto mining</p>
          <h3>{"ON" if mining_active else "OFF"}</h3>
        </div>
        """, unsafe_allow_html=True)

# ---------------- TAB 2 ----------------
with tab2:
    st.markdown("### 🏙️ City Comparison")

    city_names = list(CITY_DATA.keys())
    city_values = [
        water if name == city else CITY_DATA[name]["water"]
        for name in city_names
    ]

    bar = go.Figure(go.Bar(
        x=[name.split(",")[0] for name in city_names],
        y=city_values,
        marker_color=[
            YELLOW if name == city else color
            for name, color in zip(
                city_names, [SKY, AQUA, "#9D8CFF", "#FF8BA7", ORANGE]
            )
        ],
        text=[f"{value:,.0f} L" for value in city_values],
        textposition="outside"
    ))

    bar.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL,
        font=dict(color=WHITE),
        yaxis_title="Illustrative water indicator (L/day)",
        height=380
    )
    bar.update_xaxes(gridcolor=BORDER)
    bar.update_yaxes(gridcolor=BORDER)
    st.plotly_chart(bar, use_container_width=True)

    st.markdown("### 🚦 Example Risk Distribution")

    pie = go.Figure(go.Pie(
        labels=["Low", "Medium", "High", "Critical"],
        values=[52, 28, 16, 4],
        hole=0.62,
        marker=dict(colors=[GREEN, YELLOW, ORANGE, RED]),
        textinfo="label+percent"
    ))

    pie.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=WHITE),
        height=350
    )
    st.plotly_chart(pie, use_container_width=True)

    st.caption("The risk distribution is illustrative sample data.")

# ---------------- TAB 3 ----------------
with tab3:
    st.markdown("### 🧠 Explainable AI")

    st.write(
        "This chart is a visual placeholder showing how input explanations "
        "could appear. These are not genuine SHAP values."
    )

    factors = pd.DataFrame({
        "Input factor": [
            "Active servers",
            "Temperature",
            "Cooling hours",
            "Mining workload"
        ],
        "Example contribution (%)": [30, 25, 25, 20]
    })

    xai = go.Figure(go.Bar(
        x=factors["Example contribution (%)"],
        y=factors["Input factor"],
        orientation="h",
        marker_color=[SKY, AQUA, YELLOW, ORANGE],
        text=[f"{v}%" for v in factors["Example contribution (%)"]],
        textposition="outside"
    ))

    xai.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL,
        font=dict(color=WHITE),
        xaxis_title="Illustrative contribution (%)",
        height=350
    )
    xai.update_xaxes(gridcolor=BORDER, range=[0, 40])
    xai.update_yaxes(gridcolor=BORDER)
    st.plotly_chart(xai, use_container_width=True)

    st.info(
        "Connect the trained model and SHAP calculations to display "
        "real model explanations."
    )

# ---------------- TAB 4 ----------------
with tab4:
    st.markdown("### 🔮 What-If Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Current Scenario")
        st.write(f"City: {city}")
        st.write(f"Servers: {servers}")
        st.write(f"Temperature: {temperature}°C")
        st.write(f"Cooling hours: {cooling_hours}")
        st.write(f"Mining: {'ON' if mining_active else 'OFF'}")

    with col2:
        st.markdown("#### Alternative Scenario")

        alt_servers = st.slider(
            "Alternative servers", 10, 500,
            max(10, servers - 25), key="alt_servers"
        )
        alt_temp = st.slider(
            "Alternative temperature", 15, 50,
            max(15, temperature - 3), key="alt_temp"
        )
        alt_cooling = st.slider(
            "Alternative cooling hours", 1, 24,
            max(1, cooling_hours - 1), key="alt_cooling"
        )
        alt_mining = st.toggle(
            "Alternative mining workload", key="alt_mining"
        )

    alt_water = (
        base_water
        + (alt_servers - selected["servers"]) * 2.5
        + max(alt_temp - selected["temperature"], 0) * 22
        + (alt_cooling - 6) * 8
        + (180 if alt_mining else 0)
    )

    alt_energy = (
        alt_servers * 9.5
        + max(alt_temp - 20, 0) * 3
        + alt_cooling * 4
        + (180 if alt_mining else 0)
    )
    alt_carbon = alt_energy * 0.71

    st.dataframe(
        pd.DataFrame({
            "Indicator": ["Water (L/day)", "Energy (kWh/day)", "Carbon (kg CO₂e/day)"],
            "Current": [f"{water:,.0f}", f"{energy:,.0f}", f"{carbon:,.0f}"],
            "Alternative": [f"{alt_water:,.0f}", f"{alt_energy:,.0f}", f"{alt_carbon:,.0f}"]
        }),
        use_container_width=True,
        hide_index=True
    )

    difference = water - alt_water
    if difference > 0:
        st.success(
            f"Alternative is {difference:,.0f} L/day lower in this demo calculation."
        )
    elif difference < 0:
        st.warning(
            f"Alternative is {abs(difference):,.0f} L/day higher in this demo calculation."
        )
    else:
        st.info("Both scenarios have the same demo water indicator.")

    st.caption("These formulas are illustrative, not validated model predictions.")

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
  <div style="color:#35E0C1;font-weight:800;font-size:1rem;">
    Developed by SRI BHARATHI VN
  </div>
  <div>III B.Sc. Computer Science with Cognitive Systems</div>
  <div>EcoXAI 2.0 · Saving Water. Explaining Why. Intelligently.</div>
</div>
""", unsafe_allow_html=True)
