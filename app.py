
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# =========================================================
# ECOXAI 2.0 — EXPLAINABLE IoT WATER INTELLIGENCE
# =========================================================

st.set_page_config(
    page_title="EcoXAI 2.0 | Water Intelligence",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------- THEME -------------------------

NAVY = "#061426"
PANEL = "#0D223B"
PANEL2 = "#102B49"
CYAN = "#28D7F2"
AQUA = "#16D9C4"
SKY = "#77C9FF"
YELLOW = "#FFD166"
ORANGE = "#FF9F43"
RED = "#FF5C68"
GREEN = "#52D6A5"
WHITE = "#F4F8FF"
MUTED = "#A7BDD1"
BORDER = "#17496B"

# ------------------------- CSS ---------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #061426;
    color: #F4F8FF;
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(6,20,38,0.96);
}

[data-testid="stSidebar"] {
    background: #081B30;
    border-right: 1px solid #17496B;
}

[data-testid="stSidebar"] * {
    color: #F4F8FF;
}

h1, h2, h3, h4, p, label {
    color: #F4F8FF;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #102B49, #0B1D33);
    border: 1px solid #17618A;
    border-radius: 14px;
    padding: 18px 18px 12px 18px;
    min-height: 125px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.12);
}

[data-testid="stMetricLabel"] p {
    color: #BBD5E9 !important;
    font-size: 0.90rem !important;
}

[data-testid="stMetricValue"] {
    color: #FFD166 !important;
    font-weight: 800 !important;
}

[data-testid="stMetricDelta"] {
    color: #28D7F2 !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #17496B;
}

.stTabs [data-baseweb="tab"] {
    background: #0D223B;
    border: 1px solid #17496B;
    border-radius: 8px 8px 0 0;
    color: #DDEBFA;
    padding: 12px 18px;
}

.stTabs [aria-selected="true"] {
    background: #10466A !important;
    color: white !important;
    border-bottom: 3px solid #28D7F2 !important;
}

.stButton > button {
    background: linear-gradient(100deg, #087EB8, #087F8C);
    color: white;
    border: 1px solid #28D7F2;
    border-radius: 9px;
    font-weight: 700;
    min-height: 43px;
}

.stButton > button:hover {
    border-color: #FFD166;
    color: white;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #102B49;
    border-color: #17496B;
}

.hero {
    border: 1px solid #28D7F2;
    border-radius: 18px;
    min-height: 190px;
    padding: 34px 38px;
    margin-bottom: 20px;
    background:
      linear-gradient(90deg, rgba(4,20,43,0.96) 0%,
      rgba(5,39,72,0.78) 52%,
      rgba(5,35,62,0.45) 100%),
      url("https://images.unsplash.com/photo-1518837695005-2083093ee35b?auto=format&fit=crop&w=2000&q=85");
    background-size: cover;
    background-position: center;
    box-shadow: 0 8px 28px rgba(0,180,220,0.10);
}

.hero-title {
    font-size: 2.65rem;
    font-weight: 800;
    letter-spacing: 1px;
    color: white;
    margin: 0;
}

.hero-title span {
    color: #28D7F2;
}

.hero-subtitle {
    color: #D6EAF8;
    font-size: 1.05rem;
    margin-top: 8px;
}

.hero-tag {
    color: #82E8E5;
    font-size: 0.78rem;
    letter-spacing: 2px;
    font-weight: 700;
    margin-bottom: 10px;
}

.section-title {
    color: #F4F8FF;
    font-size: 1.2rem;
    font-weight: 800;
    margin: 8px 0 10px 0;
}

.panel {
    background: #0D223B;
    border: 1px solid #17496B;
    border-radius: 13px;
    padding: 17px;
    margin-bottom: 10px;
}

.panel-title {
    color: #77C9FF;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 8px;
}

.muted {
    color: #A7BDD1;
    font-size: 0.87rem;
    line-height: 1.55;
}

.alert {
    background: #38271D;
    border: 1px solid #80522B;
    border-left: 5px solid #FF9F43;
    border-radius: 9px;
    padding: 13px 16px;
    color: #FFE0B5;
    margin: 8px 0;
}

.success {
    background: #102F32;
    border: 1px solid #24665C;
    border-left: 5px solid #52D6A5;
    border-radius: 9px;
    padding: 13px 16px;
    color: #C8F8E5;
    margin: 8px 0;
}

.footer {
    border-top: 1px solid #17496B;
    text-align: center;
    margin-top: 28px;
    padding: 20px 5px 8px 5px;
    color: #A7BDD1;
    font-size: 0.88rem;
}

.footer strong {
    color: #28D7F2;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #17496B;
    border-radius: 10px;
}

hr {
    border-color: #17496B;
}
</style>
""", unsafe_allow_html=True)


# ---------------------- SAMPLE CITY DATA ----------------------
# These are demonstration values, not measured city statistics.

city_data = {
    "Chennai": {
        "country": "India",
        "lat": 13.0827,
        "lon": 80.2707,
        "water": 1240,
        "energy": 3980,
        "carbon": 2830,
        "risk": "Medium"
    },
    "Mumbai": {
        "country": "India",
        "lat": 19.0760,
        "lon": 72.8777,
        "water": 1680,
        "energy": 4320,
        "carbon": 3067,
        "risk": "High"
    },
    "Hyderabad": {
        "country": "India",
        "lat": 17.3850,
        "lon": 78.4867,
        "water": 1420,
        "energy": 4150,
        "carbon": 2947,
        "risk": "Medium"
    },
    "Bengaluru": {
        "country": "India",
        "lat": 12.9716,
        "lon": 77.5946,
        "water": 1320,
        "energy": 4050,
        "carbon": 2876,
        "risk": "Low"
    }
}


# ------------------------- SIDEBAR -------------------------

with st.sidebar:
    st.markdown(
        "<h2 style='margin-bottom:0;'>💧 EcoXAI <span style='color:#28D7F2'>2.0</span></h2>",
        unsafe_allow_html=True
    )
    st.caption("SAVING WATER. EXPLAINING WHY.")
    st.divider()

    st.markdown("### ⚙️ Live Controls")

    servers = st.slider(
        "Number of Servers", 10, 500, 250, 10
    )

    temperature = st.slider(
        "Temperature (°C)", 15, 50, 32
    )

    cooling_hours = st.slider(
        "Cooling Hours / Day", 1, 24, 6
    )

    mining_active = st.toggle(
        "Crypto Mining", value=False
    )

    st.divider()

    st.markdown("### 📍 Data Centre Location")

    selected_city = st.selectbox(
        "Choose a city",
        list(city_data.keys()),
        index=0
    )

    st.divider()

    simulate = st.toggle(
        "Simulate changing readings", value=False
    )

    st.caption("Demo mode — no physical sensor is connected.")

    if st.button("🔎 Run EcoXAI Prediction", use_container_width=True):
        st.session_state["last_run"] = datetime.now().strftime("%H:%M:%S")
        st.toast("Demo indicators updated!")


# ---------------------- DEMO CALCULATIONS ----------------------
# Replace these formulas with your actual trained model when connected.

city = city_data[selected_city]

water = (
    150
    + servers * 3.0
    + max(temperature - 20, 0) * 18
    + cooling_hours * 12
    + (150 if mining_active else 0)
)

energy = (
    servers * 9.5
    + max(temperature - 20, 0) * 3
    + cooling_hours * 4
    + (180 if mining_active else 0)
)

carbon = energy * 0.71

if simulate:
    rng = np.random.default_rng()
    water = max(0, water + rng.normal(0, water * 0.025))
    energy = max(0, energy + rng.normal(0, energy * 0.02))
    carbon = energy * 0.71

# ------------------------- HERO -------------------------

st.markdown(f"""
<div class="hero">
    <div class="hero-tag">AI · IoT · SUSTAINABLE DATA CENTRES</div>
    <div class="hero-title">💧 EcoXAI <span>2.0</span></div>
    <div class="hero-subtitle">
        Saving Water. Explaining Why. Intelligently.
    </div>
    <div style="color:#A9DDF5;margin-top:12px;font-size:0.9rem;">
        Smarter Data Centres · A Greener Tomorrow
    </div>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Dashboard demonstration · Example values are simulated and are not "
    "verified measurements from real data centres."
)


# ---------------------- KPI CARDS ----------------------

st.markdown(
    '<div class="section-title">Resource Overview</div>',
    unsafe_allow_html=True
)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("💧 Water Usage", f"{water:,.0f} L")
    st.caption("Illustrative daily indicator")

with k2:
    st.metric("⚡ Energy Consumption", f"{energy:,.0f} kWh")
    st.caption("Illustrative daily indicator")

with k3:
    st.metric("🌍 Carbon Emission", f"{carbon:,.0f} kg CO₂e")
    st.caption("Example emissions factor")

with k4:
    performance = max(70, min(99, 98 - (temperature - 25) * 0.25))
    st.metric("◉ Performance", f"{performance:.1f}%")
    st.caption("Illustrative score")


# ---------------------- LOCATION + MAP ----------------------

st.markdown(
    '<div class="section-title">📍 Data Centre Locations</div>',
    unsafe_allow_html=True
)

map_col, city_col = st.columns([1.25, 1])

with map_col:
    map_df = pd.DataFrame([
        {
            "City": name,
            "Latitude": values["lat"],
            "Longitude": values["lon"],
            "Water Indicator (L/day)": values["water"],
            "Selected": name == selected_city
        }
        for name, values in city_data.items()
    ])

    fig_map = px.scatter_geo(
        map_df,
        lat="Latitude",
        lon="Longitude",
        hover_name="City",
        hover_data={
            "Water Indicator (L/day)": True,
            "Latitude": False,
            "Longitude": False,
            "Selected": False
        },
        color="Selected",
        color_discrete_map={
            True: "#FFD166",
            False: "#28D7F2"
        },
        size="Water Indicator (L/day)",
        size_max=18
    )

    fig_map.update_geos(
        scope="asia",
        projection_type="natural earth",
        showland=True,
        landcolor="#102B49",
        showocean=True,
        oceancolor="#061426",
        showcountries=True,
        countrycolor="#31516B",
        showcoastlines=True,
        coastlinecolor="#28D7F2",
        lataxis_range=[4, 38],
        lonaxis_range=[65, 100],
        bgcolor="#0D223B"
    )

    fig_map.update_layout(
        paper_bgcolor="#0D223B",
        plot_bgcolor="#0D223B",
        font=dict(color="#F4F8FF"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=320,
        legend_title_text=""
    )

    st.plotly_chart(fig_map, use_container_width=True)

with city_col:
    st.markdown(
        f"<div class='panel'>"
        f"<div class='panel-title'>📍 {selected_city}, India</div>"
        f"<div class='muted'>Selected sample location</div>"
        f"<hr>"
        f"<div style='font-size:1.8rem;color:#FFD166;font-weight:800;'>"
        f"{city['water']:,} L</div>"
        f"<div class='muted'>Example water indicator / day</div>"
        f"<br><div class='panel-title'>Location status</div>"
        f"<div style='color:#77C9FF;font-size:1.05rem;font-weight:700;'>"
        f"{city['risk']} risk (demo)</div>"
        f"<p class='muted'>City figures are illustrative comparison values. "
        f"They are not actual city measurements.</p>"
        f"</div>",
        unsafe_allow_html=True
    )


# ------------------------- ALERT -------------------------

st.markdown(
    '<div class="section-title">🚦 What should you check first?</div>',
    unsafe_allow_html=True
)

if temperature >= 35:
    st.markdown(
        f"<div class='alert'>🌡️ <b>Temperature needs attention</b><br>"
        f"The selected input is {temperature}°C. Check cooling conditions "
        f"against the facility's approved operating limits.</div>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f"<div class='success'>✓ <b>No demo temperature alert</b><br>"
        f"The selected input is {temperature}°C. This does not confirm "
        f"that a real data centre is safe.</div>",
        unsafe_allow_html=True
    )

if mining_active:
    st.markdown(
        "<div class='alert'>⚙️ Crypto-mining is enabled in this scenario. "
        "Its real resource impact depends on hardware, workload, and "
        "cooling conditions.</div>",
        unsafe_allow_html=True
    )


# ------------------------- TABS -------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Explorer",
    "📈 Visual Findings",
    "🧠 XAI Explanations",
    "🔮 Live Prediction"
])


# ---------------------- TAB 1: DATA EXPLORER ----------------------

with tab1:
    st.markdown("### 💧 Water Usage vs Temperature")

    temps = list(range(15, 51))
    water_curve = [
        150
        + servers * 3.0
        + max(t - 20, 0) * 18
        + cooling_hours * 12
        + (150 if mining_active else 0)
        for t in temps
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=temps,
        y=water_curve,
        mode="lines",
        name="Example water indicator",
        line=dict(color="#28D7F2", width=3),
        fill="tozeroy",
        fillcolor="rgba(40,215,242,0.10)"
    ))

    fig.add_vline(
        x=temperature,
        line_dash="dash",
        line_color="#FFD166",
        annotation_text="Selected temperature",
        annotation_font_color="#FFD166"
    )

    fig.update_layout(
        paper_bgcolor="#0D223B",
        plot_bgcolor="#0D223B",
        font=dict(color="#F4F8FF"),
        xaxis_title="Temperature (°C)",
        yaxis_title="Illustrative water indicator (L/day)",
        margin=dict(l=10, r=10, t=25, b=10),
        height=400
    )
    fig.update_xaxes(gridcolor="#17496B")
    fig.update_yaxes(gridcolor="#17496B")

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "This graph demonstrates how the example formula changes. "
        "It is not a validated water-consumption prediction."
    )


# ---------------------- TAB 2: VISUAL FINDINGS ----------------------

with tab2:
    left, right = st.columns([1.2, 0.8])

    with left:
        st.markdown("### 🏙️ City Comparison")

        city_df = pd.DataFrame([
            {
                "City": name,
                "Water (L/day)": values["water"]
            }
            for name, values in city_data.items()
        ])

        city_fig = px.bar(
            city_df,
            x="City",
            y="Water (L/day)",
            color="City",
            text="Water (L/day)",
            color_discrete_sequence=[
                "#28D7F2", "#16D9C4", "#77C9FF", "#FFD166"
            ]
        )

        city_fig.update_traces(textposition="outside")
        city_fig.update_layout(
            paper_bgcolor="#0D223B",
            plot_bgcolor="#0D223B",
            font=dict(color="#F4F8FF"),
            showlegend=False,
            yaxis_title="Illustrative water indicator (L/day)",
            xaxis_title="Sample city",
            margin=dict(l=10, r=10, t=25, b=10),
            height=380
        )
        city_fig.update_xaxes(gridcolor="#17496B")
        city_fig.update_yaxes(gridcolor="#17496B")

        st.plotly_chart(city_fig, use_container_width=True)

    with right:
        st.markdown("### 🚦 Demo Risk Distribution")

        risk_fig = go.Figure(data=[go.Pie(
            labels=["Low", "Medium", "High", "Critical"],
            values=[52, 28, 16, 4],
            hole=0.62,
            marker=dict(
                colors=["#16D9C4", "#FFD166", "#FF9F43", "#FF5C68"]
            ),
            textinfo="label+percent"
        )])

        risk_fig.update_layout(
            paper_bgcolor="#0D223B",
            plot_bgcolor="#0D223B",
            font=dict(color="#F4F8FF"),
            showlegend=False,
            margin=dict(l=5, r=5, t=25, b=5),
            height=380
        )

        st.plotly_chart(risk_fig, use_container_width=True)

    st.caption(
        "City comparison and risk distribution are illustrative demo values, "
        "not findings from a verified live monitoring network."
    )


# ---------------------- TAB 3: XAI EXPLANATIONS ----------------------

with tab3:
    st.markdown("### 🧠 What may influence the water indicator?")

    st.write(
        "The chart below shows the relative weights used by this demo formula. "
        "It is not a real SHAP explanation."
    )

    factors = pd.DataFrame({
        "Factor": [
            "Active servers",
            "Temperature",
            "Cooling hours",
            "Mining workload"
        ],
        "Illustrative contribution": [
            servers * 3.0,
            max(temperature - 20, 0) * 18,
            cooling_hours * 12,
            150 if mining_active else 0
        ]
    })

    factors = factors.sort_values(
        "Illustrative contribution", ascending=True
    )

    xai_fig = px.bar(
        factors,
        x="Illustrative contribution",
        y="Factor",
        orientation="h",
        text="Illustrative contribution",
        color="Factor",
        color_discrete_sequence=[
            "#28D7F2", "#16D9C4", "#77C9FF", "#FFD166"
        ]
    )

    xai_fig.update_layout(
        paper_bgcolor="#0D223B",
        plot_bgcolor="#0D223B",
        font=dict(color="#F4F8FF"),
        showlegend=False,
        xaxis_title="Demo formula contribution (not SHAP)",
        yaxis_title="",
        margin=dict(l=10, r=10, t=20, b=10),
        height=360
    )
    xai_fig.update_xaxes(gridcolor="#17496B")
    xai_fig.update_yaxes(gridcolor="#17496B")

    st.plotly_chart(xai_fig, use_container_width=True)

    st.markdown(
        "<div class='panel'>"
        "<div class='panel-title'>Why Explainable AI?</div>"
        "<div class='muted'>"
        "A prediction is more useful when users can understand which inputs "
        "influenced it. The complete EcoXAI 2.0 system should connect the "
        "trained model and calculate genuine SHAP values."
        "</div></div>",
        unsafe_allow_html=True
    )

    st.warning(
        "Real SHAP explanations are not connected in this version. "
        "Do not describe this chart as actual model explainability."
    )


# ---------------------- TAB 4: LIVE PREDICTION ----------------------

with tab4:
    st.markdown("### 🔮 Explore an alternative scenario")

    st.write(
        "Change the values below to compare an alternative scenario with "
        "the current inputs."
    )

    a, b = st.columns(2)

    with a:
        st.markdown("#### Current scenario")
        st.write(f"**City:** {selected_city}")
        st.write(f"**Servers:** {servers}")
        st.write(f"**Temperature:** {temperature}°C")
        st.write(f"**Cooling:** {cooling_hours} hours/day")
        st.write(f"**Mining:** {'On' if mining_active else 'Off'}")

    with b:
        st.markdown("#### Alternative scenario")

        alt_servers = st.slider(
            "Alternative servers", 10, 500, servers, 10
        )
        alt_temp = st.slider(
            "Alternative temperature (°C)", 15, 50, temperature
        )
        alt_cooling = st.slider(
            "Alternative cooling hours/day", 1, 24, cooling_hours
        )
        alt_mining = st.toggle(
            "Alternative mining workload", value=mining_active
        )

    alt_water = (
        150
        + alt_servers * 3.0
        + max(alt_temp - 20, 0) * 18
        + alt_cooling * 12
        + (150 if alt_mining else 0)
    )

    alt_energy = (
        alt_servers * 9.5
        + max(alt_temp - 20, 0) * 3
        + alt_cooling * 4
        + (180 if alt_mining else 0)
    )

    alt_carbon = alt_energy * 0.71

    compare_df = pd.DataFrame({
        "Indicator": ["Water", "Energy", "Carbon"],
        "Current": [water, energy, carbon],
        "Alternative": [alt_water, alt_energy, alt_carbon]
    })

    compare_fig = go.Figure()

    compare_fig.add_trace(go.Bar(
        name="Current",
        x=compare_df["Indicator"],
        y=compare_df["Current"],
        marker_color="#28D7F2"
    ))

    compare_fig.add_trace(go.Bar(
        name="Alternative",
        x=compare_df["Indicator"],
        y=compare_df["Alternative"],
        marker_color="#FFD166"
    ))

    compare_fig.update_layout(
        barmode="group",
        paper_bgcolor="#0D223B",
        plot_bgcolor="#0D223B",
        font=dict(color="#F4F8FF"),
        yaxis_title="Illustrative value (different units)",
        margin=dict(l=10, r=10, t=25, b=10),
        height=370
    )
    compare_fig.update_xaxes(gridcolor="#17496B")
    compare_fig.update_yaxes(gridcolor="#17496B")

    st.plotly_chart(compare_fig, use_container_width=True)

    delta = water - alt_water

    if delta > 0:
        st.success(
            f"The alternative has a {delta:,.1f} L lower demo water indicator."
        )
    elif delta < 0:
        st.warning(
            f"The alternative has a {abs(delta):,.1f} L higher demo water indicator."
        )
    else:
        st.info("Both scenarios have the same demo water indicator.")

    st.caption(
        "This comparison uses illustrative formulas. It does not prove actual "
        "water savings or predict a real facility's performance."
    )


# ------------------------- FOOTER -------------------------

st.markdown("""
<div class="footer">
    <div><strong>Developed by SRI BHARATHI VN</strong></div>
    <div>III B.Sc. Computer Science with Cognitive Systems</div>
    <div style="margin-top:8px;">
        EcoXAI 2.0 · Explainable IoT Water Intelligence for AI Data Centres
    </div>
    <div style="margin-top:8px;font-size:0.76rem;">
        Saving Water. Explaining Why. Intelligently.
    </div>
</div>
""", unsafe_allow_html=True)
