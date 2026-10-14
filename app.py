        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Operating conditions</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    details = [("Active servers", str(s)), ("Temperature", f"{t:.1f} °C"),
               ("Cooling duration", f"{c:.1f} h/day"), ("Mining activity", "Active" if mining else "Inactive")]
    for col, (label, value) in zip(cols, details):
        with col:
            st.markdown(f'<div class="card"><div class="card-label">{label}</div><div class="card-value">{value}</div><div class="card-note">Simulated input</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="notice"><b>Transparency:</b> This starter uses illustrative equations and demo feature scores. They are not actual model predictions, measurements, validated results, or SHAP explanations.</div>', unsafe_allow_html=True)


if live:
    try:
        st.fragment
        fragment_available = True
    except AttributeError:
        fragment_available = False
    if fragment_available:
        @st.fragment(run_every="3s")
        def live_panel():
            # Small random variation for demonstration; not sensor data.
            s = int(np.clip(servers + random.choice([-2, -1, 0, 1, 2]), 1, 100))
            t = float(np.clip(temperature + random.choice([-0.5, 0, 0.5, 1]), 15, 50))
            c = float(np.clip(cooling_hours + random.choice([-0.5, 0, 0.5]), 1, 24))
            show_dashboard(s, t, c, mining_active, True)
        # function is defined below; Python resolves it when this function runs.
        live_panel()
    else:
        st.info("Your Streamlit version does not support automatic fragments. Showing a simulated snapshot; upgrade Streamlit to enable timed refresh.")
        show_dashboard(servers, temperature, cooling_hours, mining_active, False)
else:
    show_dashboard(servers, temperature, cooling_hours, mining_active, False)

# Supporting tabs
st.markdown("")
tab1, tab2, tab3, tab4 = st.tabs(["What-if lab", "Architecture", "Validation & limits", "Project story"])

with tab1:
    st.markdown('<div class="section-title">Compare two scenarios</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Change inputs and compare the illustrative output. Replace the demo calculation with your trained model before judging.</div>', unsafe_allow_html=True)
    baseline = demo_predict(servers, temperature, cooling_hours, mining_active)
    x, y = st.columns(2)
    with x:
        st.markdown("**Baseline**")
        st.write(f"{servers} servers · {temperature:.1f}°C · {cooling_hours:.1f} cooling hours/day · Mining {'on' if mining_active else 'off'}")
    with y:
        st.markdown("**Alternative conditions**")
        alt_servers = st.slider("Servers", 1, 100, servers, key="alt_servers")
        alt_temp = st.slider("Temperature (°C)", 15.0, 50.0, max(15.0, min(50.0, temperature - 3)), .5, key="alt_temp")
        alt_cooling = st.slider("Cooling duration (h/day)", 1.0, 24.0, max(1.0, min(24.0, cooling_hours - 1)), .5, key="alt_cooling")
        alt_mining = st.toggle("Mining active", value=mining_active, key="alt_mining")
    alternative = demo_predict(alt_servers, alt_temp, alt_cooling, alt_mining)
    comp = pd.DataFrame([
        {"Indicator": "Energy (kWh)", "Baseline": baseline["Energy"], "Alternative": alternative["Energy"]},
        {"Indicator": "Cooling water (L)", "Baseline": baseline["Cooling water"], "Alternative": alternative["Cooling water"]},
        {"Indicator": "Carbon (kg CO₂e)", "Baseline": baseline["Carbon"], "Alternative": alternative["Carbon"]},
    ])
    st.dataframe(comp, use_container_width=True, hide_index=True)
    delta = baseline["Cooling water"] - alternative["Cooling water"]
    st.info(f"Demo difference in cooling-water indicator: {delta:+.2f} L. This is not evidence of real-world savings.")

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
**Current status:** This file is a UI starter. It does not load the trained EcoXAI model yet.

Before presenting validation, add:
- Held-out test-set R² and MAE for each model target.
- Dataset source, row count, features and units.
- Train/test split method and preprocessing details.
- Actual SHAP values from the same trained model.

**Limitations**
- Monitoring values are simulated in this starter.
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
