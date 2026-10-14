# EcoXAI-2.0
EcoXAI 2.0 makes AI data centres smarter and more sustainable by predicting water usage, identifying the factors behind resource consumption, and helping users understand where water can be saved through Explainable AI (XAI) and IoT-based monitoring.

EcoXAI 2.0 💧
Understand the environmental impact of computing
EcoXAI 2.0 is a sustainability project that helps people explore how data-centre and mining activity may affect energy use, cooling-water demand, and carbon emissions.
The goal is simple: instead of showing a number alone, EcoXAI 2.0 aims to help users understand which operating conditions may need attention.
Project status: This version includes a dashboard demonstration. The live-monitoring feed and example calculations in the starter UI are simulated. They are not readings from physical sensors or validated model predictions. The real trained model and SHAP explanations must be connected and tested before those features are described as fully operational.

1. The problem
Data centres run the computers and services behind many digital tools. Some computing workloads, including cryptocurrency mining, can use substantial electricity and require cooling.
This creates important questions:
- How much energy might a workload use?
- What cooling-water demand might be associated with operating conditions?
- What factors are linked to higher resource-use estimates?
- What should an operator investigate first?
A dashboard that shows numbers without explaining them may be difficult to act on. EcoXAI 2.0 focuses on making resource-impact information easier to understand.
2. Our proposed solution
EcoXAI 2.0 brings together resource estimates, explanations, and simple decision-support features in one dashboard.
Main features
- Resource overview: Shows energy, cooling-water, carbon, and performance indicators.
- Monitoring demonstration: Displays changing simulated inputs so users can see how a monitoring screen could work.
- “Fix First” guidance: Gives a simple suggestion when an example operating threshold is crossed.
- Explainability: The intended full system uses SHAP to explain which model inputs contributed to a prediction.
- What-if comparison: Lets users compare a baseline scenario with changed operating conditions.
- System architecture: Explains how data, a machine-learning model, explanations, and recommendations fit together.
- Validation and limitations: Provides a place to report genuine model test results and clearly describe what is not yet implemented.
3. What makes EcoXAI 2.0 different?
EcoXAI 2.0 is designed to go beyond displaying a prediction. It aims to explain why a model produced that prediction and help the user decide what to investigate next.
Its longer-term design can accept data from IoT sensors, such as temperature and power sensors, when suitable hardware and data connections are implemented.
4. How it works
1. Collect inputs: Start with a dataset or verified operating readings.
2. Prepare the data: Check that the values, units, and input fields are correct.
3. Estimate resource impact: Use a trained machine-learning model to predict the supported indicators.
4. Explain the estimate: Use SHAP with the trained model to show which inputs influenced the result.
5. Support a decision: Present understandable alerts, suggestions, and what-if comparisons.
6. Show the results: Display the information in the EcoXAI 2.0 dashboard.
Important: The supplied starter interface currently uses illustrative formulas and demo feature scores. These must be replaced with the real trained model outputs and real SHAP values before they are presented as actual AI results.
5. Technology used
- Python: Main programming language.
- Streamlit: Builds the interactive web dashboard.
- Pandas and NumPy: Organise and process data.
- Plotly: Creates interactive charts.
- Scikit-learn / Random Forest: Intended prediction model, once connected to the saved trained model.
- SHAP: Intended method for explaining predictions, once connected to the trained model.
- IoT sensors and a microcontroller: Possible future extension for real sensor readings.
6. How to run the dashboard starter
Install the packages
pip install streamlit numpy pandas plotly
Start the app
Save the Python file as ecoxai_streamlit_starter.py, open a terminal in that folder, and run:
streamlit run ecoxai_streamlit_starter.py
The dashboard should open in your browser.
7. Current status
Feature	Status in the starter interface
Sustainability dashboard	UI demonstration available
Simulated monitoring feed	Demonstration only
Example alerts and suggestions	Rule-based demonstration
What-if comparison	Uses illustrative calculations
Random Forest predictions	Must be connected to the actual trained model
SHAP explanations	Must be connected to actual SHAP output
Test metrics (R² and MAE)	Must be calculated on a held-out test set and added
Physical IoT sensor integration	Future extension unless already built and tested


8. Validation plan
Before presenting EcoXAI 2.0 as a validated predictive system:
1. Evaluate the saved model on data that was not used for training.
2. Report the actual R² and MAE for each supported prediction target.
3. Confirm that the model inputs match the dataset features and units.
4. Generate SHAP explanations from the same trained model.
5. Test alerts and what-if comparisons with known example cases.
6. Clearly label simulated inputs and disclose dataset limitations.
Do not add performance numbers unless they come from the actual evaluation.
9. Limitations
- Simulated readings are not real sensor measurements.
- Example calculations are not evidence of real energy or water savings.
- A model trained on one dataset may not work equally well at every real data centre.
- Carbon estimates depend on the electricity-emissions factor used.
- Real-time IoT monitoring requires sensor hardware, data transfer, and testing.
10. Future improvements
- Connect and test the trained Random Forest model.
- Connect genuine SHAP explanations.
- Integrate suitable temperature and power sensors.
- Test with reliable operational data.
- Improve recommendations using validated operating limits and domain guidance.
11. One-minute project explanation
EcoXAI 2.0 is an explainable AI framework for sustainable computing. It aims to help users understand the possible energy, cooling-water, and carbon impact of data-centre and mining workloads. Its key idea is to combine resource prediction with explanations, so users can see which inputs influenced an estimate and what conditions may deserve attention. The dashboard also demonstrates monitoring, alerts, and what-if comparisons. The current interface uses simulated inputs; real model outputs and physical IoT integration must be validated before they are described as deployed.
Project principle
Predict clearly. Explain the result. Help people decide what to check next.
EcoXAI 2.0 is a decision-support concept, not a replacement for verified facility measurements or professional operating procedures.
Developed by: SRI BHARATHI VN
Programme: III B.Sc. Computer Science with Cognitive Syste
