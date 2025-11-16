# Smart Energy Forecasting System for Steel Manufacturing

[![Domain](https://img.shields.io/badge/Domain-Energy%20Analytics-blue?style=for-the-badge)](https://github.com/Ashish-Sinha07/Smart-Energy-Forecasting-System-for-Steel-Manufacturing)
[![Tech](https://img.shields.io/badge/Tech-Python%20%7C%20ML%20%7C%20DL-green?style=for-the-badge)]()

Welcome — this repository contains an end-to-end AI-powered energy analytics system tailored for real-world steel manufacturing. It includes data cleaning, EDA, feature engineering, time-series splits, modeling (classical ML & deep learning), and example dashboards for visualization and monitoring.

Quick jump:
- Demo / Try it now
- Installation & Quickstart
- Dashboard integration
- Contributing & 👋 contact

Demo
----
Try a quick interactive demo locally:

1. Clone the repo:
```bash
git clone https://github.com/Ashish-Sinha07/Smart-Energy-Forecasting-System-for-Steel-Manufacturing.git
cd Smart-Energy-Forecasting-System-for-Steel-Manufacturing
```

2. Create a venv and install:
```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

3. Run a small example (sample CSV included in data/sample/):
```bash
python scripts/run_quick_forecast.py --data data/sample/sample_energy.csv --model xgboost --output results/quick_forecast.csv
```

4. Launch the dashboard (Streamlit):
```bash
streamlit run src/dashboard/energy_dashboard.py -- --data results/quick_forecast.csv
```

Highlights / Why this repo
--------------------------
- Built for highly dynamic energy consumption in steel manufacturing (reactive power, active power, PF, voltages, currents).
- Reproducible ML pipeline: cleaning → EDA → features → model training → evaluation → serving.
- Multiple model options: XGBoost, RandomForest, GradientBoosting, LSTM/GRU (Keras), Prophet.
- Example integration with dashboards (Streamlit / Tableau) for visualization and decision support.

Table of Contents
-----------------
- About
- Features
- Tech stack
- Quickstart
- Notebooks and Scripts
- Data format
- Dashboard & Deployment
- Contributing
- Contact

About
-----
Industrial energy forecasting helps:
- Optimize electricity usage and reduce costs
- Prevent power factor penalties
- Improve load scheduling and production planning
- Support ESG reporting and decarbonization

This project demonstrates practical patterns for time-series energy forecasting and operational analytics in steel manufacturing.

Features
--------
- Data ingestion pipelines and cleaning utilities
- EDA notebooks with automated visualizations
- Feature engineering helpers (lag features, rolling stats, calendar features)
- Train / validation splitting for time-series
- Model training scripts (classical & DL)
- Model evaluation & backtesting utilities
- Basic inference API and dashboard samples

Tech stack
----------
- Python 3.8+
- Data: pandas, numpy
- Visualization: matplotlib, seaborn, plotly
- Modeling: scikit-learn, xgboost, tensorflow/keras, prophet (optional)
- Serving / UI: streamlit, flask (examples)
- Extras: mlflow (optional) for experiment tracking


Data format
-----------
Place your input CSV(s) in data/ in this format (example header):
timestamp, active_power_kw, reactive_power_kvar, voltage_v, current_a, power_factor, load_id, other_meta...

- timestamp required in ISO format (YYYY-MM-DD HH:MM:SS)
- one row per measurement / aggregated window (e.g., 1min, 15min, 1h)
- missing data handling: notebooks show interpolation and gap detection strategies

Quick example: preview the sample data
```python
import pandas as pd
df = pd.read_csv('data/sample/sample_energy.csv', parse_dates=['timestamp'])
df.set_index('timestamp').head()
```
Each training script writes model artifacts to models/<model-name>/ with timestamps.

Evaluation & Metrics
--------------------
Common metrics included:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- Directional Accuracy (for operational decisions)

Backtesting utilities are in src/evaluate/backtest.py — they calculate rolling-window backtests and aggregate metrics per horizon.

Interactive examples:
- Run full training + backtest:
```bash
python scripts/train_model.py --config configs/xgboost_config.yaml
python scripts/backtest.py --checkpoint models/xgboost/<latest> --data data/production.csv
```

Dashboard & Deployment
----------------------
- Streamlit sample in src/dashboard/energy_dashboard.py — quickly visualize raw data, features, and model predictions.
- Example Flask app for inference included in src/api/ (simple POST /predict).
- Dockerfile & docker-compose.yml (example) available for containerized deployment.

Run Streamlit locally:
```bash
streamlit run src/dashboard/energy_dashboard.py
```

Run the example API server:
```bash
python src/api/server.py --model models/xgboost/<latest>/model.pkl
curl -X POST -H "Content-Type: application/json" -d '{"timestamp":"2025-01-01 00:00:00", ...}' http://localhost:5000/predict
```

Got a dashboard tool (Tableau)?
- Use the output CSV from scripts/run_quick_forecast.py as a datasource in Tableau.
- Or use the Streamlit UI as a light-weight interactive dashboard.

Tips for Interactivity
----------------------
- Use Jupyter widgets to explore feature importance dynamically.
- Use Streamlit sliders to change prediction horizons and see model responses.
- Connect model outputs to rule-based alerts (e.g., predicted PF drop → notify).

Contributing
------------
Contributions are welcome! A few ways to help:
- Open issues for bugs, feature requests, or suggestions
- Add new datasets or notebooks demonstrating different pre-processing
- Improve model baselines and add benchmarking
- Add CI tests for data pipelines and scripts

Suggested workflow:
1. Fork the repo
2. Create a branch topic/my-feature
3. Add tests (if applicable)
4. Open a Pull Request describing your changes

Contact
-------
Maintainer: Ashish Sinha (@Ashish-Sinha07)
- GitHub: https://github.com/Ashish-Sinha07
- For questions or collabs, open an issue or connect on GitHub.
