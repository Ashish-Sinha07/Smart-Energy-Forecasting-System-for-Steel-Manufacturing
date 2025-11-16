# Smart Energy Forecasting System for Steel Manufacturing

[![Domain](https://img.shields.io/badge/Domain-Energy%20Analytics-blue?style=for-the-badge)](https://github.com/Ashish-Sinha07/Smart-Energy-Forecasting-System-for-Steel-Manufacturing)
[![Tech](https://img.shields.io/badge/Tech-Python%20%7C%20ML%20%7C%20DL-green?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Welcome — this repository contains an end-to-end AI-powered energy analytics system tailored for real-world steel manufacturing. It includes data cleaning, EDA, feature engineering, time-series splits, modeling (classical ML & deep learning), and example dashboards for visualization and monitoring.

Quick jump:
- Demo / Try it now
- Installation & Quickstart
- Notebooks & scripts
- Models & evaluation
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
- Models & Model Zoo
- Data format
- Evaluation & Metrics
- Dashboard & Deployment
- Contributing
- License & Contact

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

Notebooks & Scripts (Interactive)
--------------------------------
- notebooks/01_EDA.ipynb — Exploratory Data Analysis with visualizations and checks
- notebooks/02_feature_engineering.ipynb — Create lag/rolling and calendar features
- notebooks/03_modeling_baseline.ipynb — Baseline classical ML models
- notebooks/04_deep_learning.ipynb — LSTM/GRU experiments
- scripts/run_quick_forecast.py — Minimal CLI to run a quick forecast
- scripts/train_model.py — Full training script with config file support
- src/dashboard/energy_dashboard.py — Streamlit dashboard to visualize data + predictions

Try in Google Colab / Binder
---------------------------
Open the primary notebooks interactively in Colab by uploading the notebook or using "Open in Colab" links (you can add direct Colab badges to the notebooks if desired). For reproducible environments try Binder.

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

Model Zoo
---------
- classical/xgboost_train.py — Train XGBoost on engineered features
- classical/rf_train.py — RandomForest baseline
- dl/lstm_train.py — LSTM/GRU models with Keras
- prophet/prophet_forecast.py — Prophet forecasting example

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

License
-------
This project uses the MIT License — see LICENSE file for details.

Contact
-------
Maintainer: Ashish Sinha (@Ashish-Sinha07)
- GitHub: https://github.com/Ashish-Sinha07
- For questions or collabs, open an issue or connect on GitHub.

What's next
-----------
- Try the Quickstart flow above, run the notebooks, and launch the Streamlit dashboard.
- If you'd like, I can:
  - Add "Open in Colab" badges for the notebooks
  - Create a Binder/Repo2Docker config
  - Scaffold a GitHub Actions CI workflow to run tests and linting

If you want any of those, tell me which and I'll prepare the files and steps.
