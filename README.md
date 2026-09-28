# 📈 Trading Risk & Return Analyzer

> **Bayesian · Kernel · Statistical Machine Learning Dashboard**  
> Built with Streamlit · Plotly · Scikit-Learn · NumPy

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧠 Overview

A production-grade ML visualization dashboard that trains **11 custom statistical models** on live market data (yfinance) and presents end-to-end analysis across three insight tabs.

Every model is implemented **from scratch in NumPy** — no sklearn shortcuts for the core math — then benchmarked against sklearn equivalents.

---

## 🤖 Models Implemented

| Type | Model | Implementation |
|------|-------|---------------|
| Regression | Ordinary Least Squares | Custom NumPy (Normal Equation) |
| Regression | Robust Huber (IRLS) | Custom NumPy |
| Regression | Ridge Regression | Custom NumPy (L2 Closed-Form) |
| Regression | Bayesian Linear | Custom NumPy (Posterior Update) |
| Classification | Logistic Regression | Custom NumPy (Gradient Descent) |
| Classification | Bayesian Logistic (Laplace) | Custom NumPy (Newton-Raphson + Probit) |
| Classification | Kernel Logistic (RBF) | Custom NumPy (Dual Gradient Descent) |
| Classification | Linear Discriminant Analysis | Scikit-Learn |
| Classification | Gaussian Naive Bayes | Scikit-Learn |
| Classification | SVM (RBF kernel) | Scikit-Learn |
| Classification | SVM (Linear kernel) | Scikit-Learn |

---

## 📊 Features

### Tab 1 · Data Profiling & Feature Space
- Interactive candlestick OHLCV chart (last 500 sessions)
- Feature distribution histograms (multiselect)
- Full 13×13 Pearson correlation heatmap
- Rolling volatility (5-day / 20-day) with area fill
- Log-return distribution vs Normal fit (fat-tail check)
- Class balance analysis

### Tab 2 · Model Performance & Evaluation
- Regression leaderboard (RMSE, MAE, R²)
- Return prediction chart with **Bayesian 95% uncertainty band**
- Classification leaderboard (Accuracy, Precision, Recall, F1, ROC-AUC)
- ROC curves + Precision-Recall curves for all 7 classifiers
- Confusion matrix heatmap (best model)
- Multi-model radar chart

### Tab 3 · XAI & Interpretability
- OLS coefficient magnitude chart (feature importance)
- Bayesian Linear predictive uncertainty (σ) over test timeline
- Bayesian Logistic Laplace activation uncertainty
- 3D PCA feature space projection (coloured by true class)
- Ridge λ sensitivity sweep (regularization path)
- KLR predicted probability histogram by class
- Residual vs fitted plots
- Downloadable classification metrics CSV

---

## 🎛️ Engineered Features (13 total)

| Feature | Description |
|---------|-------------|
| `log_ret_1/5/20` | Log returns over 1, 5, 20 days |
| `sma_5/20_ratio` | Close / SMA ratio (price momentum) |
| `ema_12_26` | EMA(12)/EMA(26) − 1 (MACD signal) |
| `rsi_14` | RSI normalized to [0,1] |
| `volatility_5/20` | Rolling std of log returns |
| `atr_14_pct` | ATR as % of price |
| `volume_change_5` | 5-day volume change ratio |
| `range_pct` | Intraday range as % of close |
| `close_position_20` | Close position in 20-day range |

**Targets:**
- `target_return` — next-day log return (regression)
- `target_direction` — 1 if next-day return > 0 (classification)

**Train/Val/Test split:** 70% / 15% / 15% chronological (no look-ahead)

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/trading-risk-return-analyzer.git
cd trading-risk-return-analyzer

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## ☁️ Deploy to Streamlit Community Cloud (Free)

1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo
4. Set **Main file path** → `app.py`
5. Click **Deploy**

No extra config needed — `requirements.txt` is auto-detected.

---

## 📁 Project Structure

```
trading-risk-return-analyzer/
├── app.py                  # Main Streamlit dashboard
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| [Streamlit](https://streamlit.io) | Web app framework |
| [Plotly](https://plotly.com/python/) | Interactive visualizations |
| [yfinance](https://github.com/ranaroussi/yfinance) | Market data |
| [NumPy](https://numpy.org) | Custom model math |
| [Scikit-Learn](https://scikit-learn.org) | Benchmark models + preprocessing |
| [SciPy](https://scipy.org) | Statistical distributions |

---

## 👤 Author

**Vijay** — B.Tech CSE @ SRMIST Kattankulathur  
ML/AI Builder · Funded Trader

---

## 📄 License

MIT License — free to use, modify, and distribute.
