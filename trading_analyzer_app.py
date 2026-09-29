"""
Trading Risk & Return Analyzer
Bayesian, Kernel & Statistical ML — Production Streamlit Dashboard
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TradeStatML",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ─── Base & Reset ─── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0B0C10 !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stAppViewContainer"] > .main { background-color: #0B0C10; }
[data-testid="stHeader"] { background-color: #0B0C10; }
[data-testid="stSidebar"] { background-color: #0E1117 !important; border-right: 1px solid #1F2833; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #C5C6C7; }

/* ─── Hide Streamlit chrome ─── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="collapsedControl"] { color: #66FCF1; }

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0E1117; }
::-webkit-scrollbar-thumb { background: #1F2833; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #66FCF1; }

/* ─── Page title ─── */
.page-header {
    background: linear-gradient(135deg, #0E1117 0%, #1F2833 100%);
    border: 1px solid #1F2833;
    border-left: 4px solid #66FCF1;
    border-radius: 0 12px 12px 0;
    padding: 20px 28px;
    margin-bottom: 24px;
}
.page-header h1 {
    color: #FFFFFF;
    font-size: 1.7rem;
    font-weight: 700;
    margin: 0 0 4px 0;
    letter-spacing: -0.02em;
}
.page-header p { color: #66FCF1; font-size: 0.82rem; margin: 0; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; }

/* ─── KPI Cards ─── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }
.kpi-card {
    background: #1F2833;
    border: 1px solid #2E3035;
    border-radius: 12px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}
.kpi-card.cyan::before { background: linear-gradient(90deg, #66FCF1, #45A29E); }
.kpi-card.purple::before { background: linear-gradient(90deg, #8A2BE2, #C77DFF); }
.kpi-card.green::before { background: linear-gradient(90deg, #00FF88, #00C96A); }
.kpi-card.orange::before { background: linear-gradient(90deg, #FF6B35, #FF8E53); }
.kpi-label { color: #66FCF1; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px; }
.kpi-value { color: #FFFFFF; font-size: 1.85rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; line-height: 1; margin-bottom: 6px; }
.kpi-delta { font-size: 0.76rem; font-weight: 500; }
.kpi-delta.up { color: #00FF88; }
.kpi-delta.down { color: #FF4757; }
.kpi-delta.neutral { color: #C5C6C7; }
.kpi-sub { color: #66768A; font-size: 0.72rem; margin-top: 4px; }

/* ─── Section headers ─── */
.section-title {
    color: #C5C6C7;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 28px 0 14px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::after { content: ''; flex: 1; height: 1px; background: #1F2833; }

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    background: #0E1117;
    border-bottom: 1px solid #1F2833;
    gap: 0;
    padding: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #66768A;
    border-radius: 0;
    padding: 12px 24px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #66FCF1 !important;
    border-bottom: 2px solid #66FCF1 !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #C5C6C7 !important; }
.stTabs [data-baseweb="tab-panel"] { background: #0B0C10; padding: 24px 0; }

/* ─── Metric cards inside tabs ─── */
.metric-card {
    background: #1F2833;
    border: 1px solid #2E3035;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.metric-card h4 { color: #66FCF1; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; margin: 0 0 8px 0; }
.metric-card .val { color: #FFF; font-family: 'JetBrains Mono', monospace; font-size: 1.2rem; font-weight: 600; }

/* ─── Info / Warning banners ─── */
.info-banner {
    background: rgba(102, 252, 241, 0.06);
    border: 1px solid rgba(102, 252, 241, 0.2);
    border-radius: 10px;
    padding: 16px 20px;
    color: #66FCF1;
    font-size: 0.84rem;
    margin: 16px 0;
}
.warn-banner {
    background: rgba(255, 107, 53, 0.08);
    border: 1px solid rgba(255, 107, 53, 0.25);
    border-radius: 10px;
    padding: 14px 20px;
    color: #FF8E53;
    font-size: 0.82rem;
    margin: 12px 0;
}

/* ─── Sidebar elements ─── */
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label { color: #C5C6C7 !important; font-size: 0.8rem; }
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stMultiSelect > div { background: #1F2833 !important; border: 1px solid #2E3035 !important; border-radius: 8px; color: #FFF; }
[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] { color: #66768A; }

/* ─── Dataframe ─── */
[data-testid="stDataFrame"] { border: 1px solid #1F2833; border-radius: 10px; overflow: hidden; }
[data-testid="stDataFrame"] thead tr th { background: #1F2833 !important; color: #66FCF1 !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stDataFrame"] tbody tr:hover td { background: rgba(102,252,241,0.04) !important; }

/* ─── Spinner ─── */
[data-testid="stSpinner"] { color: #66FCF1; }

/* ─── Divider ─── */
hr { border-color: #1F2833; }

/* ─── Plotly chart background ─── */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ── Plotly dark template ────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#1F2833",
    plot_bgcolor="#1F2833",
    font=dict(family="Inter, sans-serif", color="#C5C6C7", size=12),
    xaxis=dict(gridcolor="#2E3035", linecolor="#2E3035", zeroline=False),
    yaxis=dict(gridcolor="#2E3035", linecolor="#2E3035", zeroline=False),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2E3035", borderwidth=1),
    hoverlabel=dict(bgcolor="#0E1117", bordercolor="#66FCF1", font_color="#FFF", font_size=12),
)

CYAN   = "#66FCF1"
PURPLE = "#8A2BE2"
GREEN  = "#00FF88"
ORANGE = "#FF6B35"
RED    = "#FF4757"
YELLOW = "#FFD700"
COLORS = [CYAN, PURPLE, GREEN, ORANGE, RED, YELLOW, "#C77DFF", "#45A29E"]


# ══════════════════════════════════════════════════════════════════════════════
#  DATA & MODEL ENGINE  (all cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_engineer(ticker: str, start: str, end: str):
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    # ─ Returns
    df["log_ret_1"]  = np.log(df["Close"] / df["Close"].shift(1))
    df["log_ret_5"]  = np.log(df["Close"] / df["Close"].shift(5))
    df["log_ret_20"] = np.log(df["Close"] / df["Close"].shift(20))

    # ─ SMA
    df["sma_5"]       = df["Close"].rolling(5).mean()
    df["sma_20"]      = df["Close"].rolling(20).mean()
    df["sma_5_ratio"] = (df["Close"] / df["sma_5"]) - 1
    df["sma_20_ratio"]= (df["Close"] / df["sma_20"]) - 1

    # ─ EMA MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["ema_12_26"] = (ema12 / ema26) - 1

    # ─ RSI
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["rsi_14"] = (100 - (100 / (1 + rs))) / 100.0

    # ─ Volatility
    df["volatility_5"]  = df["log_ret_1"].rolling(5).std()
    df["volatility_20"] = df["log_ret_1"].rolling(20).std()

    # ─ ATR%
    h_l  = df["High"] - df["Low"]
    h_pc = np.abs(df["High"] - df["Close"].shift(1))
    l_pc = np.abs(df["Low"]  - df["Close"].shift(1))
    tr   = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    df["atr_14_pct"] = tr.rolling(14).mean() / df["Close"]

    # ─ Volume
    df["volume_change_5"] = (df["Volume"] / df["Volume"].shift(5)) - 1

    # ─ Range & Position
    df["range_pct"] = (df["High"] - df["Low"]) / df["Close"]
    rmin = df["Close"].rolling(20).min()
    rmax = df["Close"].rolling(20).max()
    df["close_position_20"] = (df["Close"] - rmin) / (rmax - rmin + 1e-9)

    # ─ Targets
    df["target_return"]    = df["log_ret_1"].shift(-1)
    df["target_direction"] = (df["target_return"] > 0).astype(int)

    return df


@st.cache_data(show_spinner=False)
def split_and_scale(df, features_list):
    from sklearn.preprocessing import StandardScaler

    cleaned = df.copy()
    for col in features_list + ["target_return", "target_direction"]:
        cleaned[col] = cleaned[col].replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(subset=features_list + ["target_return", "target_direction"])

    X     = cleaned[features_list].values
    y_reg = cleaned["target_return"].values
    y_cls = cleaned["target_direction"].values

    n         = len(cleaned)
    train_idx = int(n * 0.70)
    val_idx   = int(n * 0.85)

    X_tr, y_tr_r, y_tr_c = X[:train_idx], y_reg[:train_idx], y_cls[:train_idx]
    X_va, y_va_r, y_va_c = X[train_idx:val_idx], y_reg[train_idx:val_idx], y_cls[train_idx:val_idx]
    X_te, y_te_r, y_te_c = X[val_idx:], y_reg[val_idx:], y_cls[val_idx:]

    scaler       = StandardScaler()
    X_tr_s       = scaler.fit_transform(X_tr)
    X_va_s       = scaler.transform(X_va)
    X_te_s       = scaler.transform(X_te)

    def bias(M):
        return np.hstack([np.ones((M.shape[0], 1)), M])

    return {
        "cleaned": cleaned,
        "train_idx": train_idx, "val_idx": val_idx,
        "X_tr": X_tr_s, "X_va": X_va_s, "X_te": X_te_s,
        "X_tr_b": bias(X_tr_s), "X_va_b": bias(X_va_s), "X_te_b": bias(X_te_s),
        "y_tr_r": y_tr_r, "y_va_r": y_va_r, "y_te_r": y_te_r,
        "y_tr_c": y_tr_c, "y_va_c": y_va_c, "y_te_c": y_te_c,
        "scaler": scaler,
    }


# ──── Custom Model Classes ─────────────────────────────────────────────────────

class OLS:
    def fit(self, X, y):
        self.beta = np.linalg.pinv(X.T @ X) @ X.T @ y; return self
    def predict(self, X): return X @ self.beta

class RobustHuber:
    def __init__(self, delta=1.345, max_iter=100, tol=1e-5):
        self.delta, self.max_iter, self.tol = delta, max_iter, tol
    def fit(self, X, y):
        n = X.shape[0]; self.beta = np.linalg.pinv(X.T @ X) @ X.T @ y
        for _ in range(self.max_iter):
            old = self.beta.copy()
            r   = y - X @ self.beta; a = np.abs(r)
            w   = np.where(a > self.delta, self.delta / (a + 1e-9), 1.0)
            W   = np.diag(w)
            self.beta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
            if np.linalg.norm(self.beta - old) < self.tol: break
        return self
    def predict(self, X): return X @ self.beta

class Ridge:
    def __init__(self, lmbda=1.0): self.lmbda = lmbda
    def fit(self, X, y):
        p = X.shape[1]; I = np.eye(p); I[0,0] = 0
        self.beta = np.linalg.pinv(X.T @ X + self.lmbda * I) @ X.T @ y; return self
    def predict(self, X): return X @ self.beta

class BayesianLinear:
    def __init__(self, alpha=1.0, beta_prior=1.0):
        self.alpha, self.beta_prior = alpha, beta_prior
    def fit(self, X, y):
        p = X.shape[1]; S0i = self.alpha * np.eye(p); S0i[0,0] = 1e-5
        self.S_N = np.linalg.inv(S0i + self.beta_prior * (X.T @ X))
        self.m_N = self.beta_prior * (self.S_N @ X.T @ y); return self
    def predict(self, X):
        mean = X @ self.m_N
        var  = (1.0 / self.beta_prior) + np.sum((X @ self.S_N) * X, axis=1)
        return mean, np.sqrt(var)

class LogisticRegression:
    def __init__(self, lr=0.01, max_iter=1000): self.lr, self.max_iter = lr, max_iter
    def sigmoid(self, z): return 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        for _ in range(self.max_iter):
            p = self.sigmoid(X @ self.w)
            self.w -= self.lr * (X.T @ (p - y)) / len(y)
        return self
    def predict_proba(self, X): return self.sigmoid(X @ self.w)

class BayesianLogisticLaplace:
    def __init__(self, alpha=1.0, max_iter=100): self.alpha, self.max_iter = alpha, max_iter
    def sigmoid(self, z): return 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))
    def fit(self, X, y):
        p = X.shape[1]; self.w_map = np.zeros(p)
        for _ in range(self.max_iter):
            pe = self.sigmoid(X @ self.w_map)
            g  = X.T @ (pe - y) + self.alpha * self.w_map; g[0] -= self.alpha * self.w_map[0]
            W  = np.diag(pe * (1 - pe))
            H  = X.T @ W @ X + self.alpha * np.eye(p); H[0,0] -= self.alpha
            self.w_map -= np.linalg.pinv(H) @ g
        pe = self.sigmoid(X @ self.w_map); W = np.diag(pe * (1 - pe))
        H  = X.T @ W @ X + self.alpha * np.eye(p); H[0,0] -= self.alpha
        self.cov = np.linalg.pinv(H); return self
    def predict_proba(self, X):
        mu = X @ self.w_map; sig2 = np.sum((X @ self.cov) * X, axis=1)
        return self.sigmoid(mu / np.sqrt(1 + (np.pi / 8) * sig2)), np.sqrt(sig2)

def rbf_kernel(X1, X2, gamma=0.1):
    d = np.sum(X1**2, 1).reshape(-1,1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
    return np.exp(-gamma * d)

class KernelLogistic:
    def __init__(self, gamma=0.1, lmbda=0.1, max_iter=200):
        self.gamma, self.lmbda, self.max_iter = gamma, lmbda, max_iter
    def sigmoid(self, z): return 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))
    def fit(self, X, y):
        self.X_tr = X; n = X.shape[0]; K = rbf_kernel(X, X, self.gamma); self.alpha = np.zeros(n)
        for _ in range(self.max_iter):
            p = self.sigmoid(K @ self.alpha)
            self.alpha -= 0.1 * ((K.T @ (p - y)) / n + self.lmbda * self.alpha)
        return self
    def predict_proba(self, X):
        return self.sigmoid(rbf_kernel(X, self.X_tr, self.gamma) @ self.alpha)


@st.cache_resource(show_spinner=False)
def train_all_models(split_key: str, ridge_lmbda: float, bayes_alpha: float, bayes_beta: float,
                     klr_gamma: float, klr_lmbda: float, ticker: str, start: str, end: str):
    """Train all models and return fitted objects + metrics."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                                 accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, roc_curve, precision_recall_curve,
                                 confusion_matrix)

    df  = load_and_engineer(ticker, start, end)
    spl = split_and_scale(df, FEATURES)

    X_tr_b, X_te_b = spl["X_tr_b"], spl["X_te_b"]
    X_tr,   X_te   = spl["X_tr"],   spl["X_te"]
    y_tr_r, y_te_r = spl["y_tr_r"], spl["y_te_r"]
    y_tr_c, y_te_c = spl["y_tr_c"], spl["y_te_c"]

    # ── Regression ──
    ols_m   = OLS().fit(X_tr_b, y_tr_r)
    hub_m   = RobustHuber().fit(X_tr_b, y_tr_r)
    rid_m   = Ridge(lmbda=ridge_lmbda).fit(X_tr_b, y_tr_r)
    bay_m   = BayesianLinear(alpha=bayes_alpha, beta_prior=bayes_beta).fit(X_tr_b, y_tr_r)

    pred_ols  = ols_m.predict(X_te_b)
    pred_hub  = hub_m.predict(X_te_b)
    pred_rid  = rid_m.predict(X_te_b)
    pred_bay, pred_bay_std = bay_m.predict(X_te_b)

    reg_metrics = []
    for nm, prd in [("OLS", pred_ols), ("Huber", pred_hub), ("Ridge", pred_rid), ("Bayesian", pred_bay)]:
        reg_metrics.append({
            "Model": nm,
            "RMSE":  float(np.sqrt(mean_squared_error(y_te_r, prd))),
            "MAE":   float(mean_absolute_error(y_te_r, prd)),
            "R²":    float(r2_score(y_te_r, prd)),
        })

    # ── Classification ──
    lda_m  = LDA().fit(X_tr, y_tr_c)
    gnb_m  = GaussianNB().fit(X_tr, y_tr_c)
    log_m  = LogisticRegression().fit(X_tr_b, y_tr_c)
    bll_m  = BayesianLogisticLaplace(alpha=0.1).fit(X_tr_b, y_tr_c)
    klr_m  = KernelLogistic(gamma=klr_gamma, lmbda=klr_lmbda).fit(X_tr, y_tr_c)
    svm_r  = SVC(kernel="rbf",   probability=True).fit(X_tr, y_tr_c)
    svm_l  = SVC(kernel="linear",probability=True).fit(X_tr, y_tr_c)

    probs = {
        "LDA":               lda_m.predict_proba(X_te)[:,1],
        "Gaussian NB":       gnb_m.predict_proba(X_te)[:,1],
        "Logistic":          log_m.predict_proba(X_te_b),
        "Bayesian Logistic": bll_m.predict_proba(X_te_b)[0],
        "Kernel Logistic":   klr_m.predict_proba(X_te),
        "RBF SVM":           svm_r.predict_proba(X_te)[:,1],
        "Linear SVM":        svm_l.predict_proba(X_te)[:,1],
    }

    cls_metrics = []
    roc_data, pr_data = {}, {}
    for nm, pb in probs.items():
        pd_ = (pb >= 0.5).astype(int)
        cls_metrics.append({
            "Model":     nm,
            "Accuracy":  float(accuracy_score(y_te_c, pd_)),
            "Precision": float(precision_score(y_te_c, pd_, zero_division=0)),
            "Recall":    float(recall_score(y_te_c, pd_, zero_division=0)),
            "F1":        float(f1_score(y_te_c, pd_, zero_division=0)),
            "ROC-AUC":   float(roc_auc_score(y_te_c, pb)),
        })
        fpr, tpr, _ = roc_curve(y_te_c, pb)
        roc_data[nm] = (fpr, tpr)
        prec_arr, rec_arr, _ = precision_recall_curve(y_te_c, pb)
        pr_data[nm] = (prec_arr, rec_arr)

    # Best model confusion matrix
    best_nm = max(cls_metrics, key=lambda x: x["ROC-AUC"])["Model"]
    cm = confusion_matrix(y_te_c, (probs[best_nm] >= 0.5).astype(int))

    # Bayesian logistic uncertainty
    bll_std = bll_m.predict_proba(X_te_b)[1]

    return {
        "df": df, "spl": spl,
        "pred_ols": pred_ols, "pred_hub": pred_hub, "pred_rid": pred_rid,
        "pred_bay": pred_bay, "pred_bay_std": pred_bay_std,
        "reg_metrics": pd.DataFrame(reg_metrics),
        "cls_metrics": pd.DataFrame(cls_metrics),
        "probs": probs, "roc_data": roc_data, "pr_data": pr_data,
        "cm": cm, "best_nm": best_nm,
        "bll_std": bll_std,
    }


FEATURES = [
    "log_ret_1","log_ret_5","log_ret_20","sma_5_ratio","sma_20_ratio",
    "ema_12_26","rsi_14","atr_14_pct","volatility_5","volatility_20",
    "volume_change_5","range_pct","close_position_20",
]

TICKER_OPTIONS = {
    "NIFTY 50 (^NSEI)":     "^NSEI",
    "SENSEX (^BSESN)":      "^BSESN",
    "S&P 500 (^GSPC)":      "^GSPC",
    "NASDAQ (^IXIC)":       "^IXIC",
    "Gold Futures (GC=F)":  "GC=F",
    "Crude Oil (CL=F)":     "CL=F",
}


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 24px 0;'>
        <div style='font-size:2.2rem;'>📈</div>
        <div style='color:#66FCF1; font-size:0.9rem; font-weight:700; letter-spacing:0.05em;'>TRADING ANALYZER</div>
        <div style='color:#66768A; font-size:0.68rem; margin-top:2px; text-transform:uppercase; letter-spacing:0.1em;'>Bayesian · Kernel · Statistical ML</div>
    </div>
    <hr style='border-color:#1F2833; margin: 0 0 20px 0;'>
    """, unsafe_allow_html=True)

    st.markdown("**📡 Data Source**")
    ticker_label = st.selectbox("Market Index", options=list(TICKER_OPTIONS.keys()), index=0, label_visibility="collapsed")
    ticker = TICKER_OPTIONS[ticker_label]

    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("Start", value=pd.Timestamp("2015-01-01"), label_visibility="visible")
    with col_e:
        end_date   = st.date_input("End",   value=pd.Timestamp("2024-12-31"), label_visibility="visible")

    st.markdown("<div style='margin-top:20px;'><b>⚙️ Model Hyperparameters</b></div>", unsafe_allow_html=True)

    ridge_lmbda = st.slider("Ridge λ (Regularization)",  0.01, 10.0, 1.0, 0.01,
                             help="L2 penalty for Ridge regression")
    bayes_alpha = st.slider("Bayesian α (Prior Precision)", 0.001, 0.5, 0.01, 0.001,
                             help="Precision of Gaussian prior over weights")
    bayes_beta  = st.slider("Bayesian β (Noise Precision)", 1.0, 200.0, 50.0, 1.0,
                             help="Precision of noise likelihood")
    klr_gamma   = st.slider("KLR γ (RBF bandwidth)",  0.001, 1.0, 0.01, 0.001,
                             help="RBF kernel bandwidth for Kernel Logistic")
    klr_lmbda   = st.slider("KLR λ (Kernel L2)",  0.01, 2.0, 0.10, 0.01,
                             help="Regularization for Kernel Logistic")

    run_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

    st.markdown("<hr style='border-color:#1F2833; margin:24px 0 16px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#66768A; font-size:0.7rem; line-height:1.6;'>
    <b style='color:#C5C6C7;'>Models trained:</b><br>
    📐 OLS · Huber · Ridge · Bayesian Linear<br>
    🔷 Logistic · Bayesian Logistic (Laplace)<br>
    🔮 Kernel Logistic (RBF) · LDA · NB · SVM
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if "results" not in st.session_state:
    st.session_state.results = None

if run_btn:
    st.session_state.results = None  # invalidate cache key
    with st.spinner("⚙️  Downloading market data & training 11 models…"):
        try:
            key = f"{ticker}_{start_date}_{end_date}_{ridge_lmbda}_{bayes_alpha}_{bayes_beta}_{klr_gamma}_{klr_lmbda}"
            st.session_state.results = train_all_models(
                key, ridge_lmbda, bayes_alpha, bayes_beta,
                klr_gamma, klr_lmbda, ticker,
                str(start_date), str(end_date)
            )
            st.session_state.ticker_label = ticker_label
        except Exception as e:
            st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class='page-header'>
  <p>📊 TradeStatML</p>
  <h1>Bayesian · Kernel · Statistical Machine Learning</h1>
</div>
""", unsafe_allow_html=True)

R = st.session_state.results

# ── Empty state ────────────────────────────────────────────────────────────────
if R is None:
    st.markdown("""
    <div class='info-banner'>
      ⚡ Configure your market index and hyperparameters in the sidebar, then click
      <strong>Run Analysis</strong> to train all 11 models and explore the full ML pipeline.
    </div>
    """, unsafe_allow_html=True)

    # Teaser cards
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, title, desc in [
        (c1, "📐", "4 Regressors", "OLS · Huber · Ridge · Bayesian"),
        (c2, "🔷", "7 Classifiers", "LDA · NB · Logistic · SVM · KLR"),
        (c3, "🔮", "Kernel Methods", "Custom RBF Kernel Logistic"),
        (c4, "📊", "3 Analysis Tabs", "EDA · Performance · XAI"),
    ]:
        with col:
            st.markdown(f"""
            <div class='kpi-card cyan' style='text-align:center; padding:28px 16px;'>
              <div style='font-size:2rem;'>{icon}</div>
              <div style='color:#66FCF1; font-size:0.85rem; font-weight:700; margin:8px 0 4px;'>{title}</div>
              <div style='color:#66768A; font-size:0.74rem;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.stop()


# ── KPI Ribbon ─────────────────────────────────────────────────────────────────
df    = R["df"]
spl   = R["spl"]
cm    = R["cls_metrics"]
rm    = R["reg_metrics"]

best_cls  = cm.loc[cm["ROC-AUC"].idxmax()]
best_f1   = cm["F1"].max()
best_rmse = rm["RMSE"].min()
n_samples = len(spl["cleaned"])
n_feat    = len(FEATURES)

st.markdown(f"""
<div class='kpi-grid'>
  <div class='kpi-card cyan'>
    <div class='kpi-label'>Best ROC-AUC</div>
    <div class='kpi-value'>{best_cls['ROC-AUC']:.4f}</div>
    <div class='kpi-delta up'>▲ {best_cls['Model']}</div>
    <div class='kpi-sub'>Classification leader</div>
  </div>
  <div class='kpi-card purple'>
    <div class='kpi-label'>Best F1-Score</div>
    <div class='kpi-value'>{best_f1:.4f}</div>
    <div class='kpi-delta up'>▲ {cm.loc[cm['F1'].idxmax(),'Model']}</div>
    <div class='kpi-sub'>Harmonic precision-recall</div>
  </div>
  <div class='kpi-card green'>
    <div class='kpi-label'>Best RMSE</div>
    <div class='kpi-value'>{best_rmse:.5f}</div>
    <div class='kpi-delta up'>▲ {rm.loc[rm['RMSE'].idxmin(),'Model']}</div>
    <div class='kpi-sub'>Return prediction error</div>
  </div>
  <div class='kpi-card orange'>
    <div class='kpi-label'>Dataset Size</div>
    <div class='kpi-value'>{n_samples:,}</div>
    <div class='kpi-delta neutral'>→ {n_feat} features · {st.session_state.get('ticker_label','')}</div>
    <div class='kpi-sub'>Clean samples after NaN drop</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "📊  Data Profiling & Feature Space",
    "🤖  Model Performance & Evaluation",
    "🔍  XAI & Interpretability",
])


# ════════════════════════════════════ TAB 1 ═══════════════════════════════════
with tab1:
    cleaned = spl["cleaned"]

    # ── Price chart ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>OHLCV Price History</div>", unsafe_allow_html=True)

    fig_price = go.Figure()
    fig_price.add_trace(go.Candlestick(
        x=df.index[-500:], open=df["Open"].iloc[-500:], high=df["High"].iloc[-500:],
        low=df["Low"].iloc[-500:], close=df["Close"].iloc[-500:],
        increasing_line_color=GREEN, decreasing_line_color=RED,
        increasing_fillcolor=GREEN, decreasing_fillcolor=RED,
        name="OHLCV", line_width=1,
    ))
    vol_colors = [GREEN if c >= o else RED
                  for c, o in zip(df["Close"].iloc[-500:], df["Open"].iloc[-500:])]
    fig_price.add_trace(go.Bar(
        x=df.index[-500:], y=df["Volume"].iloc[-500:],
        marker_color=vol_colors, opacity=0.35,
        name="Volume", yaxis="y2",
    ))
    fig_price.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Volume"),
        xaxis_rangeslider_visible=False,
        title=dict(text=f"Last 500 sessions · {st.session_state.get('ticker_label','')}", font_color=CYAN, font_size=13),
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # ── Feature distributions ────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Feature Distribution Overview</div>", unsafe_allow_html=True)

    feat_sel = st.multiselect(
        "Select features to profile:", FEATURES,
        default=["log_ret_1","rsi_14","volatility_20","atr_14_pct"],
    )

    if feat_sel:
        n_cols = min(len(feat_sel), 2)
        n_rows = (len(feat_sel) + 1) // 2
        fig_dist = make_subplots(rows=n_rows, cols=n_cols,
                                  subplot_titles=[f"<b>{f}</b>" for f in feat_sel])
        for i, feat in enumerate(feat_sel):
            r, c = divmod(i, n_cols)
            vals = cleaned[feat].dropna()
            fig_dist.add_trace(go.Histogram(
                x=vals, nbinsx=60,
                marker_color=COLORS[i % len(COLORS)], opacity=0.8,
                name=feat, showlegend=False,
            ), row=r+1, col=c+1)
        fig_dist.update_layout(**PLOTLY_LAYOUT, height=280*n_rows,
                                title_text="Feature Distributions", title_font_color=CYAN)
        fig_dist.update_annotations(font_color="#C5C6C7", font_size=11)
        st.plotly_chart(fig_dist, use_container_width=True)

    # ── Correlation heatmap ──────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Feature Correlation Matrix</div>", unsafe_allow_html=True)

    corr = cleaned[FEATURES].corr()
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0,"#FF4757"], [0.5,"#1F2833"], [1,"#66FCF1"]],
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}", textfont_size=9,
        hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.3f}<extra></extra>",
    ))
    fig_corr.update_layout(**PLOTLY_LAYOUT, height=480,
                            title=dict(text="Pearson Correlation · Feature Space", font_color=CYAN))
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Rolling volatility ───────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Realized Volatility & Log-Return Distribution</div>", unsafe_allow_html=True)

    col_v, col_h = st.columns([2, 1])
    with col_v:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=cleaned.index, y=cleaned["volatility_20"],
            fill="tozeroy", fillcolor="rgba(138,43,226,0.15)",
            line=dict(color=PURPLE, width=1.5), name="20-Day Vol",
        ))
        fig_vol.add_trace(go.Scatter(
            x=cleaned.index, y=cleaned["volatility_5"],
            line=dict(color=CYAN, width=1, dash="dot"), name="5-Day Vol",
        ))
        fig_vol.update_layout(**PLOTLY_LAYOUT, height=280,
                               title=dict(text="Rolling Realized Volatility", font_color=CYAN))
        st.plotly_chart(fig_vol, use_container_width=True)

    with col_h:
        rets = cleaned["log_ret_1"].dropna()
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Histogram(
            x=rets, nbinsx=80,
            marker_color=CYAN, opacity=0.75, name="Log Returns",
        ))
        # Normal overlay
        x_norm = np.linspace(rets.min(), rets.max(), 200)
        from scipy.stats import norm
        pdf = norm.pdf(x_norm, rets.mean(), rets.std()) * len(rets) * (rets.max()-rets.min()) / 80
        fig_ret.add_trace(go.Scatter(
            x=x_norm, y=pdf, mode="lines",
            line=dict(color=ORANGE, width=2, dash="dash"), name="Normal fit",
        ))
        fig_ret.update_layout(**PLOTLY_LAYOUT, height=280,
                               title=dict(text="Return Distribution", font_color=CYAN))
        st.plotly_chart(fig_ret, use_container_width=True)

    # ── Class balance ────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Target Class Balance</div>", unsafe_allow_html=True)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        counts = pd.Series(spl["y_tr_c"]).value_counts().sort_index()
        fig_cls = go.Figure(go.Bar(
            x=["Down / Flat (0)", "Up (1)"], y=counts.values,
            marker_color=[RED, GREEN], text=counts.values,
            textposition="outside", textfont_color="#FFF",
        ))
        fig_cls.update_layout(**PLOTLY_LAYOUT, height=300,
                               title=dict(text="Train Class Distribution", font_color=CYAN))
        st.plotly_chart(fig_cls, use_container_width=True)

    with col_b2:
        pct = counts / counts.sum() * 100
        st.markdown(f"""
        <div class='metric-card' style='margin-top:24px;'>
          <h4>Class Balance</h4>
          <div class='val' style='color:{GREEN};'>{pct.iloc[1]:.1f}% Up (Class 1)</div>
          <div class='val' style='color:{RED};'>{pct.iloc[0]:.1f}% Down/Flat (Class 0)</div>
        </div>
        <div class='warn-banner'>
          ⚠ Near-balanced dataset. Kernel Logistic may default to majority class —
          monitor Precision & Recall separately.
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════ TAB 2 ═══════════════════════════════════
with tab2:

    # ── Regression metrics table ─────────────────────────────────────────────
    st.markdown("<div class='section-title'>Regression Leaderboard (Test Set)</div>", unsafe_allow_html=True)

    rm_styled = R["reg_metrics"].copy()
    rm_styled["RMSE"] = rm_styled["RMSE"].map(lambda x: f"{x:.6f}")
    rm_styled["MAE"]  = rm_styled["MAE"].map(lambda x: f"{x:.6f}")
    rm_styled["R²"]   = rm_styled["R²"].map(lambda x: f"{x:.5f}")
    st.dataframe(rm_styled, use_container_width=True, hide_index=True)

    # ── Regression return prediction chart ───────────────────────────────────
    st.markdown("<div class='section-title'>Return Predictions vs Actual (Test Split)</div>", unsafe_allow_html=True)

    test_idx = spl["cleaned"].index[spl["val_idx"]:]
    y_te_r   = spl["y_te_r"]
    pred_bay_std = R["pred_bay_std"]

    fig_reg = go.Figure()
    fig_reg.add_trace(go.Scatter(
        x=test_idx, y=y_te_r,
        line=dict(color="#C5C6C7", width=1), opacity=0.5, name="Actual",
    ))
    for nm, pred, clr in [
        ("OLS",  R["pred_ols"], CYAN),
        ("Huber",R["pred_hub"], PURPLE),
        ("Ridge",R["pred_rid"], ORANGE),
    ]:
        fig_reg.add_trace(go.Scatter(x=test_idx, y=pred,
                                      line=dict(color=clr, width=1.2),
                                      opacity=0.8, name=nm))
    fig_reg.add_trace(go.Scatter(
        x=test_idx, y=R["pred_bay"],
        line=dict(color=GREEN, width=1.5, dash="dot"), name="Bayesian Mean",
    ))
    fig_reg.add_trace(go.Scatter(
        x=list(test_idx) + list(test_idx[::-1]),
        y=list(R["pred_bay"] + 1.96*pred_bay_std) + list((R["pred_bay"] - 1.96*pred_bay_std)[::-1]),
        fill="toself", fillcolor="rgba(0,255,136,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="Bayesian 95% CI",
    ))
    fig_reg.update_layout(**PLOTLY_LAYOUT, height=380,
                           title=dict(text="Log-Return Predictions with Bayesian Uncertainty Band", font_color=CYAN))
    st.plotly_chart(fig_reg, use_container_width=True)

    # ── Classification leaderboard ───────────────────────────────────────────
    st.markdown("<div class='section-title'>Classification Leaderboard (Test Set)</div>", unsafe_allow_html=True)

    cm_styled = R["cls_metrics"].copy()
    for col in ["Accuracy","Precision","Recall","F1","ROC-AUC"]:
        cm_styled[col] = cm_styled[col].map(lambda x: f"{x:.4f}")
    st.dataframe(cm_styled, use_container_width=True, hide_index=True)

    # ── ROC curves ───────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>ROC Curves · All Classifiers</div>", unsafe_allow_html=True)

    col_roc, col_pr = st.columns(2)
    with col_roc:
        fig_roc = go.Figure()
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                           line=dict(color="#2E3035", dash="dash"))
        for i, (nm, (fpr, tpr)) in enumerate(R["roc_data"].items()):
            auc_val = R["cls_metrics"].loc[R["cls_metrics"]["Model"]==nm, "ROC-AUC"].values[0]
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                line=dict(color=COLORS[i % len(COLORS)], width=1.8),
                name=f"{nm} ({auc_val:.3f})",
                hovertemplate="FPR=%{x:.3f}<br>TPR=%{y:.3f}<extra></extra>",
            ))
        fig_roc.update_layout(**PLOTLY_LAYOUT, height=380,
                               xaxis_title="False Positive Rate",
                               yaxis_title="True Positive Rate",
                               title=dict(text="ROC Curves — AUC in Legend", font_color=CYAN))
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_pr:
        fig_pr = go.Figure()
        base_pr = np.mean(spl["y_te_c"])
        fig_pr.add_shape(type="line", x0=0, y0=base_pr, x1=1, y1=base_pr,
                          line=dict(color="#2E3035", dash="dash"))
        for i, (nm, (prec_arr, rec_arr)) in enumerate(R["pr_data"].items()):
            fig_pr.add_trace(go.Scatter(
                x=rec_arr, y=prec_arr, mode="lines",
                line=dict(color=COLORS[i % len(COLORS)], width=1.8),
                name=nm,
                hovertemplate="Recall=%{x:.3f}<br>Precision=%{y:.3f}<extra></extra>",
            ))
        fig_pr.update_layout(**PLOTLY_LAYOUT, height=380,
                              xaxis_title="Recall", yaxis_title="Precision",
                              title=dict(text="Precision-Recall Curves", font_color=CYAN))
        st.plotly_chart(fig_pr, use_container_width=True)

    # ── Confusion matrix ─────────────────────────────────────────────────────
    st.markdown(f"<div class='section-title'>Confusion Matrix · {R['best_nm']} (Best AUC)</div>",
                unsafe_allow_html=True)

    col_cm, col_bar = st.columns([1, 2])
    with col_cm:
        cm_arr = R["cm"]
        labels = ["Down/Flat", "Up"]
        fig_cm  = go.Figure(go.Heatmap(
            z=cm_arr, x=labels, y=labels,
            colorscale=[[0,"#0B0C10"],[0.5,"rgba(138,43,226,0.6)"],[1,"#66FCF1"]],
            text=cm_arr, texttemplate="<b>%{text}</b>", textfont_size=20,
            hovertemplate="True=%{y}<br>Pred=%{x}<br>Count=%{z}<extra></extra>",
        ))
        fig_cm.update_layout(**PLOTLY_LAYOUT, height=340,
                              xaxis_title="Predicted", yaxis_title="Actual",
                              title=dict(text="Confusion Matrix", font_color=CYAN))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_bar:
        best_row = R["cls_metrics"].loc[R["cls_metrics"]["Model"] == R["best_nm"]].iloc[0]
        metrics_bar = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
        vals_bar    = [best_row[m] for m in metrics_bar]
        fig_bar = go.Figure(go.Bar(
            x=metrics_bar, y=vals_bar,
            marker=dict(color=[CYAN, PURPLE, GREEN, ORANGE, YELLOW], opacity=0.85),
            text=[f"{v:.4f}" for v in vals_bar],
            textposition="outside", textfont=dict(color="#FFF"),
        ))
        fig_bar.update_layout(**PLOTLY_LAYOUT, height=340, yaxis_range=[0, 1.12],
                               title=dict(text=f"{R['best_nm']} — All Metrics", font_color=CYAN))
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Classifier comparison radar ───────────────────────────────────────────
    st.markdown("<div class='section-title'>Model Comparison Radar</div>", unsafe_allow_html=True)

    radar_feats = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
    fig_radar = go.Figure()
    for i, row in R["cls_metrics"].iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row[f] for f in radar_feats] + [row[radar_feats[0]]],
            theta=radar_feats + [radar_feats[0]],
            fill="toself", fillcolor=COLORS[i % len(COLORS)].replace("#","rgba(").rstrip(")") if False else "rgba(0,0,0,0)",
            line=dict(color=COLORS[i % len(COLORS)], width=1.6),
            name=row["Model"], opacity=0.8,
        ))
    fig_radar.update_layout(
        **PLOTLY_LAYOUT, height=440,
        polar=dict(
            radialaxis=dict(visible=True, range=[0.4, 1.0], gridcolor="#2E3035",
                            tickfont_color="#66768A", linecolor="#2E3035"),
            angularaxis=dict(gridcolor="#2E3035", linecolor="#2E3035", tickfont_color="#C5C6C7"),
            bgcolor="#1F2833",
        ),
        title=dict(text="Multi-Metric Radar · All Classifiers", font_color=CYAN),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ════════════════════════════════════ TAB 3 ═══════════════════════════════════
with tab3:

    # ── Feature importance (OLS coefficient magnitudes) ──────────────────────
    st.markdown("<div class='section-title'>Regression Weight Analysis (OLS Coefficients)</div>", unsafe_allow_html=True)

    from sklearn.preprocessing import StandardScaler as _SS
    _ols = OLS().fit(spl["X_tr_b"], spl["y_tr_r"])
    coef = _ols.beta[1:]   # skip intercept
    importance_df = pd.DataFrame({"Feature": FEATURES, "Coefficient": coef}).sort_values("Coefficient", key=abs, ascending=True)

    fig_coef = go.Figure(go.Bar(
        x=importance_df["Coefficient"],
        y=importance_df["Feature"],
        orientation="h",
        marker=dict(
            color=importance_df["Coefficient"].apply(lambda v: GREEN if v > 0 else RED),
            opacity=0.85,
        ),
        text=importance_df["Coefficient"].map(lambda x: f"{x:.5f}"),
        textposition="outside",
        textfont_color="#FFF",
    ))
    fig_coef.update_layout(**PLOTLY_LAYOUT, height=480,
                            xaxis_title="Coefficient Value",
                            title=dict(text="OLS Coefficients (Sorted by |Magnitude|)", font_color=CYAN))
    st.plotly_chart(fig_coef, use_container_width=True)

    # ── Bayesian uncertainty band ─────────────────────────────────────────────
    st.markdown("<div class='section-title'>Bayesian Posterior Uncertainty (Predictive Std Dev)</div>", unsafe_allow_html=True)

    test_idx  = spl["cleaned"].index[spl["val_idx"]:]
    bay_std   = R["pred_bay_std"]
    bll_std   = R["bll_std"]

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        fig_bay_std = go.Figure()
        fig_bay_std.add_trace(go.Scatter(
            x=test_idx, y=bay_std,
            fill="tozeroy", fillcolor="rgba(102,252,241,0.1)",
            line=dict(color=CYAN, width=1.5), name="Posterior σ",
        ))
        fig_bay_std.update_layout(**PLOTLY_LAYOUT, height=310,
                                   yaxis_title="σ (Predictive Std Dev)",
                                   title=dict(text="Bayesian Linear — Predictive Uncertainty", font_color=CYAN))
        st.plotly_chart(fig_bay_std, use_container_width=True)

    with col_u2:
        fig_bll_std = go.Figure()
        fig_bll_std.add_trace(go.Scatter(
            x=test_idx, y=bll_std,
            fill="tozeroy", fillcolor="rgba(138,43,226,0.12)",
            line=dict(color=PURPLE, width=1.5), name="Laplace σ",
        ))
        fig_bll_std.update_layout(**PLOTLY_LAYOUT, height=310,
                                   yaxis_title="Activation Std Dev",
                                   title=dict(text="Bayesian Logistic (Laplace) — Classification Uncertainty", font_color=CYAN))
        st.plotly_chart(fig_bll_std, use_container_width=True)

    # ── PCA 3D feature space ──────────────────────────────────────────────────
    st.markdown("<div class='section-title'>3D PCA Feature Space Projection</div>", unsafe_allow_html=True)

    with st.spinner("Computing 3D PCA projection…"):
        from sklearn.decomposition import PCA
        pca3 = PCA(n_components=3)
        X_pca3 = pca3.fit_transform(spl["X_tr"])
        ev = pca3.explained_variance_ratio_ * 100

        pca_df = pd.DataFrame({
            "PC1": X_pca3[:,0], "PC2": X_pca3[:,1], "PC3": X_pca3[:,2],
            "Direction": ["Up" if y == 1 else "Down/Flat" for y in spl["y_tr_c"]],
        })
        fig_pca = px.scatter_3d(
            pca_df, x="PC1", y="PC2", z="PC3", color="Direction",
            color_discrete_map={"Up": GREEN, "Down/Flat": RED},
            opacity=0.5, height=520,
            labels={"PC1": f"PC1 ({ev[0]:.1f}%)", "PC2": f"PC2 ({ev[1]:.1f}%)", "PC3": f"PC3 ({ev[2]:.1f}%)"},
        )
        fig_pca.update_traces(marker=dict(size=2.5))
        fig_pca.update_layout(
            **PLOTLY_LAYOUT,
            scene=dict(
                bgcolor="#1F2833",
                xaxis=dict(backgroundcolor="#1F2833", gridcolor="#2E3035",
                            showbackground=True, linecolor="#2E3035"),
                yaxis=dict(backgroundcolor="#1F2833", gridcolor="#2E3035",
                            showbackground=True, linecolor="#2E3035"),
                zaxis=dict(backgroundcolor="#1F2833", gridcolor="#2E3035",
                            showbackground=True, linecolor="#2E3035"),
            ),
            title=dict(text=f"3D PCA Projection · Total variance explained = {sum(ev):.1f}%", font_color=CYAN),
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    # ── Hyperparameter sensitivity ────────────────────────────────────────────
    st.markdown("<div class='section-title'>Ridge Regularization — λ Sensitivity</div>", unsafe_allow_html=True)

    with st.spinner("Running λ sweep…"):
        lambdas  = np.logspace(-2, 2, 40)
        rmse_rid, mae_rid = [], []
        for lm in lambdas:
            m = Ridge(lmbda=lm).fit(spl["X_tr_b"], spl["y_tr_r"])
            p = m.predict(spl["X_te_b"])
            rmse_rid.append(np.sqrt(np.mean((spl["y_te_r"] - p)**2)))
            mae_rid.append(np.mean(np.abs(spl["y_te_r"] - p)))

        fig_lam = go.Figure()
        fig_lam.add_trace(go.Scatter(
            x=lambdas, y=rmse_rid, mode="lines+markers",
            marker=dict(color=CYAN, size=5), line=dict(color=CYAN, width=1.8),
            name="Test RMSE",
        ))
        fig_lam.add_trace(go.Scatter(
            x=lambdas, y=mae_rid, mode="lines+markers",
            marker=dict(color=PURPLE, size=5), line=dict(color=PURPLE, width=1.8),
            name="Test MAE",
        ))
        fig_lam.add_vline(x=ridge_lmbda, line_dash="dash", line_color=GREEN, line_width=1.5,
                           annotation_text=f"Current λ={ridge_lmbda}", annotation_font_color=GREEN)
        fig_lam.update_layout(**PLOTLY_LAYOUT, height=340,
                               xaxis_type="log", xaxis_title="λ (log scale)",
                               yaxis_title="Error",
                               title=dict(text="Ridge Regularization Path", font_color=CYAN))
        st.plotly_chart(fig_lam, use_container_width=True)

    # ── KLR probability calibration ───────────────────────────────────────────
    st.markdown("<div class='section-title'>Kernel Logistic — Predicted Probability Distribution</div>", unsafe_allow_html=True)

    probs_klr = R["probs"].get("Kernel Logistic", np.array([]))
    y_te_c    = spl["y_te_c"]

    fig_cal = go.Figure()
    fig_cal.add_trace(go.Histogram(
        x=probs_klr[y_te_c == 1], nbinsx=40,
        marker_color=GREEN, opacity=0.7, name="True Up (Class 1)",
    ))
    fig_cal.add_trace(go.Histogram(
        x=probs_klr[y_te_c == 0], nbinsx=40,
        marker_color=RED, opacity=0.7, name="True Down/Flat (Class 0)",
    ))
    fig_cal.add_vline(x=0.5, line_dash="dash", line_color=CYAN, line_width=1.5,
                       annotation_text="Decision threshold 0.5", annotation_font_color=CYAN)
    fig_cal.update_layout(**PLOTLY_LAYOUT, height=320, barmode="overlay",
                           xaxis_title="Predicted Probability P(Y=1)",
                           yaxis_title="Count",
                           title=dict(text="KLR Predicted Probability Histogram (by True Class)", font_color=CYAN))
    st.plotly_chart(fig_cal, use_container_width=True)

    # ── Residuals analysis ────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Regression Residual Analysis</div>", unsafe_allow_html=True)

    col_r1, col_r2 = st.columns(2)
    for col, nm, pred, clr in [
        (col_r1, "OLS",  R["pred_ols"], CYAN),
        (col_r2, "Huber",R["pred_hub"], PURPLE),
    ]:
        resid = spl["y_te_r"] - pred
        with col:
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(
                x=pred, y=resid, mode="markers",
                marker=dict(color=clr, size=4, opacity=0.5),
                name="Residuals",
                hovertemplate="Pred=%{x:.5f}<br>Resid=%{y:.5f}<extra></extra>",
            ))
            fig_res.add_hline(y=0, line_dash="dash", line_color="#66768A", line_width=1)
            fig_res.update_layout(**PLOTLY_LAYOUT, height=300,
                                   xaxis_title="Predicted", yaxis_title="Residual",
                                   title=dict(text=f"{nm} Residuals vs Fitted", font_color=CYAN))
            st.plotly_chart(fig_res, use_container_width=True)

    # ── Raw metrics summary ───────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Full Classification Metrics Export</div>", unsafe_allow_html=True)
    cm_exp = R["cls_metrics"].copy()
    def _color(val, col):
        if col not in ["ROC-AUC","F1","Accuracy"]: return "#1F2833"
        norm = min(max((val - 0.4) / 0.6, 0), 1)
        r = int(31  + norm * (102 - 31))
        g = int(40  + norm * (252 - 40))
        b = int(51  + norm * (241 - 51))
        return f"rgba({r},{g},{b},0.25)"

    fig_tbl = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in cm_exp.columns],
            fill_color="#1F2833", font=dict(color=CYAN, size=12),
            align="center", line_color="#2E3035", height=36,
        ),
        cells=dict(
            values=[cm_exp[c].map(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
                    for c in cm_exp.columns],
            fill_color=[
                ["#1F2833"] * len(cm_exp) if c not in ["ROC-AUC","F1","Accuracy"]
                else [_color(v, c) for v in cm_exp[c]]
                for c in cm_exp.columns
            ],
            font=dict(color="#C5C6C7", size=12),
            align="center", line_color="#2E3035", height=32,
        ),
    ))
    TABLE_LAYOUT = {
        k: v for k, v in PLOTLY_LAYOUT.items()
        if k not in ("xaxis", "yaxis", "margin")
    }

    fig_tbl.update_layout(
        **TABLE_LAYOUT,
        height=320,
        margin=dict(l=0, r=0, t=10, b=0)
    )

    st.plotly_chart(fig_tbl, use_container_width=True)

    csv = R["cls_metrics"].to_csv(index=False).encode()
    st.download_button(
        "⬇ Download Classification Metrics (CSV)",
        csv,
        "classification_metrics.csv",
        "text/csv",
        use_container_width=True
    )
