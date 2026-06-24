"""
===============================================================================
Stock Market Analysis Using Statistical Methods and Machine Learning
Unit 6 AS3 Project #1 | Python Programming | Kean University

Author : Joseph Barragan
Date   : June 15, 2026
Course : Python Programming

Description:
    Generates synthetic stock price data for NVDA, AMZN, MSFT, and AAPL
    via Geometric Brownian Motion, then performs:
      - Descriptive statistics & return distribution visualisation
      - Pairwise correlation analysis
      - 30-day rolling annualised volatility
      - One-sample t-test for NVDA mean return
      - OLS Linear Regression for next-day return prediction
      - Random Forest Classifier for directional price prediction

Usage:
    pip install pandas numpy matplotlib seaborn scikit-learn scipy
    python stock_analysis.py

Output:
    - Figures saved to ./figures/
    - Numerical metrics printed to console
===============================================================================
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────────
SEED       = 42
N_DAYS     = 1259          # trading days 2020-01-02 to 2024-12-31
LAGS       = 5             # lagged features for ML models
TEST_FRAC  = 0.20
FIGURES_DIR = './figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA GENERATION  (Geometric Brownian Motion)
# ══════════════════════════════════════════════════════════════════════════════

def gbm(S0: float, mu: float, sigma: float, n: int = N_DAYS) -> np.ndarray:
    """
    Simulate n daily closing prices via Geometric Brownian Motion.

    Parameters
    ----------
    S0    : Initial price
    mu    : Annual drift (e.g. 0.72 for 72% annualised return)
    sigma : Annual volatility
    n     : Number of trading days

    Returns
    -------
    np.ndarray of shape (n,)
    """
    dt = 1 / 252
    log_returns = np.random.normal(
        (mu - 0.5 * sigma ** 2) * dt,
        sigma * np.sqrt(dt),
        n
    )
    prices = S0 * np.exp(np.cumsum(log_returns))
    return np.concatenate([[S0], prices])[:n]


# Calibrated to approximate historical statistics for each ticker
TICKER_PARAMS = {
    'NVDA': dict(S0=50,  mu=0.72, sigma=0.52),
    'AMZN': dict(S0=84,  mu=0.24, sigma=0.30),
    'MSFT': dict(S0=160, mu=0.32, sigma=0.28),
    'AAPL': dict(S0=75,  mu=0.35, sigma=0.30),
}

dates  = pd.bdate_range('2020-01-02', periods=N_DAYS)
prices = pd.DataFrame(
    {t: gbm(**p) for t, p in TICKER_PARAMS.items()},
    index=dates
)

# Daily simple returns
returns = prices.pct_change().dropna()

print(f"Dataset: {len(prices)} trading days  |  {len(returns)} return observations")
print(f"Period : {dates[0].date()} to {dates[-1].date()}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def annualised_stats(ret: pd.DataFrame) -> pd.DataFrame:
    """Return annual return, volatility, and Sharpe ratio for each column."""
    ann_ret = ret.mean() * 252
    ann_vol = ret.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol
    total   = (prices.iloc[-1] / prices.iloc[0]) - 1
    return pd.DataFrame({
        'Ann. Return'    : ann_ret,
        'Ann. Volatility': ann_vol,
        'Sharpe Ratio'   : sharpe,
        'Total Return'   : total,
    })

stats_df = annualised_stats(returns)
print("=== Annualised Statistics ===")
print(stats_df.to_string(float_format='{:.4f}'.format))
print()

# One-sample t-test: NVDA mean return vs 0
t_stat, p_val = stats.ttest_1samp(returns['NVDA'], 0)
print(f"NVDA t-test:  t = {t_stat:.4f},  p = {p_val:.2e}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 3.  VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {'NVDA': '#76b900', 'AMZN': '#FF9900', 'MSFT': '#00a4ef', 'AAPL': '#A2AAAD'}


# ── Fig 1: Normalised cumulative price performance ────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
for t, c in COLORS.items():
    ax.plot(prices.index, prices[t] / prices[t].iloc[0] * 100,
            label=t, color=c, linewidth=1.8)
ax.set_title('Normalised Price Performance (Base = 100, 2020–2024)',
             fontsize=11, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Normalised Price')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig1_price_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_price_performance.png")


# ── Fig 2: Return distributions ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
for ax, (t, c) in zip(axes.flat, COLORS.items()):
    r = returns[t]
    ax.hist(r, bins=60, density=True, alpha=0.45, color=c, edgecolor='white')
    x   = np.linspace(r.min(), r.max(), 300)
    ax.plot(x, stats.norm.pdf(x, r.mean(), r.std()), 'k--', lw=1.5, label='Normal fit')
    ax.set_title(f'{t}  μ={r.mean():.4f}  σ={r.std():.4f}', fontsize=9)
    ax.set_xlabel('Daily Return')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.suptitle('Daily Return Distributions (2020–2024)', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig2_return_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_return_distributions.png")


# ── Fig 3: Correlation heatmap ────────────────────────────────────────────────
corr_matrix = returns.corr()
fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=-1, vmax=1, ax=ax, linewidths=0.5, annot_kws={'size': 10})
ax.set_title('Pairwise Return Correlation Matrix', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig3_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3_correlation_heatmap.png")
print(f"\nCorrelation matrix:\n{corr_matrix.to_string(float_format='{:.4f}'.format)}\n")


# ── Fig 4: Rolling 30-day annualised volatility ───────────────────────────────
rolling_vol = returns.rolling(30).std() * np.sqrt(252)
fig, ax = plt.subplots(figsize=(9, 4.5))
for t, c in COLORS.items():
    ax.plot(rolling_vol.index, rolling_vol[t], label=t, color=c, linewidth=1.5)
ax.set_title('30-Day Rolling Annualised Volatility', fontsize=11, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Annualised Volatility')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig4_rolling_volatility.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4_rolling_volatility.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MACHINE LEARNING — LINEAR REGRESSION (next-day return prediction)
# ══════════════════════════════════════════════════════════════════════════════

def build_lag_features(series: np.ndarray, lags: int, predict_return: bool = True):
    """
    Construct lagged feature matrix from a 1D return series.

    Returns X (n_samples, lags) and y (n_samples,).
    predict_return=True  → predict next-day return (regression)
    predict_return=False → predict next-day direction 1/0 (classification)
    """
    X, y = [], []
    for i in range(lags, len(series) - 1):
        X.append(series[i - lags:i])
        if predict_return:
            y.append(series[i + 1])
        else:
            y.append(1 if series[i + 1] > 0 else 0)
    return np.array(X), np.array(y)


nvda_ret = returns['NVDA'].values

# --- Regression ---
X_reg, y_reg = build_lag_features(nvda_ret, LAGS, predict_return=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_reg, y_reg, test_size=TEST_FRAC, shuffle=False
)

lr_model  = LinearRegression()
lr_model.fit(X_tr, y_tr)
y_pred_lr = lr_model.predict(X_te)

rmse = np.sqrt(mean_squared_error(y_te, y_pred_lr))
r2   = r2_score(y_te, y_pred_lr)
print(f"Linear Regression  →  RMSE = {rmse:.5f},  R² = {r2:.4f}")

# ── Fig 5: LR prediction vs actual ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(y_te[:200], label='Actual',         color='#333333', lw=1.3)
ax.plot(y_pred_lr[:200], label='Predicted', color='#76b900', lw=1.3, ls='--')
ax.set_title('Linear Regression – NVDA Next-Day Return Prediction\n(Test Set, first 200 observations)',
             fontsize=10, fontweight='bold')
ax.set_xlabel('Observation Index')
ax.set_ylabel('Daily Return')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig5_lr_prediction.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5_lr_prediction.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MACHINE LEARNING — RANDOM FOREST CLASSIFIER (directional prediction)
# ══════════════════════════════════════════════════════════════════════════════

X_clf, y_clf = build_lag_features(nvda_ret, LAGS, predict_return=False)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    X_clf, y_clf, test_size=TEST_FRAC, shuffle=False
)

scaler     = StandardScaler()
Xc_tr_s    = scaler.fit_transform(Xc_tr)
Xc_te_s    = scaler.transform(Xc_te)

rf_model   = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=SEED)
rf_model.fit(Xc_tr_s, yc_tr)
yc_pred    = rf_model.predict(Xc_te_s)

accuracy   = accuracy_score(yc_te, yc_pred)
cm         = confusion_matrix(yc_te, yc_pred)
print(f"\nRandom Forest Classifier  →  Accuracy = {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Confusion matrix:\n{cm}")

# ── Fig 6: Confusion matrix + feature importances ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'], ax=axes[0])
axes[0].set_title('Confusion Matrix (Random Forest)', fontsize=10, fontweight='bold')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

fi = rf_model.feature_importances_
axes[1].bar([f'Lag {i+1}' for i in range(LAGS)], fi,
            color='#76b900', edgecolor='k', linewidth=0.5)
axes[1].set_title('Feature Importances', fontsize=10, fontweight='bold')
axes[1].set_xlabel('Feature')
axes[1].set_ylabel('Importance')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig6_rf_classifier.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig6_rf_classifier.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  EXPORT METRICS
# ══════════════════════════════════════════════════════════════════════════════

metrics = {
    'date_range'       : ['2020-01-02', '2024-12-31'],
    'n_observations'   : len(returns),
    'ann_return'       : stats_df['Ann. Return'].to_dict(),
    'ann_volatility'   : stats_df['Ann. Volatility'].to_dict(),
    'sharpe_ratio'     : stats_df['Sharpe Ratio'].to_dict(),
    'total_return'     : stats_df['Total Return'].to_dict(),
    'nvda_ttest_t'     : round(t_stat, 6),
    'nvda_ttest_p'     : float(f'{p_val:.2e}'),
    'lr_rmse'          : round(rmse, 6),
    'lr_r2'            : round(r2, 6),
    'rf_accuracy'      : round(accuracy, 6),
}
with open('metrics.json', 'w') as fh:
    json.dump(metrics, fh, indent=2)

print("\nAll figures saved to ./figures/")
print("Metrics exported to metrics.json")
print("\n=== Run complete ===")
