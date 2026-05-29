# ══════════════════════════════════════════════════════════
#   PREDICTIVE ANALYTICS — Sales Forecasting
#   Models: Linear Regression + Random Forest + 30-day Forecast
# ══════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, mean_absolute_percentage_error)
from datetime import timedelta

plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'#FAFAFA',
                     'axes.grid':True,'grid.alpha':0.3,'font.size':11})

print("=" * 60)
print("   PREDICTIVE ANALYTICS — Sales Forecasting System")
print("=" * 60)

# ── STEP 1: Load & Inspect ─────────────────────────────────
df = pd.read_csv('sales_historical.csv', parse_dates=['date'])
df.sort_values('date', inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"\nDataset: {len(df)} rows  |  {df['date'].min().date()} → {df['date'].max().date()}")
print(df.describe().round(1).to_string())

# ── STEP 2: Data Cleaning ──────────────────────────────────
print("\n── Data Cleaning ──")
print(f"Missing values: {df.isnull().sum().sum()}")
df.fillna(df.median(numeric_only=True), inplace=True)

# Remove outliers using IQR
Q1, Q3 = df['sales'].quantile(0.25), df['sales'].quantile(0.75)
IQR = Q3 - Q1
before = len(df)
df = df[(df['sales'] >= Q1 - 1.5*IQR) & (df['sales'] <= Q3 + 1.5*IQR)]
print(f"Outliers removed: {before - len(df)} rows")
df.reset_index(drop=True, inplace=True)

# ── STEP 3: Feature Engineering ───────────────────────────
print("\n── Feature Engineering ──")
df['day_of_week']  = df['date'].dt.dayofweek
df['day_of_year']  = df['date'].dt.dayofyear
df['quarter']      = df['date'].dt.quarter
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

# Lag features (previous days sales)
df['lag_7']  = df['sales'].shift(7)
df['lag_14'] = df['sales'].shift(14)
df['lag_30'] = df['sales'].shift(30)

# Rolling averages
df['rolling_7']  = df['sales'].shift(1).rolling(7).mean()
df['rolling_30'] = df['sales'].shift(1).rolling(30).mean()

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Features created. Final shape: {df.shape}")

# ── STEP 4: Train/Test Split (80/20 time-based) ───────────
features = ['days_since_start','month','day_of_week','day_of_year',
            'quarter','year','week','is_weekend','is_holiday_season',
            'lag_7','lag_14','lag_30','rolling_7','rolling_30']

X = df[features]
y = df['sales']

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test = df['date'].iloc[split_idx:]

print(f"\nTrain: {len(X_train)} rows  |  Test: {len(X_test)} rows")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── STEP 5: Train Models ───────────────────────────────────
print("\n── Training Models ──")

models = {
    'Linear Regression':    LinearRegression(),
    'Random Forest':        RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting':    GradientBoostingRegressor(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    if name == 'Linear Regression':
        model.fit(X_train_sc, y_train)
        preds = model.predict(X_test_sc)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)
    mape  = mean_absolute_percentage_error(y_test, preds) * 100

    results[name] = {'model': model, 'preds': preds,
                     'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
    print(f"  {name:25s}  MAE=${mae:,.0f}  RMSE=${rmse:,.0f}  R²={r2:.4f}  MAPE={mape:.2f}%")

best_name = min(results, key=lambda k: results[k]['RMSE'])
print(f"\n  Best model: {best_name}")

# ── STEP 6: 30-Day Future Forecast ────────────────────────
print("\n── Generating 30-Day Forecast ──")
best_model = results[best_name]['model']

last_date  = df['date'].max()
last_known = df.copy()

forecast_rows = []
future_sales  = []

for i in range(1, 31):
    fd = last_date + timedelta(days=i)
    lag7  = last_known['sales'].iloc[-7]  if len(last_known) >= 7  else last_known['sales'].mean()
    lag14 = last_known['sales'].iloc[-14] if len(last_known) >= 14 else last_known['sales'].mean()
    lag30 = last_known['sales'].iloc[-30] if len(last_known) >= 30 else last_known['sales'].mean()
    r7    = last_known['sales'].iloc[-7:].mean()
    r30   = last_known['sales'].iloc[-30:].mean()

    row = {
        'days_since_start': (fd - df['date'].min()).days,
        'month':            fd.month,
        'day_of_week':      fd.weekday(),
        'day_of_year':      fd.timetuple().tm_yday,
        'quarter':          (fd.month - 1) // 3 + 1,
        'year':             fd.year,
        'week':             fd.isocalendar()[1],
        'is_weekend':       1 if fd.weekday() >= 5 else 0,
        'is_holiday_season':1 if (fd.month == 12 and fd.day >= 20) else 0,
        'lag_7': lag7, 'lag_14': lag14, 'lag_30': lag30,
        'rolling_7': r7, 'rolling_30': r30,
    }
    feat_row = pd.DataFrame([row])[features]

    if best_name == 'Linear Regression':
        pred = best_model.predict(scaler.transform(feat_row))[0]
    else:
        pred = best_model.predict(feat_row)[0]

    pred = max(pred, 5000)
    future_sales.append(pred)
    forecast_rows.append({'date': fd, 'forecasted_sales': round(pred, 2)})

    new_row = pd.DataFrame([{**row, 'date': fd, 'sales': pred}])
    last_known = pd.concat([last_known, new_row], ignore_index=True)

forecast_df = pd.DataFrame(forecast_rows)
print(f"  30-day avg forecast: ${np.mean(future_sales):,.0f}")
print(f"  30-day total forecast: ${sum(future_sales):,.0f}")

# ══════════════════════════════════════════════════════════
#   CHARTS
# ══════════════════════════════════════════════════════════

# ── Chart 1: Full Sales History ────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
monthly = df.groupby(df['date'].dt.to_period('M'))['sales'].mean()
monthly.index = monthly.index.to_timestamp()
ax.plot(monthly.index, monthly.values, color='#378ADD', linewidth=2, label='Monthly avg sales')
ax.fill_between(monthly.index, monthly.values, alpha=0.15, color='#378ADD')
ax.set_title('Historical Sales Trend (2020–2024)', fontsize=14, fontweight='bold')
ax.set_ylabel('Sales ($)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=30)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}k'))
plt.legend(); plt.tight_layout()
plt.savefig('01_sales_history.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: 01_sales_history.png")

# ── Chart 2: Actual vs Predicted ──────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
colors_m = {'Linear Regression':'#D85A30','Random Forest':'#1D9E75','Gradient Boosting':'#7F77DD'}
for ax, (name, res) in zip(axes, results.items()):
    ax.plot(dates_test.values, y_test.values, color='#378ADD', linewidth=1.5, label='Actual', alpha=0.8)
    ax.plot(dates_test.values, res['preds'],   color=colors_m[name], linewidth=1.5,
            label=f'Predicted  R²={res["R2"]:.3f}  MAPE={res["MAPE"]:.1f}%', linestyle='--')
    ax.set_title(name, fontweight='bold')
    ax.set_ylabel('Sales ($)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}k'))
    ax.legend(loc='upper left', fontsize=10)
plt.suptitle('Actual vs Predicted Sales — All Models', fontsize=14, fontweight='bold', y=1.01)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('02_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 02_actual_vs_predicted.png")

# ── Chart 3: 30-Day Forecast ──────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
hist_tail = df[df['date'] >= df['date'].max() - timedelta(days=90)]
ax.plot(hist_tail['date'], hist_tail['sales'], color='#378ADD', linewidth=2, label='Historical (last 90 days)')
ax.plot(forecast_df['date'], forecast_df['forecasted_sales'],
        color='#D85A30', linewidth=2.5, linestyle='--', label='30-day forecast', marker='o', markersize=4)
ax.axvline(last_date, color='gray', linestyle=':', linewidth=1.5, label='Forecast start')
upper = [v * 1.07 for v in forecast_df['forecasted_sales']]
lower = [v * 0.93 for v in forecast_df['forecasted_sales']]
ax.fill_between(forecast_df['date'], lower, upper, alpha=0.18, color='#D85A30', label='±7% confidence band')
ax.set_title(f'30-Day Sales Forecast ({best_name})', fontsize=14, fontweight='bold')
ax.set_ylabel('Sales ($)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}k'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
plt.xticks(rotation=30)
plt.legend(); plt.tight_layout()
plt.savefig('03_forecast_30days.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 03_forecast_30days.png")

# ── Chart 4: Model Comparison ─────────────────────────────
metrics = ['MAE','RMSE','MAPE']
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
bar_colors = ['#378ADD','#1D9E75','#D85A30']
for ax, metric in zip(axes, metrics):
    vals  = [results[m][metric] for m in results]
    names = list(results.keys())
    bars  = ax.bar(names, vals, color=bar_colors, edgecolor='white', width=0.55)
    ax.set_title(metric, fontweight='bold')
    ax.tick_params(axis='x', rotation=20)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.01,
                f'{val:,.0f}' if metric!='MAPE' else f'{val:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='500')
plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('04_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 04_model_comparison.png")

# ── Chart 5: Feature Importance ───────────────────────────
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
colors_fi = ['#7F77DD' if v > importances.median() else '#B4B2A9' for v in importances]
importances.plot(kind='barh', ax=ax, color=colors_fi, edgecolor='white')
ax.set_title('Feature Importance — Random Forest', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('05_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 05_feature_importance.png")

# ── Chart 6: Residuals ────────────────────────────────────
best_preds = results[best_name]['preds']
residuals  = y_test.values - best_preds
fig, axes  = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(best_preds, residuals, alpha=0.4, color='#378ADD', s=20, edgecolors='none')
axes[0].axhline(0, color='#D85A30', linewidth=1.5, linestyle='--')
axes[0].set_title('Residuals vs Predicted', fontweight='bold')
axes[0].set_xlabel('Predicted Sales ($)'); axes[0].set_ylabel('Residual ($)')
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}k'))
sns.histplot(residuals, kde=True, ax=axes[1], color='#7F77DD', edgecolor='white')
axes[1].set_title('Residual Distribution', fontweight='bold')
axes[1].set_xlabel('Residual ($)')
plt.suptitle(f'Residual Analysis — {best_name}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('06_residual_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 06_residual_analysis.png")

# ── Chart 7: Seasonal Pattern ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
month_avg = df.groupby('month')['sales'].mean()
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
bar_c = ['#D85A30' if v == month_avg.max() else '#378ADD' for v in month_avg]
axes[0].bar(month_names, month_avg.values, color=bar_c, edgecolor='white', width=0.65)
axes[0].set_title('Average Sales by Month', fontweight='bold')
axes[0].set_ylabel('Avg Sales ($)')
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}k'))

dow_avg   = df.groupby('day_of_week')['sales'].mean()
dow_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
bar_c2    = ['#888780' if i >= 5 else '#1D9E75' for i in range(7)]
axes[1].bar(dow_names, dow_avg.values, color=bar_c2, edgecolor='white', width=0.65)
axes[1].set_title('Average Sales by Day of Week', fontweight='bold')
axes[1].set_ylabel('Avg Sales ($)')
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}k'))
plt.suptitle('Seasonal & Weekly Patterns', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('07_seasonal_patterns.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 07_seasonal_patterns.png")

# ── STEP 7: Export Results ────────────────────────────────
forecast_df.to_csv('forecast_30days.csv', index=False)

results_rows = []
for name, res in results.items():
    results_rows.append({'Model': name, 'MAE': round(res['MAE'],2),
                         'RMSE': round(res['RMSE'],2), 'R2': round(res['R2'],4),
                         'MAPE_%': round(res['MAPE'],2)})
pd.DataFrame(results_rows).to_csv('model_results.csv', index=False)

print("\n" + "="*60)
print("  ALL DONE — Files saved:")
print("  Data    : sales_historical.csv")
print("  Script  : predictive_analytics.py")
print("  Charts  : 01_sales_history.png")
print("            02_actual_vs_predicted.png")
print("            03_forecast_30days.png")
print("            04_model_comparison.png")
print("            05_feature_importance.png")
print("            06_residual_analysis.png")
print("            07_seasonal_patterns.png")
print("  Outputs : forecast_30days.csv")
print("            model_results.csv")
print("="*60)
print("\n── Final Model Scores ──")
for name, res in results.items():
    print(f"  {name:25s}  R²={res['R2']:.4f}  MAPE={res['MAPE']:.2f}%")
print(f"\n  Best: {best_name}")