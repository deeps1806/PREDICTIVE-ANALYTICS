# Predictive Analytics — Sales Forecasting

Forecasts future daily sales using 5 years of historical data and three machine learning models.

## Models Used
- Linear Regression
- Random Forest
- Gradient Boosting ← best performer

## Results
| Model | R² | MAPE |
|---|---|---|
| Linear Regression | 0.7820 | 5.03% |
| Random Forest | 0.7898 | 4.95% |
| Gradient Boosting | 0.8070 | 4.86% |

## Project Structure
```
predictive-analytics/
├── sales_historical.csv       # Raw historical sales data (1827 days)
├── predictive_analytics.py    # Main ML script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
```

## How to Run
```bash
pip install -r requirements.txt
python predictive_analytics.py
```

## Output Files Generated
| File | Description |
|---|---|
| 01_sales_history.png | Full 5-year sales trend |
| 02_actual_vs_predicted.png | Model predictions vs actual |
| 03_forecast_30days.png | 30-day future forecast |
| 04_model_comparison.png | MAE / RMSE / MAPE comparison |
| 05_feature_importance.png | Most influential features |
| 06_residual_analysis.png | Error distribution analysis |
| 07_seasonal_patterns.png | Monthly & weekly patterns |
| forecast_30days.csv | 30-day forecast values |
| model_results.csv | All model accuracy scores |
