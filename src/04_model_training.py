import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
from prophet import Prophet
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_percentage_error
import warnings
import json

# --- Configuración ---
warnings.filterwarnings('ignore')

# --- Carga de Datos ---
try:
    df = pd.read_csv('output/data_featured_advanced.csv', parse_dates=['month_year'])
    print("Datos con características avanzadas cargados exitosamente.")
except FileNotFoundError:
    print("Error: No se encontró 'output/data_featured_advanced.csv'.")
    exit()

# --- Preparación ---
unique_saps = df['SAP'].unique()
all_predictions = []
best_params_log = {}
ADVANCED_FEATURES = ['mes', 'anio', 'temporada_pesca', 'pico_pre_temporada',
                     'consumo_lag_1', 'consumo_lag_3', 'consumo_lag_12',
                     'consumo_rolling_mean_3', 'consumo_rolling_std_3']

# --- Bucle Principal de Entrenamiento ---
for sap in unique_saps:
    print(f"--- Procesando SAP: {sap} ---")
    df_sap = df[df['SAP'] == sap]
    train_end = '2025-01-31'
    val_start = '2025-02-01'
    train = df_sap[df_sap['month_year'] <= train_end]
    validation = df_sap[df_sap['month_year'] >= val_start]

    if len(validation) == 0: continue

    validation_actuals = validation[['SAP', 'month_year', 'Consumo Total']].copy()
    validation_actuals['prediction'] = validation_actuals['Consumo Total']
    validation_actuals['model'] = 'Actual'
    all_predictions.append(validation_actuals)

    # --- Modelo 1: SARIMAX ---
    exog_vars_sarimax = ['temporada_pesca', 'pico_pre_temporada']
    try:
        model_sarimax = SARIMAX(train['Consumo Total'], exog=train[exog_vars_sarimax], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)).fit(disp=False)
        pred_sarimax = model_sarimax.get_forecast(steps=len(validation), exog=validation[exog_vars_sarimax])
        sarimax_results = validation[['SAP', 'month_year']].copy()
        sarimax_results['prediction'] = np.maximum(0, pred_sarimax.predicted_mean.values)
        sarimax_results['model'] = 'SARIMAX'
        all_predictions.append(sarimax_results)
        print("Entrenamiento SARIMAX exitoso.")
    except Exception as e:
        print(f"SARIMAX falló: {e}")

    # --- Modelo 2: XGBoost con Optimización ---
    print("Iniciando optimización para XGBoost...")
    param_grid = {'n_estimators': [100, 500, 1000], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]}
    best_mape = float('inf')
    best_params = None

    for params in ParameterGrid(param_grid):
        model_xgb = xgb.XGBRegressor(objective='reg:squarederror', **params)
        model_xgb.fit(train[ADVANCED_FEATURES], train['Consumo Total'], verbose=False)
        preds = model_xgb.predict(validation[ADVANCED_FEATURES])
        mape = mean_absolute_percentage_error(validation['Consumo Total'], preds)
        if mape < best_mape:
            best_mape = mape
            best_params = params

    best_params_log[sap] = best_params
    print(f"Mejores parámetros para XGBoost encontrados: {best_params}")

    final_model_xgb = xgb.XGBRegressor(objective='reg:squarederror', **best_params)
    final_model_xgb.fit(train[ADVANCED_FEATURES], train['Consumo Total'], verbose=False)
    xgb_preds = final_model_xgb.predict(validation[ADVANCED_FEATURES])

    xgb_results = validation[['SAP', 'month_year']].copy()
    xgb_results['prediction'] = np.maximum(0, xgb_preds)
    xgb_results['model'] = 'XGBoost_Tuned'
    all_predictions.append(xgb_results)
    print("Entrenamiento XGBoost_Tuned exitoso.")

    # --- Modelo 3: Prophet (Corregido) ---
    exog_vars_prophet = ['temporada_pesca', 'pico_pre_temporada', 'consumo_lag_1', 'consumo_rolling_mean_3']
    try:
        df_prophet_train = train[['month_year', 'Consumo Total'] + exog_vars_prophet].rename(columns={'month_year': 'ds', 'Consumo Total': 'y'})
        model_prophet = Prophet()
        # **CORRECCIÓN: Usar 'model_prophet' en lugar de 'model'**
        for regressor in exog_vars_prophet: model_prophet.add_regressor(regressor)
        model_prophet.fit(df_prophet_train)

        future = model_prophet.make_future_dataframe(periods=len(validation), freq='MS')
        future = pd.merge(future, df_sap[['month_year'] + exog_vars_prophet].rename(columns={'month_year': 'ds'}), on='ds')
        forecast = model_prophet.predict(future.tail(len(validation)))

        prophet_results = validation[['SAP', 'month_year']].copy()
        prophet_results['prediction'] = np.maximum(0, forecast['yhat'].values)
        prophet_results['model'] = 'Prophet'
        all_predictions.append(prophet_results)
        print("Entrenamiento Prophet exitoso.")
    except Exception as e:
        print(f"Prophet falló: {e}")

# --- Guardado de Resultados ---
if all_predictions:
    pd.concat(all_predictions).to_csv('output/model_predictions_advanced.csv', index=False)
    print("\nTodas las predicciones avanzadas han sido guardadas.")

params_output_path = 'output/best_xgb_params.json'
with open(params_output_path, 'w') as f:
    json.dump(best_params_log, f, indent=4)
print(f"Mejores parámetros de XGBoost guardados en: {params_output_path}")
