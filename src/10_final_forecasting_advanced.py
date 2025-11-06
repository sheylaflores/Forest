import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
from prophet import Prophet
import warnings
import json

# --- Configuración ---
warnings.filterwarnings('ignore')

# --- Carga de Datos ---
try:
    df = pd.read_csv('output/data_featured_advanced.csv', parse_dates=['month_year'])
    best_models_df = pd.read_csv('output/best_models.csv')
    with open('output/best_xgb_params.json', 'r') as f:
        best_xgb_params = json.load(f)
    print("Datos y lista de mejores modelos avanzados cargados.")
except FileNotFoundError as e:
    print(f"Error: No se encontró un archivo requerido. {e}")
    exit()

# --- Preparación ---
final_forecasts = []
ADVANCED_FEATURES = ['mes', 'anio', 'temporada_pesca', 'pico_pre_temporada',
                     'consumo_lag_1', 'consumo_lag_3', 'consumo_lag_12',
                     'consumo_rolling_mean_3', 'consumo_rolling_std_3']

# --- Bucle Principal de Pronóstico ---
for _, row in best_models_df.iterrows():
    sap = row['SAP']
    best_model_name = row['Model']
    print(f"--- Pronosticando para SAP: {sap} con el mejor modelo: {best_model_name} ---")

    df_sap = df[df['SAP'] == sap]
    history_len = len(df_sap)

    future_dates = pd.date_range(start='2025-11-01', periods=6, freq='MS')
    future_df = pd.DataFrame({'month_year': future_dates})
    future_df['mes'] = future_df['month_year'].dt.month
    future_df['anio'] = future_df['month_year'].dt.year
    future_df['temporada_pesca'] = future_df['month_year'].dt.month.isin([4, 5, 6, 11, 12]).astype(int)
    future_df['pico_pre_temporada'] = future_df['month_year'].dt.month.isin([2, 3, 9, 10]).astype(int)

    # --- **CORRECCIÓN: Generación Robusta de Características Futuras** ---
    future_df['consumo_lag_1'] = df_sap['Consumo Total'].iloc[-1] if history_len >= 1 else 0
    future_df['consumo_lag_3'] = df_sap['Consumo Total'].iloc[-3] if history_len >= 3 else (df_sap['Consumo Total'].iloc[-1] if history_len >= 1 else 0)
    future_df['consumo_lag_12'] = df_sap['Consumo Total'].iloc[-12] if history_len >= 12 else (df_sap['Consumo Total'].iloc[-1] if history_len >= 1 else 0)

    last_3_months = df_sap['Consumo Total'].tail(3)
    future_df['consumo_rolling_mean_3'] = last_3_months.mean()
    future_df['consumo_rolling_std_3'] = last_3_months.std()
    future_df.fillna(0, inplace=True)

    forecast_values = []

    try:
        if best_model_name == 'SARIMAX':
            model = SARIMAX(df_sap['Consumo Total'], exog=df_sap[['temporada_pesca', 'pico_pre_temporada']], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)).fit(disp=False)
            forecast = model.get_forecast(steps=6, exog=future_df[['temporada_pesca', 'pico_pre_temporada']])
            forecast_values = forecast.predicted_mean.values

        elif best_model_name == 'XGBoost_Tuned':
            params = best_xgb_params.get(sap)
            if not params: raise ValueError(f"No se encontraron parámetros para SAP {sap}.")
            model = xgb.XGBRegressor(objective='reg:squarederror', **params)
            model.fit(df_sap[ADVANCED_FEATURES], df_sap['Consumo Total'], verbose=False)
            forecast_values = model.predict(future_df[ADVANCED_FEATURES])

        elif best_model_name == 'Prophet':
            exog_vars_prophet = ['temporada_pesca', 'pico_pre_temporada', 'consumo_lag_1', 'consumo_rolling_mean_3']
            df_prophet_train = df_sap[['month_year', 'Consumo Total'] + exog_vars_prophet].rename(columns={'month_year': 'ds', 'Consumo Total': 'y'})
            model = Prophet()
            for regressor in exog_vars_prophet: model.add_regressor(regressor)
            model.fit(df_prophet_train)
            future_prophet = future_df[['month_year'] + exog_vars_prophet].rename(columns={'month_year': 'ds'})
            forecast_df = model.predict(future_prophet)
            forecast_values = forecast_df['yhat'].values

        forecast_values = np.round(np.maximum(0, forecast_values)).astype(int)

        temp_forecast_df = pd.DataFrame({'SAP': sap, 'month_year': future_dates, 'predicted_demand': forecast_values})
        final_forecasts.append(temp_forecast_df)
        print("Pronóstico generado exitosamente.")

    except Exception as e:
        print(f"Falló la generación de pronóstico para SAP {sap}: {e}")

# --- Guardado de Resultados ---
if final_forecasts:
    final_forecasts_df = pd.concat(final_forecasts)
    output_path = 'output/final_predictions_advanced.xlsx'
    pivot_forecast = final_forecasts_df.pivot(index='SAP', columns='month_year', values='predicted_demand')
    pivot_forecast.to_excel(output_path)
    print(f"\nTodos los pronósticos finales mejorados han sido guardados en: {output_path}")
else:
    print("\nNo se generaron pronósticos finales.")
