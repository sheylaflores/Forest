import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
from prophet import Prophet
import warnings

# Ignorar advertencias
warnings.filterwarnings('ignore')

# Cargar los datos y la lista de mejores modelos
try:
    df = pd.read_csv('output/data_featured.csv')
    df['month_year'] = pd.to_datetime(df['month_year'])
    best_models_df = pd.read_csv('output/best_models.csv')
    print("Datos y lista de mejores modelos cargados.")
except FileNotFoundError as e:
    print(f"Error: No se encontró un archivo requerido. {e}")
    exit()

# Contenedor para los pronósticos finales
final_forecasts = []

# Iterar sobre cada SAP en la lista de mejores modelos
for _, row in best_models_df.iterrows():
    sap = row['SAP']
    best_model_name = row['Model']
    print(f"--- Pronosticando para SAP: {sap} con el mejor modelo: {best_model_name} ---")

    # Preparar datos completos para el SAP actual
    df_sap_original = df[df['SAP'] == sap]
    start_date = df_sap_original['month_year'].min()
    end_date = pd.to_datetime('2025-10-01')
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')

    df_sap = df_sap_original.set_index('month_year').reindex(date_range).reset_index()
    df_sap.rename(columns={'index': 'month_year'}, inplace=True)

    df_sap['Consumo Total'] = df_sap['Consumo Total'].fillna(0)
    df_sap['SAP'] = sap
    df_sap.ffill(inplace=True)
    df_sap.bfill(inplace=True)

    # Re-generar características para el rango de fechas completo
    df_sap['mes'] = df_sap['month_year'].dt.month
    df_sap['anio'] = df_sap['month_year'].dt.year
    df_sap['temporada_pesca'] = df_sap['month_year'].dt.month.isin([4, 5, 6, 11, 12]).astype(int)
    df_sap['pico_pre_temporada'] = df_sap['month_year'].dt.month.isin([2, 3, 9, 10]).astype(int)

    # Definir el futuro y las variables exógenas
    future_dates = pd.date_range(start='2025-11-01', periods=6, freq='MS')
    future_df = pd.DataFrame({'month_year': future_dates})
    future_df['mes'] = future_df['month_year'].dt.month
    future_df['anio'] = future_df['month_year'].dt.year
    future_df['temporada_pesca'] = future_df['month_year'].dt.month.isin([4, 5, 6, 11, 12]).astype(int)
    future_df['pico_pre_temporada'] = future_df['month_year'].dt.month.isin([2, 3, 9, 10]).astype(int)

    exog_vars = ['temporada_pesca', 'pico_pre_temporada']
    forecast_values = []

    # Re-entrenar el mejor modelo y predecir
    try:
        if best_model_name == 'SARIMAX':
            model = SARIMAX(df_sap['Consumo Total'], exog=df_sap[exog_vars], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
            results = model.fit(disp=False)
            forecast = results.get_forecast(steps=6, exog=future_df[exog_vars])
            forecast_values = forecast.predicted_mean.values

        elif best_model_name == 'XGBoost':
            features = ['mes', 'anio'] + exog_vars
            target = 'Consumo Total'
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, early_stopping_rounds=50, eval_metric="rmse")
            model.fit(df_sap[features], df_sap[target], eval_set=[(df_sap[features], df_sap[target])], verbose=False)
            forecast_values = model.predict(future_df[features])

        elif best_model_name == 'Prophet':
            df_prophet_train = df_sap[['month_year', 'Consumo Total'] + exog_vars].rename(columns={'month_year': 'ds', 'Consumo Total': 'y'})
            model = Prophet()
            for regressor in exog_vars:
                model.add_regressor(regressor)
            model.fit(df_prophet_train)

            future_prophet = future_df[['month_year'] + exog_vars].rename(columns={'month_year': 'ds'})
            forecast_df = model.predict(future_prophet)
            forecast_values = forecast_df['yhat'].values

        # Asegurar que las predicciones no sean negativas y redondearlas
        forecast_values = np.maximum(0, forecast_values)
        forecast_values = np.round(forecast_values).astype(int)

        # Crear DataFrame de resultados
        temp_forecast_df = pd.DataFrame({
            'SAP': sap,
            'month_year': future_dates,
            'predicted_demand': forecast_values
        })
        final_forecasts.append(temp_forecast_df)
        print(f"Pronóstico generado exitosamente.")

    except Exception as e:
        print(f"Falló la generación de pronóstico para SAP {sap} con {best_model_name}: {e}")

# Combinar todos los pronósticos y guardar en Excel
if final_forecasts:
    final_forecasts_df = pd.concat(final_forecasts)
    output_path = 'output/final_predictions.xlsx'

    # Organizar en formato pivote para mejor legibilidad
    pivot_forecast = final_forecasts_df.pivot(index='SAP', columns='month_year', values='predicted_demand')
    pivot_forecast.to_excel(output_path)

    print(f"\nTodos los pronósticos finales han sido guardados en: {output_path}")
else:
    print("\nNo se generaron pronósticos finales.")
