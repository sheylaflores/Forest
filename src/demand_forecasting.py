
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

# Define a function for Mean Absolute Percentage Error (MAPE)
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

# Function to create the output directory if it doesn't exist
def ensure_output_dir(directory="output/"):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Data Loading and Initial Preprocessing
def load_and_prepare_data(filepath="data.xlsx"):
    df = pd.read_excel(filepath)
    df['month_year'] = pd.to_datetime(df['month_year'])
    df = df.rename(columns={'Consumo Total': 'y', 'month_year': 'ds'})
    # Aggregate data to handle duplicates, summing up consumption for each product per month.
    df = df.groupby(['SAP', 'ds'])['y'].sum().reset_index()
    return df

# Feature Engineering
def create_features(df):
    df['temporada_pesca'] = df['ds'].dt.month.isin([4, 5, 6, 11, 12]).astype(int)
    df['pico_pre_temporada'] = df['ds'].dt.month.isin([2, 3, 9, 10]).astype(int)
    # Lag and rolling features can be added here if needed, but SARIMA and Prophet handle time dependencies internally.
    # For XGBoost, we'll create them inside the XGBoost-specific function.
    return df

def run_forecasting_pipeline():
    ensure_output_dir()

    # List of 20 critical products
    critical_products = [
        'A18110001067', 'A18110009021', 'A18110009037', 'A18130007380', 'A18130006673',
        'A18110008863', 'A18110009031', 'A18130006711', 'A18130006764', 'A18130007768',
        'A18130007812', 'A22020000062', 'A18110000362', 'A18110002547', 'A18130007396',
        'A18130007399', 'A18130007814', 'A18110008772', 'A18110010465', 'A18130005688', 'A18130007393'
    ]

    full_df = load_and_prepare_data()

    all_metrics = []
    all_forecasts = []

    for i, product_sap in enumerate(critical_products):
        print(f"Processing Product {i+1}/{len(critical_products)}: {product_sap}")

        product_df = full_df[full_df['SAP'] == product_sap].copy()

        # Create a complete date range for the product
        date_range = pd.date_range(start='2022-01-01', end='2025-10-01', freq='MS')
        product_df = product_df.set_index('ds').reindex(date_range, fill_value=0).reset_index()
        product_df = product_df.rename(columns={'index': 'ds'})
        product_df['SAP'] = product_sap

        product_df = create_features(product_df)

        # Split data: 37 months for training, 9 for validation
        train_df = product_df.iloc[:37]
        validation_df = product_df.iloc[37:]

        # --- Model Training and Evaluation ---

        # SARIMA Model
        try:
            exog_vars = ['temporada_pesca', 'pico_pre_temporada']
            sarima_model = SARIMAX(train_df['y'], exog=train_df[exog_vars], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            sarima_fit = sarima_model.fit(disp=False)
            sarima_pred = sarima_fit.get_forecast(steps=len(validation_df), exog=validation_df[exog_vars]).predicted_mean
            sarima_pred.index = validation_df.index
            sarima_pred[sarima_pred < 0] = 0
        except Exception as e:
            print(f"SARIMA failed for {product_sap}: {e}")
            sarima_pred = pd.Series([0] * len(validation_df), index=validation_df.index)


        # Prophet Model
        prophet_model = Prophet()
        prophet_model.add_regressor('temporada_pesca')
        prophet_model.add_regressor('pico_pre_temporada')
        prophet_model.fit(train_df[['ds', 'y', 'temporada_pesca', 'pico_pre_temporada']])
        future = prophet_model.make_future_dataframe(periods=len(validation_df), freq='MS')
        future = pd.merge(future, product_df[['ds', 'temporada_pesca', 'pico_pre_temporada']], on='ds', how='left')
        prophet_forecast = prophet_model.predict(future)
        prophet_pred = prophet_forecast['yhat'].iloc[-len(validation_df):]
        prophet_pred.index = validation_df.index
        prophet_pred[prophet_pred < 0] = 0

        # XGBoost Model
        xgb_train_df = train_df.copy()
        for lag in [1, 3, 6, 12]:
            xgb_train_df[f'lag_{lag}'] = xgb_train_df['y'].shift(lag)
        xgb_train_df['rolling_mean_3'] = xgb_train_df['y'].shift(1).rolling(window=3).mean()
        xgb_train_df = xgb_train_df.dropna()

        features = exog_vars + [col for col in xgb_train_df.columns if 'lag_' in col or 'rolling_' in col]
        X_train, y_train = xgb_train_df[features], xgb_train_df['y']

        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        xgb_model.fit(X_train, y_train)

        # Create features for validation set
        X_val = validation_df[exog_vars].copy()
        history = list(train_df['y'])
        xgb_preds = []
        for t in range(len(validation_df)):
            # Create lag/rolling features for the current step
            X_val_t = X_val.iloc[[t]]
            for lag in [1, 3, 6, 12]:
                X_val_t[f'lag_{lag}'] = history[-lag]
            X_val_t['rolling_mean_3'] = np.mean(history[-3:])

            pred = xgb_model.predict(X_val_t[features])[0]
            pred = max(0, pred)
            xgb_preds.append(pred)
            history.append(pred) # Use prediction for next step's features
        xgb_pred = pd.Series(xgb_preds, index=validation_df.index)

        # --- Evaluate Models ---
        models = {'SARIMA': sarima_pred, 'Prophet': prophet_pred, 'XGBoost': xgb_pred}
        for name, pred in models.items():
            rmse = np.sqrt(mean_squared_error(validation_df['y'], pred))
            mae = mean_absolute_error(validation_df['y'], pred)
            mape = calculate_mape(validation_df['y'], pred)
            all_metrics.append({'SAP': product_sap, 'Model': name, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape})

        # --- Select Best Model ---
        metrics_df = pd.DataFrame(all_metrics)
        product_metrics = metrics_df[metrics_df['SAP'] == product_sap]
        best_model_name = product_metrics.loc[product_metrics['MAPE'].idxmin()]['Model']
        print(f"Best model for {product_sap}: {best_model_name}")

        # --- Generate Final Forecast with Best Model ---
        full_train_df = product_df.copy()
        forecast_dates = pd.date_range(start='2025-11-01', periods=6, freq='MS')
        forecast_df = pd.DataFrame({'ds': forecast_dates})
        forecast_df = create_features(forecast_df)

        final_forecast = []
        if best_model_name == 'SARIMA':
            try:
                final_model = SARIMAX(full_train_df['y'], exog=full_train_df[exog_vars], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                final_fit = final_model.fit(disp=False)
                final_forecast = final_fit.get_forecast(steps=6, exog=forecast_df[exog_vars]).predicted_mean
            except Exception as e:
                print(f"Final SARIMA failed for {product_sap}: {e}")
                final_forecast = pd.Series([0] * 6)

        elif best_model_name == 'Prophet':
            final_model = Prophet()
            final_model.add_regressor('temporada_pesca')
            final_model.add_regressor('pico_pre_temporada')
            final_model.fit(full_train_df[['ds', 'y', 'temporada_pesca', 'pico_pre_temporada']])
            prophet_future = pd.concat([full_train_df[['ds', 'temporada_pesca', 'pico_pre_temporada']], forecast_df])
            forecast_results = final_model.predict(prophet_future)
            final_forecast = forecast_results['yhat'].iloc[-6:]

        elif best_model_name == 'XGBoost':
            xgb_full_train_df = full_train_df.copy()
            for lag in [1, 3, 6, 12]:
                xgb_full_train_df[f'lag_{lag}'] = xgb_full_train_df['y'].shift(lag)
            xgb_full_train_df['rolling_mean_3'] = xgb_full_train_df['y'].shift(1).rolling(window=3).mean()
            xgb_full_train_df = xgb_full_train_df.dropna()

            X_full_train, y_full_train = xgb_full_train_df[features], xgb_full_train_df['y']
            final_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
            final_xgb.fit(X_full_train, y_full_train)

            history = list(full_train_df['y'])
            forecast_features = forecast_df[exog_vars].copy()
            xgb_final_preds = []
            for t in range(len(forecast_features)):
                forecast_step_features = forecast_features.iloc[[t]]
                for lag in [1, 3, 6, 12]:
                    forecast_step_features[f'lag_{lag}'] = history[-lag]
                forecast_step_features['rolling_mean_3'] = np.mean(history[-3:])

                pred = final_xgb.predict(forecast_step_features[features])[0]
                xgb_final_preds.append(pred)
                history.append(pred)
            final_forecast = pd.Series(xgb_final_preds)

        final_forecast[final_forecast < 0] = 0
        final_forecast_df = pd.DataFrame({'SAP': product_sap, 'ds': forecast_dates, 'Forecast': final_forecast.values})
        all_forecasts.append(final_forecast_df)

        # --- Generate Plots (for the first 5 products) ---
        if i < 5:
            # Validation Plot
            plt.figure(figsize=(12, 6))
            plt.plot(train_df['ds'], train_df['y'], label='Train Data')
            plt.plot(validation_df['ds'], validation_df['y'], label='Actual Validation Data', color='black')
            for name, pred in models.items():
                plt.plot(validation_df['ds'], pred, label=f'{name} Prediction')
            plt.title(f'Validation Performance for {product_sap}')
            plt.legend()
            plt.savefig(f"output/validation_{product_sap}.png")
            plt.close()

            # Forecast Plot
            plt.figure(figsize=(12, 6))
            plt.plot(product_df['ds'], product_df['y'], label='Historical Data')
            plt.plot(final_forecast_df['ds'], final_forecast_df['Forecast'], label='Forecast', color='red')
            plt.title(f'6-Month Demand Forecast for {product_sap}')
            plt.legend()
            plt.savefig(f"output/forecast_{product_sap}.png")
            plt.close()

            # Absolute Error Plot
            abs_error = np.abs(validation_df['y'] - models[best_model_name])
            plt.figure(figsize=(10, 5))
            plt.bar(validation_df['ds'], abs_error, width=20, label=f'Absolute Error ({best_model_name})')
            plt.title(f'Absolute Error per Month for {product_sap}')
            plt.legend()
            plt.savefig(f"output/abs_error_{product_sap}.png")
            plt.close()

    # --- Save Final Deliverables ---
    metrics_summary = pd.DataFrame(all_metrics)
    metrics_summary.to_excel("output/error_summary.xlsx", index=False)

    final_forecasts_df = pd.concat(all_forecasts)
    final_forecasts_df.to_excel("output/demand_forecast.xlsx", index=False)

    print("Forecasting pipeline completed successfully!")

if __name__ == '__main__':
    run_forecasting_pipeline()
