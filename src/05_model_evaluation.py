import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# --- Función de Métrica Segura ---
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = np.where(y_true == 0, 1e-6, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true))

# --- Carga de Datos de Predicciones ---
try:
    df_advanced = pd.read_csv('output/model_predictions_advanced.csv', parse_dates=['month_year'])
    print("Predicciones del modelo avanzado cargadas exitosamente.")
except FileNotFoundError as e:
    print(f"Error: No se encontró 'output/model_predictions_advanced.csv'. {e}")
    exit()

# --- Cálculo de Nuevas Métricas de Error ---
actuals_df = df_advanced[df_advanced['model'] == 'Actual'][['SAP', 'month_year', 'prediction']].rename(columns={'prediction': 'actual'})
predictions_df = df_advanced[df_advanced['model'] != 'Actual']
merged_df = pd.merge(predictions_df, actuals_df, on=['SAP', 'month_year'])

error_metrics_advanced = (
    merged_df.groupby(['SAP', 'model'])
    .apply(lambda g: pd.Series({
        'RMSE': np.sqrt(mean_squared_error(g['actual'], g['prediction'])),
        'MAE': mean_absolute_error(g['actual'], g['prediction']),
        'MAPE': calculate_mape(g['actual'], g['prediction'])
    }))
    .reset_index()
    .rename(columns={'model': 'Model'})
)

# --- Creación de la Nueva Tabla de Errores ---
pivot_df_advanced = error_metrics_advanced.pivot_table(index='SAP', columns='Model', values=['RMSE', 'MAE', 'MAPE'])
if not pivot_df_advanced.empty:
    pivot_df_advanced = pivot_df_advanced.swaplevel(0, 1, axis=1).sort_index(axis=1)
    pivot_df_advanced.columns = ['_'.join(col).strip() for col in pivot_df_advanced.columns.values]

# --- **CORRECCIÓN: Guardar la Lista de Mejores Modelos Correcta** ---
# Identificar el mejor modelo para cada SAP basado en el MAPE de los nuevos resultados
best_models_advanced = error_metrics_advanced.loc[error_metrics_advanced.groupby('SAP')['MAPE'].idxmin()]

# Guardar esta lista para que la use el script de pronóstico
best_models_output_path = 'output/best_models.csv'
best_models_advanced[['SAP', 'Model']].to_csv(best_models_output_path, index=False)
print(f"\nLista de mejores modelos actualizada y guardada correctamente en: {best_models_output_path}")
print(best_models_advanced[['SAP', 'Model', 'MAPE']])

# --- Guardado de la Tabla de Errores Detallada ---
summary_df_advanced = pivot_df_advanced.reset_index()
summary_df_advanced = summary_df_advanced.merge(
    best_models_advanced[['SAP', 'Model']].rename(columns={'Model': 'Selected_Model_Advanced'}),
    on='SAP', how='left'
).set_index('SAP')

output_advanced_errors_path = 'output/error_metrics_advanced.xlsx'
summary_df_advanced.to_excel(output_advanced_errors_path)
print(f"Nueva tabla de errores detallada guardada en: {output_advanced_errors_path}")
