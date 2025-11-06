import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Configuración Inicial ---
sns.set_theme(style="whitegrid")
output_dir = 'output/demand_diagnostics'
os.makedirs(output_dir, exist_ok=True)

# --- Carga de Datos ---
try:
    df_history = pd.read_csv('output/data_featured.csv', parse_dates=['month_year'])
    df_errors = pd.read_excel('output/error_metrics_enhanced.xlsx', index_col='SAP')
    print("Datos históricos y de errores cargados.")
except FileNotFoundError as e:
    print(f"Error: No se encontró un archivo requerido. {e}")
    exit()

# --- Identificar los 5 Productos con Peor Rendimiento (Corrección Definitiva) ---
# Seleccionar solo las columnas que contienen 'MAPE' en su nombre
mape_cols = [col for col in df_errors.columns if 'MAPE' in col and not 'Best' in col]
if not mape_cols:
    print("Error: No se encontraron columnas de MAPE en el archivo de errores.")
    exit()

# Calcular el MAPE promedio entre todos los modelos para cada producto
df_errors['mean_MAPE'] = df_errors[mape_cols].mean(axis=1)
worst_performers_saps = df_errors.nlargest(5, 'mean_MAPE').index.tolist()
print(f"Análisis de diagnóstico para los 5 productos con peor rendimiento (basado en MAPE promedio): {worst_performers_saps}")

# --- Análisis y Visualización ---
diagnostic_results = []

for sap_code in worst_performers_saps:
    sap_data = df_history[df_history['SAP'] == sap_code].set_index('month_year')['Consumo Total']

    if sap_data.empty:
        continue

    cv = np.std(sap_data) / np.mean(sap_data) if np.mean(sap_data) > 0 else 0
    zero_proportion = (sap_data == 0).mean()

    diagnostic_results.append({'SAP': sap_code, 'Coeficiente de Variación': cv, 'Proporción de Ceros': zero_proportion})

    plt.figure(figsize=(14, 7))
    sap_data.plot(marker='o', linestyle='-')
    plt.title(f'Serie de Tiempo para Producto: {sap_code}\nCV: {cv:.2f} | Proporción de Ceros: {zero_proportion:.2%}', fontsize=16)
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Consumo Total', fontsize=12)

    output_path = os.path.join(output_dir, f'timeseries_{sap_code}.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Gráfico de diagnóstico para {sap_code} guardado.")

# --- Mostrar Tabla de Diagnóstico ---
diagnostics_df = pd.DataFrame(diagnostic_results).set_index('SAP')
print("\n--- Resultados del Diagnóstico de Demanda ---")
print(diagnostics_df)
print("\nConclusión del diagnóstico: Un Coeficiente de Variación > 1.0 indica una demanda altamente volátil.")
print("Una alta Proporción de Ceros indica una demanda intermitente (difícil de predecir).")
