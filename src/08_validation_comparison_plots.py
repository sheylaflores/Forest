import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuración Inicial ---
sns.set_theme(style="whitegrid")
output_dir = 'output/validation_plots_advanced'
os.makedirs(output_dir, exist_ok=True)

# --- Carga de Datos ---
try:
    df_history = pd.read_csv('output/data_featured_advanced.csv', parse_dates=['month_year'])
    df_validation_preds = pd.read_csv('output/model_predictions_advanced.csv', parse_dates=['month_year'])
    print("Datos históricos y de predicción de validación avanzada cargados.")
except FileNotFoundError as e:
    print(f"Error: No se encontró un archivo requerido. {e}")
    exit()

# --- Identificar los 5 Productos Principales ---
top_5_saps = df_history.groupby('SAP')['Consumo Total'].sum().nlargest(5).index.tolist()
print(f"Se generarán gráficos para los 5 productos de mayor consumo: {top_5_saps}")

# --- Generación de Gráficos ---
for sap_code in top_5_saps:
    validation_data_sap = df_validation_preds[df_validation_preds['SAP'] == sap_code]
    actuals_sap = validation_data_sap[validation_data_sap['model'] == 'Actual']
    predictions_sap = validation_data_sap[validation_data_sap['model'] != 'Actual']

    if actuals_sap.empty:
        print(f"No se encontraron datos de validación para {sap_code}, saltando.")
        continue

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=actuals_sap, x='month_year', y='prediction', label='Consumo Real', color='black', linewidth=2.5, marker='o')
    sns.lineplot(data=predictions_sap, x='month_year', y='prediction', hue='model', style='model', markers=True, dashes=False)

    plt.title(f'Comparativo Avanzado de Predicciones vs. Real para: {sap_code}', fontsize=16)
    plt.xlabel('Meses de Validación (Feb-2025 a Oct-2025)', fontsize=12)
    plt.ylabel('Consumo Total', fontsize=12)
    plt.legend(title='Leyenda')
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'validation_comparison_advanced_{sap_code}.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Gráfico para {sap_code} guardado en: {output_path}")

print("\n¡Generación de gráficos de comparación de validación avanzada completada!")
