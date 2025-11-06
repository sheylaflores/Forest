import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuración Inicial ---
sns.set_theme(style="whitegrid")
os.makedirs('output/individual_forecasts_advanced', exist_ok=True)

# --- Carga de Datos ---
try:
    df_history = pd.read_csv('output/data_featured_advanced.csv', parse_dates=['month_year'])
    df_final_preds_pivot = pd.read_excel('output/final_predictions_advanced.xlsx', index_col='SAP')
    # Convertir a formato largo
    df_final_preds = df_final_preds_pivot.melt(ignore_index=False, var_name='month_year', value_name='predicted_demand').reset_index()
    df_final_preds['month_year'] = pd.to_datetime(df_final_preds['month_year'])
    print("Datos históricos y pronósticos finales avanzados cargados.")
except FileNotFoundError as e:
    print(f"Error: No se encontró un archivo requerido. {e}")
    exit()

# --- Generación de Gráficos de Pronóstico Individual ---
print("Generando 20 gráficos de pronóstico individual actualizados...")
for sap_code in df_history['SAP'].unique():
    history_sap = df_history[df_history['SAP'] == sap_code]
    forecast_sap = df_final_preds[df_final_preds['SAP'] == sap_code]

    if forecast_sap.empty:
        print(f"  - No se encontró pronóstico para {sap_code}, saltando.")
        continue

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=history_sap, x='month_year', y='Consumo Total', label='Consumo Histórico', color='royalblue')
    sns.lineplot(data=forecast_sap, x='month_year', y='predicted_demand', label='Demanda Pronosticada (Avanzado)', color='darkorange', linestyle='--')

    plt.axvline(history_sap['month_year'].max(), color='red', linestyle='--', label='Inicio del Pronóstico')

    plt.title(f'Pronóstico de Demanda Avanzado para: {sap_code}', fontsize=14)
    plt.xlabel('Fecha')
    plt.ylabel('Demanda')
    plt.legend()
    plt.tight_layout()
    # Guardar en una nueva carpeta para evitar sobreescribir
    plt.savefig(f'output/individual_forecasts_advanced/forecast_{sap_code}.png')
    plt.close()
print("Todos los gráficos individuales actualizados generados.")

print("\n¡Visualización de pronósticos completada!")
