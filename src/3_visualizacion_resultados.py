# -*- coding: utf-8 -*-
"""
Script 3: Visualización de Resultados de Predicción

Funciones:
1. Carga los datos históricos (filtrados) y las predicciones finales.
2. Identifica los 5 productos más consumidos dentro de la lista de productos clave.
3. Para cada uno de estos productos top, genera y guarda un gráfico que compara
   el consumo histórico con la demanda predicha a 12 meses.
4. Incluye las bandas de confianza en los gráficos para los modelos que las soportan (ARIMA/SARIMA).
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- Configuración ---
sns.set(style="whitegrid")
# (El resto de las rutas se definen en la función principal)

# --- Funciones Principales ---

def crear_variable_exogena(index):
    """Crea un DataFrame con la variable exógena 'temporada_pesca'."""
    df_exog = pd.DataFrame(index=index)
    df_exog['temporada_pesca'] = df_exog.index.month.isin([4, 5, 6, 11, 12]).astype(int)
    return df_exog

def generar_visualizaciones_con_bandas(df_historico_full, df_predicciones, top_n=5):
    """
    Identifica los productos top y genera gráficos de predicción, incluyendo
    bandas de confianza para modelos ARIMA/SARIMA.
    """
    OUTPUT_DIR = 'output'
    if df_historico_full is None or df_predicciones.empty:
        print("No hay datos suficientes para generar visualizaciones.")
        return

    # 1. Identificar los productos TOP N por consumo histórico
    consumo_total_por_sap = df_historico_full.groupby('sap')['consumo'].sum()
    productos_predichos = df_predicciones['sap'].unique()
    top_productos_sap = consumo_total_por_sap[consumo_total_por_sap.index.isin(productos_predichos)].nlargest(top_n).index

    print(f"\n--- Generando gráficos para los {top_n} productos más importantes ---")

    # 2. Iterar y generar un gráfico para cada producto top
    for sap in top_productos_sap:
        # Preparar datos históricos del producto
        serie_historica = df_historico_full[df_historico_full['sap'] == sap]['consumo'].resample('ME').sum()
        start_date, end_date = df_historico_full.index.min(), df_historico_full.index.max()
        full_range = pd.date_range(start=start_date, end=end_date, freq='ME')
        serie_historica = serie_historica.reindex(full_range, fill_value=0)

        # Obtener predicción y metadatos
        prediccion_producto = df_predicciones[df_predicciones['sap'] == sap]
        pred_serie = prediccion_producto.set_index('fecha_prediccion')['demanda_predicha']

        info = prediccion_producto.iloc[0]
        descripcion, mejor_modelo, rmse_modelo = info['descripcion'], info['mejor_modelo'], info['modelo_rmse']

        # Crear el gráfico
        plt.figure(figsize=(16, 8))
        plt.plot(serie_historica.index, serie_historica.values, label='Consumo Histórico', color='teal', marker='o', linestyle='-')
        plt.plot(pred_serie.index, pred_serie.values, label='Demanda Predicha', color='crimson', marker='x', linestyle='--')

        # Añadir bandas de confianza si el modelo es ARIMA o SARIMA
        if mejor_modelo in ['ARIMA', 'SARIMA']:
            try:
                exog_historico = crear_variable_exogena(serie_historica.index)
                exog_futuro = crear_variable_exogena(pred_serie.index)

                if mejor_modelo == 'ARIMA':
                    model = ARIMA(serie_historica, exog=exog_historico, order=(5, 1, 0)).fit()
                else: # SARIMA
                    model = SARIMAX(serie_historica, exog=exog_historico, order=(5, 1, 0), seasonal_order=(1, 1, 1, 12)).fit(disp=False)

                forecast = model.get_forecast(steps=len(pred_serie), exog=exog_futuro)
                conf_int = forecast.conf_int()

                plt.fill_between(pred_serie.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='crimson', alpha=0.2, label='Intervalo de Confianza (95%)')
            except Exception as e:
                print(f"No se pudieron generar bandas de confianza para {sap}: {e}")

        plt.title(f'Predicción de Demanda para: {descripcion} (SAP: {sap})\n(Mejor Modelo: {mejor_modelo} | RMSE: {rmse_modelo:.2f})')
        plt.xlabel('Fecha')
        plt.ylabel('Consumo')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        plot_path = os.path.join(OUTPUT_DIR, f'prediccion_producto_{sap}.png')
        plt.savefig(plot_path)
        print(f">>> Gráfico para el producto {sap} guardado.")
        plt.close()

# --- Bloque Principal ---

def main():
    # Definir rutas
    CONSUMO_DATA_PATH = 'data/kardexASTEC_filtrado.xlsx'
    SAP_INFO_PATH = 'data/14.06 MP_ARTICULOS_SAP.xlsx'
    PREDICCIONES_PATH = 'output/predicciones_por_producto.csv'

    # Cargar datos históricos filtrados
    df_sap = pd.read_excel(SAP_INFO_PATH, usecols=['Codigo_SAP', 'Marca'])
    df_sap.rename(columns={'Codigo_SAP': 'sap'}, inplace=True)
    marcas_a_excluir = ['MATERIALES VARIOS', 'GENÉRICO']
    saps_a_excluir = df_sap[df_sap['Marca'].isin(marcas_a_excluir)]['sap'].unique().tolist()

    df_consumo = pd.read_excel(CONSUMO_DATA_PATH, usecols=['SAP', 'month_year', 'Consumo Total'])
    df_consumo.rename(columns={'month_year': 'fecha', 'Consumo Total': 'consumo', 'SAP': 'sap'}, inplace=True)

    df_consumo = df_consumo[~df_consumo['sap'].isin(saps_a_excluir)]
    df_consumo['fecha'] = pd.to_datetime(df_consumo['fecha'])
    df_consumo = df_consumo[df_consumo['fecha'] >= '2022-01-01']
    df_consumo.set_index('fecha', inplace=True)

    # Cargar predicciones
    try:
        df_predicciones = pd.read_csv(PREDICCIONES_PATH, parse_dates=['fecha_prediccion'])
    except FileNotFoundError:
        print(f"Error: Archivo de predicciones no encontrado en {PREDICCIONES_PATH}")
        return

    # Generar gráficos
    generar_visualizaciones_con_bandas(df_consumo, df_predicciones)

if __name__ == "__main__":
    main()
