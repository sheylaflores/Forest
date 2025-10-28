# -*- coding: utf-8 -*-
"""
Script 1: Limpieza de Datos y Análisis Exploratorio (EDA)

Funciones:
1. Carga los datos de productos y consumo.
2. Filtra y excluye productos de marcas "MATERIALES VARIOS" y "GENÉRICO".
3. Guarda la lista de productos excluidos.
4. Filtra los datos de consumo para el período 2022-2025.
5. Muestra estadísticas descriptivas del consumo mensual total.
6. Genera y guarda gráficos de consumo mensual para cada año.
7. Aplica un análisis de Pareto (80/20) para identificar y guardar los productos clave.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuración ---
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 7)
CONSUMO_DATA_PATH = 'data/kardexASTEC_filtrado.xlsx'
SAP_INFO_PATH = 'data/14.06 MP_ARTICULOS_SAP.xlsx'
OUTPUT_DIR = 'output'

# --- Funciones Principales ---

def limpiar_y_analizar():
    """
    Función principal que orquesta todo el proceso de limpieza y EDA.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Cargar y filtrar por marca
    try:
        df_sap = pd.read_excel(SAP_INFO_PATH, usecols=['Codigo_SAP', 'Marca'])
        df_sap.rename(columns={'Codigo_SAP': 'sap'}, inplace=True)
    except Exception as e:
        print(f"Error crítico al leer el archivo de información de SAP: {e}")
        return

    marcas_a_excluir = ['MATERIALES VARIOS', 'GENÉRICO']
    saps_a_excluir = df_sap[df_sap['Marca'].isin(marcas_a_excluir)]['sap'].unique().tolist()

    # Guardar la lista de productos excluidos
    with open(os.path.join(OUTPUT_DIR, 'productos_excluidos.txt'), 'w') as f:
        for sap in saps_a_excluir:
            f.write(f"{sap}\n")
    print(f">>> Se identificaron {len(saps_a_excluir)} productos genéricos. La lista se guardó en 'output/productos_excluidos.txt'.")

    # 2. Cargar datos de consumo y aplicar filtros
    try:
        df_consumo = pd.read_excel(CONSUMO_DATA_PATH, usecols=['SAP', 'month_year', 'Consumo Total'])
        df_consumo.rename(columns={'month_year': 'fecha', 'Consumo Total': 'consumo', 'SAP': 'sap'}, inplace=True)
    except Exception as e:
        print(f"Error crítico al leer el archivo de consumo: {e}")
        return

    df_consumo = df_consumo[~df_consumo['sap'].isin(saps_a_excluir)]
    df_consumo['fecha'] = pd.to_datetime(df_consumo['fecha'])
    df_consumo = df_consumo[df_consumo['fecha'] >= '2022-01-01']
    df_consumo.set_index('fecha', inplace=True)
    df_consumo.sort_index(inplace=True)

    print(">>> Datos de consumo filtrados por marca y período (2022-2025).")

    # 3. Calcular y mostrar estadísticas descriptivas
    consumo_mensual_total = df_consumo['consumo'].resample('ME').sum()
    print("\n--- Resumen Estadístico del Consumo Mensual Total (2022-2025) ---")
    print(consumo_mensual_total.describe().to_string())

    # 4. Generar gráficos anuales
    print("\n--- Generando gráficos de consumo mensual por año ---")
    años = df_consumo.index.year.unique()
    for año in años:
        df_año = df_consumo[df_consumo.index.year == año]
        consumo_mensual_año = df_año['consumo'].resample('ME').sum()
        idx = pd.date_range(f'01-01-{año}', f'12-31-{año}', freq='ME')
        consumo_mensual_año = consumo_mensual_año.reindex(idx, fill_value=0)

        plt.figure(figsize=(12, 6))
        consumo_mensual_año.plot(kind='bar', color=sns.color_palette('viridis', 12))
        plt.title(f'Consumo Mensual del Año {año} (Marcas Relevantes)')
        plt.xlabel('Mes')
        plt.ylabel('Consumo Total')
        plt.xticks(ticks=range(12), labels=[d.strftime('%b') for d in idx], rotation=45)
        plt.tight_layout()

        plot_path = os.path.join(OUTPUT_DIR, f'consumo_mensual_{año}.png')
        plt.savefig(plot_path)
        print(f">>> Gráfico para el año {año} guardado.")
        plt.close()

    # 5. Identificar y guardar productos clave
    print("\n--- Identificando productos clave (Análisis ABC) ---")
    consumo_por_producto = df_consumo.groupby('sap')['consumo'].sum().sort_values(ascending=False)
    consumo_acumulado = consumo_por_producto.cumsum()
    consumo_total = consumo_por_producto.sum()
    porcentaje_acumulado = (consumo_acumulado / consumo_total) * 100

    productos_clave = porcentaje_acumulado[porcentaje_acumulado <= 80].index.tolist()

    with open(os.path.join(OUTPUT_DIR, 'productos_clave.txt'), 'w') as f:
        for sap in productos_clave:
            f.write(f"{sap}\n")

    print(f">>> Se identificaron {len(productos_clave)} productos clave (que representan el 80% del consumo).")
    print(">>> La lista se guardó en 'output/productos_clave.txt'.")
    print("\n>>> Proceso de EDA completado.")

# --- Bloque de Ejecución ---
if __name__ == "__main__":
    limpiar_y_analizar()
