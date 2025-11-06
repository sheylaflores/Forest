import pandas as pd

# Cargar los datos procesados
try:
    df = pd.read_csv('output/data_processed.csv')
    df['month_year'] = pd.to_datetime(df['month_year'])
    print("Datos procesados cargados exitosamente.")
except FileNotFoundError:
    print("Error: No se encontró el archivo output/data_processed.csv.")
    exit()

# Asegurar que los datos estén ordenados por SAP y fecha
df = df.sort_values(by=['SAP', 'month_year'])

# --- Ingeniería de Características Estándar ---
df['temporada_pesca'] = df['month_year'].dt.month.isin([4, 5, 6, 11, 12]).astype(int)
df['pico_pre_temporada'] = df['month_year'].dt.month.isin([2, 3, 9, 10]).astype(int)
df['mes'] = df['month_year'].dt.month
df['anio'] = df['month_year'].dt.year

# --- Ingeniería de Características Avanzada ---
# Agrupar por SAP para aplicar las transformaciones de series de tiempo
df_grouped = df.groupby('SAP')['Consumo Total']

# Lag Features (Memoria de Consumos Pasados)
df['consumo_lag_1'] = df_grouped.shift(1)
df['consumo_lag_3'] = df_grouped.shift(3)
df['consumo_lag_12'] = df_grouped.shift(12)

# Rolling Features (Tendencia y Volatilidad Recientes)
df['consumo_rolling_mean_3'] = df_grouped.shift(1).rolling(window=3, min_periods=1).mean()
df['consumo_rolling_std_3'] = df_grouped.shift(1).rolling(window=3, min_periods=1).std()

# Rellenar los valores NaN resultantes de las operaciones de shift y rolling
# Es importante rellenarlos para que los modelos no fallen. Usamos 0 como valor neutral.
df.fillna(0, inplace=True)

print("Ingeniería de características avanzada completada. Se han añadido 5 nuevas columnas.")

# Guardar el dataset final
output_path = 'output/data_featured_advanced.csv'
df.to_csv(output_path, index=False)
print(f"Dataset con características avanzadas guardado en: {output_path}")

# Ver una muestra de los datos con las nuevas características
print("\nMuestra de los datos con las nuevas características:")
print(df.tail())
