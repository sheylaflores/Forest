import pandas as pd

# Definir la lista de códigos SAP críticos
CRITICAL_SAPS = [
    'A18110001067', 'A18110009021', 'A18110009037', 'A18130007380', 'A18130006673',
    'A18110008863', 'A18110009031', 'A18130006711', 'A18130006764', 'A18130007768',
    'A18130007812', 'A22020000062', 'A18110000362', 'A18110002547', 'A18130007396',
    'A18130007399', 'A18130007814', 'A18110008772', 'A18110010465', 'A18130005688',
    'A18130007393'
]

# Cargar los datos
try:
    df = pd.read_excel('data/data.xlsx')
    print("Datos cargados exitosamente.")
except FileNotFoundError:
    print("Error: No se encontró el archivo data/data.xlsx.")
    exit()

# 1. Preprocesamiento de Fechas
df['month_year'] = pd.to_datetime(df['month_year'])
start_date = pd.to_datetime('2022-01-01')
end_date = pd.to_datetime('2025-10-31')
df = df[(df['month_year'] >= start_date) & (df['month_year'] <= end_date)]
print(f"Datos filtrados por fecha (Ene 2022 - Oct 2025). {len(df)} registros restantes.")

# 2. Filtrado por Códigos SAP
df = df[df['SAP'].isin(CRITICAL_SAPS)]
print(f"Datos filtrados por los 20 SAPs críticos. {len(df)} registros restantes.")

# 3. Revisión de Nulos
if df[['SAP', 'month_year', 'Consumo Total']].isnull().any().any():
    print("Se encontraron valores nulos. Se procederá a rellenarlos con 0.")
    df['Consumo Total'] = df['Consumo Total'].fillna(0)
else:
    print("No se encontraron valores nulos en las columnas clave.")

# 4. Agregar datos para asegurar una única serie de tiempo por SAP
# Agrupar por SAP y mes, sumando el consumo total
df_processed = df.groupby(['SAP', 'month_year'])['Consumo Total'].sum().reset_index()
print(f"Datos agregados. {len(df_processed)} registros finales en el dataset procesado.")

# 5. Guardar los datos procesados
output_path = 'output/data_processed.csv'
df_processed.to_csv(output_path, index=False)
print(f"Datos procesados y guardados exitosamente en: {output_path}")

# Ver una muestra de los datos procesados
print("\nMuestra de los datos procesados:")
print(df_processed.head())
