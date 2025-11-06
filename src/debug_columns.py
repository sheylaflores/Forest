import pandas as pd

try:
    df = pd.read_excel('output/error_metrics_enhanced.xlsx', index_col='SAP')
    print("Archivo cargado. Las columnas son:")
    print(df.columns.tolist())
except Exception as e:
    print(f"No se pudo leer el archivo: {e}")
