import pandas as pd

# Cargar el archivo de datos
try:
    df = pd.read_excel('data/data.xlsx')
    print("Archivo cargado exitosamente. A continuación, las primeras 5 filas:")
    print(df.head())
    print("\nInformación general del DataFrame:")
    df.info()
except FileNotFoundError:
    print("Error: No se encontró el archivo data/data.xlsx.")
except Exception as e:
    print(f"Ocurrió un error al cargar el archivo: {e}")
