import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# 1. CARGA DE DATOS
print("--- 1. CARGANDO DATOS ---")
try:
    df = pd.read_csv('Covid Data.csv', sep=',')
    print(f"Archivo cargado exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas.")
except Exception as e:
    print(f"Error cargando archivo: {e}")
    exit()

# 2. PREPROCESAMIENTO
print("\n--- 2. PREPROCESAMIENTO ---")

# A. Crear variable objetivo
if 'IS_DEAD' not in df.columns:
    print("Generando columna IS_DEAD basada en DATE_DIED...")
    # Lógica: 0 si es '9999-99-99', -1 si es una fecha válida
    df['IS_DEAD'] = df['DATE_DIED'].apply(lambda x: 0 if x == '9999-99-99' else -1)

# B. Eliminar columnas que no sirven
# DATE_DIED y MONTH_DIED se quitan para que el modelo no sepa la respuesta de antemano
cols_to_drop = ['id', 'DATE_DIED', 'MONTH_DIED', 'CLASIFFICATION_FINAL']
# Solo borramos si existen en el dataframe
df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

# C. Limpieza de valores 97, 98, 99 (Valores Faltantes)
print("Imputando valores faltantes (97, 98, 99) con la moda...")
for col in df_clean.columns:
    # No tocamos la Edad (es numérica real) ni el objetivo (IS_DEAD)
    if col != 'AGE' and col != 'IS_DEAD':
        # Calculamos la moda (valor más común) ignorando los códigos de error
        valid_values = df_clean[col][~df_clean[col].isin([97, 98, 99])]
        if len(valid_values) > 0:
            moda = valid_values.mode()[0]
            # Reemplazamos los errores con la moda
            df_clean[col] = df_clean[col].replace({97: moda, 98: moda, 99: moda})

# D. Normalización de la Edad (Escalar de 0 a 1)
print("Normalizando variable AGE...")
scaler = MinMaxScaler()
df_clean[['AGE']] = scaler.fit_transform(df_clean[['AGE']])

# 3. BALANCEO DE DATOS (Undersampling)
print("\n--- 3. BALANCEO DE DATOS (Undersampling) ---")

# Separar en dos grupos: Fallecidos (-1) y Vivos (0)
df_dead = df_clean[df_clean['IS_DEAD'] == -1]
df_alive = df_clean[df_clean['IS_DEAD'] == 0]

print(f"Conteo Original -> Fallecidos: {len(df_dead)}, Vivos: {len(df_alive)}")

# Tomamos una muestra aleatoria de los vivos igual al número de muertos
# random_state=42 asegura que siempre salga la misma muestra (reproducible)
df_alive_sample = df_alive.sample(n=len(df_dead), random_state=42)

# Unimos y mezclamos (shuffle) todo
df_balanced = pd.concat([df_dead, df_alive_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset Balanceado Final -> Total: {len(df_balanced)} registros (50% / 50%)")
# 4. GUARDAR ARCHIVO PROCESADO 
nombre_archivo_salida = 'Covid_Procesado_Final.csv'
print(f"\n--- GUARDANDO ARCHIVO: {nombre_archivo_salida} ---")
df_balanced.to_csv(nombre_archivo_salida, index=False)
print("¡Archivo guardado exitosamente en la misma carpeta!")
