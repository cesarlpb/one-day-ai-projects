# 01. Clasificador de Sentimientos en Textos

#%% Instalamos NLTK (https://www.nltk.org/howto.html):
"""pip install nltk textblob"""

#%% Dataset: https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download
# El CSV descomprimido pesa +200Mb así que vamos a dividirlo en 3 partes
# Descargo el CSV y lo renombro como data.csv

import pandas as pd

df = pd.read_csv("data.csv", header=None)
df.head() # observamos que viene sin nombre de columnas

#%% Añado los nombre de columnas usando la descripción de Kaggle:
column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Asignar los nombres de las columnas al DataFrame
df.columns = column_names

df.head()

#%% Visualizamos solo target y texto
df_acortado = df[["target", "text"]]
df_acortado.tail(25) # las filas están ordenadas por target

# %%

# Seleccionar 3 filas para cada valor en la columna "target"
sampled_df = df_acortado.groupby('target').sample(n=3, random_state=1)

print("\nDataFrame muestreado:")
print(sampled_df)

#%% Otro muestreo

# Definir una función para manejar el muestreo
def sample_n(group, n=3):
    if len(group) < n:
        return group
    return group.sample(n=n, random_state=1)

# Aplicar la función a cada grupo
sampled_df = df.groupby('target', group_keys=False).apply(sample_n, n=3)

print("\nDataFrame muestreado:")
print(sampled_df)

#%% valores únicos de target
# Ver todos los valores únicos en la columna "target"
unique_targets = df['target'].unique()

print("\nValores únicos en la columna 'target':")
print(unique_targets) # 0 y 4


#%% Bajamos uno de los datasets de NLTK:
import nltk
# Descargar datos necesarios para NLTK
nltk.download('punkt')
# Cargar tu conjunto de datos de texto aquí
