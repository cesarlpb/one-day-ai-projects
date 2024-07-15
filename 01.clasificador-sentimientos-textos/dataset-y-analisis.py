# 01. Clasificador de Sentimientos en Textos

#%% Instalamos NLTK (https://www.nltk.org/howto.html):
"""pip install nltk textblob"""

#%% Dividir dataset en tres partes:
import pandas as pd
import numpy as np

def split_and_save_dataset(df, file_prefix):
    # Dividir el DataFrame en tres partes iguales
    df1, df2, df3 = np.array_split(df, 3)
    
    # Guardar cada parte en un archivo CSV
    df1.to_csv(f'{file_prefix}_part1.csv', index=False)
    df2.to_csv(f'{file_prefix}_part2.csv', index=False)
    df3.to_csv(f'{file_prefix}_part3.csv', index=False)

    print(f"Dataset dividido en 3 partes y guardado como {file_prefix}_part1.csv, {file_prefix}_part2.csv, y {file_prefix}_part3.csv")

df = pd.read_csv("data.csv", header=None)
split_and_save_dataset(df, "data")

#%% cargamos el dataset desde las 3 partes
def load_and_combine_datasets(file_prefix):
    # Cargar los tres archivos CSV
    df1 = pd.read_csv(f'{file_prefix}_part1.csv')
    df2 = pd.read_csv(f'{file_prefix}_part2.csv')
    df3 = pd.read_csv(f'{file_prefix}_part3.csv')
    
    # Concatenar los DataFrames en uno solo
    combined_df = pd.concat([df1, df2, df3], ignore_index=True)
    
    return combined_df

combined_df = load_and_combine_datasets("data")

#%% Comparación de datasets:
# Convertir todas las columnas a string antes de la comparación
column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
df.columns = column_names
combined_df.columns = column_names

print("Número de filas en el DataFrame original :", len(df))
print("Número de filas en el DataFrame combinado:", len(combined_df))

print("Columnas en el DataFrame original :", df.columns.tolist())
print("Columnas en el DataFrame combinado:", combined_df.columns.tolist())

import pandas as pd

try:
    pd.testing.assert_frame_equal(df, combined_df)
    print("El DataFrame original y el DataFrame combinado son iguales.")
except AssertionError as e:
    print("El DataFrame original y el DataFrame combinado no son iguales.")
    print(e)

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

############################################################

#%% 
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Asegúrate de tener instalado nltk y scikit-learn:
# !pip install nltk scikit-learn

# Descargar stopwords y punkt tokenizer
nltk.download('stopwords')
nltk.download('punkt')

# Cargar el CSV
file_path = 'data.csv'  # Reemplaza con la ruta a tu archivo CSV
df = pd.read_csv(file_path, header=None, names=['target', 'id', 'date', 'query', 'user', 'text'])

# Mantener solo las columnas relevantes
df = df[['target', 'text']]

# Mostrar las primeras filas del DataFrame
print(df.head())

# %%
import nltk

# Descargar los recursos necesarios de NLTK
nltk.download('stopwords')
nltk.download('punkt')
#%%
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Definir función de preprocesamiento
def preprocess_text(text):
    # Convertir a minúsculas
    text = text.lower()
    # Tokenizar el texto
    tokens = word_tokenize(text)
    # Eliminar stopwords y signos de puntuación
    tokens = [word for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]
    # Unir tokens en una cadena
    return ' '.join(tokens)

# Aplicar preprocesamiento al texto
# Seleccionar 1000 filas aleatorias
df_sample = df.sample(n=1_000_000, random_state=42)

print("DataFrame de muestra aleatoria (primeras 5 filas):")
# print(df_sample.head())

df_sample['text'] = df_sample['text'].apply(preprocess_text)

# Mostrar las primeras filas del DataFrame preprocesado
print(df_sample.head())
#%% vectorizar
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df_sample['text'], df_sample['target'], test_size=0.2, random_state=42)

# Vectorizar los textos
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#%% Entrenar
# Entrenar el modelo
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test_vec)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

#%% Matriz de confusión
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# %%
# Visualizar la matriz de confusión
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Guardar la gráfica
plt.savefig('confusion_matrix.png')

# Mostrar la gráfica
plt.show()

# %%
