#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar o pré-processamento dos dados.
Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.
@author: Aydano Machado <aydano.machado@gmail.com>
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
import requests


print('\n - Lendo o arquivo com o dataset sobre diabetes')
df = pd.read_csv('diabetes_dataset.csv')

# Criando X and y par ao algorítmo de aprendizagem de máquina.\
print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')
# Caso queira modificar as colunas consideradas basta algera o array a seguir.
feature_cols = ['Pregnancies', 'Glucose', 'BMI',  'DiabetesPedigreeFunction', 'Age']

#Removing outliers
max_pedigree = df.DiabetesPedigreeFunction.mean()


dataNew = df.fillna(df.mean())

col = ['BMI', 'DiabetesPedigreeFunction']
df = df[(df[col] <= df[col].quantile(0.75)) & (df[col] >= df[col].quantile(0.25))]


dataNew = df.fillna(df.mean())

X = dataNew[feature_cols]
y = df.Outcome

# Ciando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

#realizando previsões com o arquivo de
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_csv('diabetes_app.csv')
y_pred = neigh.predict(data_app[feature_cols])
print(y_pred)
# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

#TODO Substituir pela sua chave aqui
DEV_KEY = "IWCO"

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,'predictions':pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text


print(" - Resposta do servidor:\n", r.text, "\n")