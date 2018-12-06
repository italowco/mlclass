import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import requests
print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_csv('diabetes_dataset.csv')

# Criando X and y par ao algorítmo de aprendizagem de máquina.\
# Caso queira modificar as colunas consideradas basta algera o array a seguir.
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
#X = data[feature_cols].dropna()


X = data[feature_cols]

print(data[feature_cols].dtypes)

y = data.Outcome

# Retirando os valores com colunas vazias
dfCapado = data.dropna()

# Separando grupos de diabeticos e nao diabeticos
dfNaoDiabetes = dfCapado[dfCapado['Outcome'] == 0]
dfDiabete = dfCapado[dfCapado['Outcome'] == 1]

dfNaoDiabetesCompleto = data[data['Outcome'] == 0]
dfDiabeteCompleto = data[data['Outcome'] == 1]

# data com o nan preenchido com a media total
#preencher apenas usando a media
dataFill = data[data['Outcome'] == 1].fillna(dfDiabeteCompleto.mean())
dataFill2 = data[data['Outcome'] == 0].fillna(dfNaoDiabetesCompleto.mean())

dataNew = pd.concat([dataFill, dataFill2])
X = dataNew.sort_index()
