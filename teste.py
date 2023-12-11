
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

import joblib



modelo = joblib.load('modelos/mlp.obj')

## Sem diabetes

dados_paciente = (2,141,84,26,175,34,0.42,36)

## Com diabetes
#dados_paciente = (2,197,70,45,543,30.5,158,53)

dados_paciente_reshape = np.array(dados_paciente).reshape(1, -1)

resultado = modelo.predict(dados_paciente_reshape)

print(resultado[0])
