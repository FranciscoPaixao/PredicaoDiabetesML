import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report
import joblib

dados = pd.read_csv('diabetes.csv')

# Substitui os valores faltantes

dados['Glucose'] = dados['Glucose'].replace(0, dados['Glucose'].mode()[0])

dados['BloodPressure'] = dados['BloodPressure'].replace(0, dados['BloodPressure'].median())

dados['SkinThickness'] = dados['SkinThickness'].replace(0, dados['SkinThickness'].median())

dados['Insulin'] = dados['Insulin'] .replace(0, dados['Insulin'].median())

dados['BMI'] = dados['BMI'].replace(0, dados['BMI'].median())


X = dados.drop(["Outcome"] ,axis="columns")
y = dados['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1337, shuffle=True)

modeloMLP = MLPClassifier(max_iter=20000, activation='logistic',verbose=False)

modeloMLP.fit(X_train, y_train)

y_predMLP = modeloMLP.predict(X_test)

relatorioMLP = classification_report(y_test, y_predMLP)

print("Relatório MLP: \n", relatorioMLP)

modeloSVM = svm.SVC(kernel='linear')
modeloSVM.fit(X_train, y_train)

y_predSVM = modeloSVM.predict(X_test)

relatorioSVM = classification_report(y_test, y_predSVM)

print("Relatório SVM: \n", relatorioSVM)

modeloLR = LogisticRegression(solver='liblinear')

modeloLR.fit(X_train, y_train)

y_predLR = modeloLR.predict(X_test)

relatorioLR = classification_report(y_test, y_predLR)

print("Relatório LR: \n", relatorioLR)

joblib.dump(modeloMLP, 'modelos/mlp.obj')
joblib.dump(modeloSVM, 'modelos/svm.obj')
joblib.dump(modeloLR, 'modelos/lr.obj')

