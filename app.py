from flask import Flask, request
from flask_restx import Resource, Api, apidoc

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

import joblib


apidoc.apidoc.url_prefix = "/api"

app = Flask(__name__)
api = Api(app)

@api.route('/api/predicaodiabetes/')
class PredicaoDiabetes(Resource):
    @api.doc(params={
      'modelo': 'svm, mlp ou lr',
      'gravidez' : '',
      'glicose' : '',
      'pressao_sanguinea' : '',
      'espessura_pele' : '',
      'insulina' : '',
      'imc' : '',
      'DiabetesPedigreeFunction' : '',
      'idade' : ''
    })
    def get(self):
        modelo = request.args.get('modelo')
        gravidez = int(request.args.get('gravidez'))
        glicose = int(request.args.get('glicose'))
        pressao_sanguinea = int(request.args.get('pressao_sanguinea'))
        espessura_pele = int(request.args.get('espessura_pele'))
        insulina = int(request.args.get('insulina'))
        IMC = float(request.args.get('imc'))
        DiabetesPedigreeFunction = float(request.args.get('DiabetesPedigreeFunction'))
        idade = int(request.args.get('idade'))

        dados_paciente = (gravidez, glicose, pressao_sanguinea, espessura_pele, insulina, IMC, DiabetesPedigreeFunction, idade)

        dados_paciente_reshape = np.array(dados_paciente).reshape(1, -1)

        if modelo == 'svm':
            modelo = joblib.load('modelos/svm.obj')
        elif modelo == 'mlp':
            modelo = joblib.load('modelos/mlp.obj')
        elif modelo == 'lr':
           modelo = joblib.load('modelos/lr.obj')
        
        resultado = modelo.predict(dados_paciente_reshape)

        if resultado[0] == 1:
            return {'resultado': 'positivo'}
        else:
            return {'resultado': 'negativo'}