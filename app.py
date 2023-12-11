from flask import Flask, request
from flask_restx import Resource, Api, apidoc

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

apidoc.apidoc.url_prefix = "/api"

app = Flask(__name__)
api = Api(app)

@api.route('/api/predicaodiabetes/')
class PredicaoDiabetes(Resource):
    @api.doc(params={
      'modelo': 'SMV',
      'gravidez' : '',
      'glicose' : '',
      'pressao_sanguinea' : '',
      'espessura_pele' : '',
      'insulina' : '',
      'imc' : '',
      'DiabetesPedigreeFunction' : ''
    })
    def get(self):
        modelo = request.args.get('modelo')
        gravidez = request.args.get('gravidez')
        glicose = request.args.get('glicose')
        pressao_sanguinea = request.args.get('pressao_sanguinea')
        espessura_pele = request.args.get('espessura_pele')
        insulina = request.args.get('insulina')
        IMC = request.args.get('imc')
        DiabetesPedigreeFunction = request.args.get('DiabetesPedigreeFunction')

        dados_paciente = [gravidez, glicose, pressao_sanguinea, espessura_pele, insulina, IMC, DiabetesPedigreeFunction]

        dados_paciente_reshape = np.array(dados_paciente).reshape(1, -1)

        dados_paciente_final = scaler.transform(dados_paciente_reshape)

        if modelo == 'modelo1':
            return {'resultado': 'Diabetes'}
        elif modelo == 'modelo2':
            return {'resultado': 'Não Diabetes'}
        else:
            return {'resultado': 'Não encontrado'}