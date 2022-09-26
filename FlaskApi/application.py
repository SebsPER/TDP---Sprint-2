
#import flask
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, request
from pickle import load

model = tf.keras.models.load_model('grades.h5')
scaler = load(open('scaler.pkl', 'rb'))

application = Flask(__name__)

def cat_2_num(data):
    newdata = data.copy()
    newdata['SEXO'].replace(['Masculino', 'Femenino'], [0,1], inplace=True)
    #data['NACIONALIDAD'].replace(['PERÚ', 'Extranjero', 'PERU'], [1,0,1], inplace=True)
    newdata['POBLACIÓN'].replace(['Urbano', 'Rural'], [0,1], inplace=True)
    newdata['TIEMPO DE ESTUDIO'].replace(['1 - 2 horas', '2 - 4 horas', '4 - 6 horas', '6 - 8 horas', '8 - más horas', 'Ninguno'], [1,2,3,4,5,0], inplace=True)
    newdata['ACTIVIDADES EXTRACADÉMICAS'].replace(['Grupos Artísticos','Equipos Deportivos','Equipos Académicos','Equipos Extracadémicos','Voluntariadados','Ninguno'], [1,2,3,4,5,0], inplace=True)
    newdata['TAMAÑO DE LA FAMILIA'].replace(['1 - 4','5 - 8','9 - 12','13 - más'], [1,2,3,4], inplace=True)
    newdata['TIPO DE VIVIENDA'].replace(['Unifamiliar','Edificio Multifamiliar','Conjunto Residencial','Quinta','Otro'], [1,2,3,4,0], inplace=True)
    newdata['ENCARGADO DEL MENOR'].replace(['Padre','Madre','Apoderado','Familiar Cercano','Personal Autorizado','Otro'], [1,2,3,4,5,0], inplace=True)
    newdata['ESTADO CIVIL - PADRE'].replace(['Soltero','Casado','Divorciado','Viudo','Conviviente'], [1,2,3,4,5], inplace=True)
    newdata['ESTADO CIVIL - MADRE'].replace(['Soltero','Casado','Divorciado','Viudo','Conviviente'], [1,2,3,4,5], inplace=True)
    newdata['EDUCACIÓN - PADRE'].replace(['Edcacion Inicial','Educacion Primaria','Educacion Secundaria',
                                    'Educación Profesional Técnica','Grado Superior','Educación Universitaria','Ninguno'], [1,2,3,4,5,6,0], inplace=True)
    newdata['EDUCACIÓN - MADRE'].replace(['Edcacion Inicial','Educacion Primaria','Educacion Secundaria',
                                    'Educación Profesional Técnica','Grado Superior','Educación Universitaria','Ninguno'], [1,2,3,4,5,6,0], inplace=True)
    newdata['TRABAJO - PADRE'].replace(['Administración y Comercio','Actividades Agrarias','Actividades Marítimo-Pesqueras','Artes Gráficas',
                                    'Artesanías y Manualidades','Computación e Informática','Comunicación, Imagen y Sonido','Construcción','Cuero y Calzado',
                                    'Electricidad y Electrónica','Estética Personal','Hostelería y Turismo','Industrias Alimentarias','Mecánica y Metales',
                                    'Mecánica y Motores','Minería','Química','Textil y Confección','Independiente','Docencia','Medicina'], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21], inplace=True)
    newdata['TRABAJO - MADRE'].replace(['Administración y Comercio','Actividades Agrarias','Actividades Marítimo-Pesqueras','Artes Gráficas',
                                    'Artesanías y Manualidades','Computación e Informática','Comunicación, Imagen y Sonido','Construcción','Cuero y Calzado',
                                    'Electricidad y Electrónica','Estética Personal','Hostelería y Turismo','Industrias Alimentarias','Mecánica y Metales',
                                    'Mecánica y Motores','Minería','Química','Textil y Confección','Independiente','Docencia','Medicina'], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21], inplace=True)
    #data['NACIONALIDAD - PADRE'].replace(['PERÚ', 'Extranjero'], [1,0], inplace=True)
    #data['NACIONALIDAD - MADRE'].replace(['PERÚ', 'Extranjero'], [1,0], inplace=True)
    newdata['VIVE CON EL ESTUDIANTE - PADRE'].replace(['SI', 'NO'], [1,2], inplace=True)
    newdata['VIVE CON EL ESTUDIANTE - MADRE'].replace(['SI', 'NO'], [1,2], inplace=True)
    newdata['ENFERMEDADES'].replace(['Dislexia','Disgrafía','Discalculia','Discapacidad de la Memoria y el Procesamiento Auditivo','Trastorno por Déficit de Atención e Hiperactividad (TDHA)'
                                ,'Trastorno del Espectro Autista/Trastorno Generalizado del Desarrollo','Discapacidad Intelectual','Otros','Ninguna'], [1,2,3,4,5,6,7,8,0], inplace=True)
    newdata['TIENE NECESIDADES ESPECIALES'].replace(['SI', 'NO'], [1,0], inplace=True)
    newdata['GRADO'].replace(['PRIMARIA', 'SECUNDARIA'], [1,2], inplace=True)
    newdata['AÑO LECTIVO'].replace(['PRIMER GRADO', 'SEGUNDO GRADO', 'TERCER GRADO', 'CUARTO GRADO', 'QUINTO GRADO', 'SEXTO GRADO',
                                'PRIMER AÑO', 'SEGUNDO AÑO', 'TERCER AÑO','CUARTO AÑO','QUINTO AÑO'], [1,2,3,4,5,6,7,8,9,10,11], inplace=True)
    newdata['ÁREA'].replace(['CIENCIA Y TECNOLOGÍA', 'CASTELLANO COMO SEGUNDA LENGUA', 'INGLÉS',
                                'MATEMÁTICA', 'COMUNICACIÓN', 'ARTE Y CULTURA', 'EDUCACIÓN FÍSICA',
                                'EDUCACIÓN RELIGIOSA', 'PERSONAL SOCIAL', 'CIENCIAS SOCIALES',
                                'DESARROLLO PERSONAL, CIUDADANÍA Y CÍVICA',
                                'EDUCACIÓN PARA EL TRABAJO'], [1,2,3,4,5,6,7,8,9,10,11,12], inplace=True)

    codes = np.arange(1, 42, 1)
    new = []
    for i in range(1, 42):
        if i < 10:
            new.append('COM000'+str(i))
            #print('COM000'+str(i))
        else: new.append('COM00'+str(i))
    newdata['COMPETENCIA'].replace(new, codes, inplace=True)
    
    return newdata

def std(data):
    newdata = scaler.transform(data)

    return newdata

@application.route('/')
def hello():
    return 'Hello everyone, this is a API'


@application.route('/predict', methods=['GET'])
def infer_image():

    apidict = {'EDAD': int(request.args['edad']), 
           'SEXO': request.args['sexo'], 
           'POBLACIÓN' : request.args['pob'], 
           'TIEMPO DE ESTUDIO': request.args['tim'],
           'ACTIVIDADES EXTRACADÉMICAS': request.args['act'], 
           'TAMAÑO DE LA FAMILIA': request.args['fam'],
           'TIPO DE VIVIENDA': request.args['viv'], 
           'ENCARGADO DEL MENOR': request.args['enc'], 
           'ESTADO CIVIL - PADRE': request.args['cp'],
           'EDUCACIÓN - PADRE': request.args['ep'], 
           'TRABAJO - PADRE': request.args['tp'],
           'VIVE CON EL ESTUDIANTE - PADRE': request.args['vp'], 
           'ESTADO CIVIL - MADRE': request.args['cm'],
           'EDUCACIÓN - MADRE': request.args['em'], 
           'TRABAJO - MADRE': request.args['tm'],
           'VIVE CON EL ESTUDIANTE - MADRE': request.args['vm'], 
           'GRADO': request.args['grad'],
           'AÑO LECTIVO': request.args['anho'], 
           'ÁREA': request.args['area'],
           'COMPETENCIA': request.args['comp'], 
           'ENFERMEDADES': request.args['enf'], 
           'TIENE NECESIDADES ESPECIALES': [request.args['esp']]}
    
    data = pd.DataFrame(apidict)
    data = cat_2_num(data)
    data = std(data)

    pred = model.predict(data)

    top_prob = np.argmax(pred)
    #preds[i] = index_max
    reps = ['C', 'B', 'A', 'AD']

    return jsonify(res= reps[top_prob-1])

@application.route('/test', methods=['GET'])
def index():
    return jsonify(message='Hello there :)')


if __name__ == '__main__':
    application.run(debug=True, host='0.0.0.0') #,  , port=8080