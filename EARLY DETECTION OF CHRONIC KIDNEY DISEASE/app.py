import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('lgr.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/Prediction',methods=['POST','GET'])
def prediction():
    return render_template('index.html')
@app.route('/Home',methods=['POST','GET'])
def my_home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features=[float(x) for x in request.form.values()]
    # Add missing features with default values
    input_features += [0.0] * 6  # add 6 missing features with default value of 0.0
    input_features += [1]  # add missing feature 'diabetes_mellitus' with default value of 1
    features_value=[np.array(input_features)]
    features_name = ['age', 'blood_pressure', 'albumin', 'sugar', 'red_blood_cells',                 'pus_cell', 'pus_cell_clumps', 'bacteria', 'blood_glucose_random',                  'blood_urea', 'serum_creatinine', 'sodium', 'potassium', 'hemoglobin',                 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',                 'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite','pedal_edema', 'anemia']
    df=pd.DataFrame(features_value, columns=features_name)
    output=model.predict(df)
    return render_template('result.html',prediction_text=output[0])

if __name__=='__main__':
    app.run(debug=False)
