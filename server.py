import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import figure
import numpy as np
#%matplotlib inline

from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
from flask import Flask, request, url_for, redirect, render_template, jsonify
import pickle
import joblib

#import portpicker
#port = portpicker.pick_unused_port()

#from gevent.pywsgi import WSGIServer
#host='localhost'



app = Flask(__name__)

#rsf = pickle.load(open('C:/Aditya/matt/watch_dm_rsf_model.pkl','rb'))

rsf = joblib.load("C:/Aditya/cvscore_app/RF_compressed.joblib")


cols = ['Age', 'SBP', 'DBP', 'BMI',
       'CurrentSmoking', 'HTN', 'Diabetes', 'HTNmeds', 'SCreat', 'HDL', 'LDL',
       'TotChol', 'Trigs', 'FPG', 'NTproBNP', 'Troponin', 'CRP', 'QRS', 'QTc',
       'LVHcv']

##x = [75.052055,	15.0,	152,	63,	0.9,	131.0,	8.070549,	33.0,	0,	1,	1]

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame(np.reshape(final, (1,20)), columns = cols)
    prediction = rsf.predict(data_unseen)
    prediction = float(prediction)
    return render_template('results.html',pred='Expected CV Score will be {}'.format(prediction))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = rsf.predict(data_unseen)
    output = prediction
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)


    
    
