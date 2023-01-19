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
import joblib
import json



app = Flask(__name__)

rsf = joblib.load("RF_compressed.joblib")


#cols = ['Age', 'SBP', 'DBP', 'BMI',
#       'CurrentSmoking', 'HTN', 'Diabetes', 'HTNmeds', 'SCreat', 'HDL', 'LDL',
#       'TotChol', 'Trigs', 'FPG', 'NTproBNP', 'Troponin', 'CRP', 'QRS', 'QTc',
#       'LVHcv']


@app.route('/predict',methods=['POST'])
def predict():
    jdata = request.json
    jdata = json.dumps(jdata)
    data_unseen = pd.read_json(jdata)
    prediction = rsf.predict(data_unseen)
    output = prediction.round(decimals= 2)
    return jsonify({"prediction":list(output)})

if __name__ == "__main__":
    app.run(debug=True)


    
    
