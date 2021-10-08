from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('stroke_classification_xgboost.pkl', 'rb'))

@app.route('/', methods = ['GET'])
def Home():
    return render_template('index.html')

standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        hypertension = request.form['hypertension']
        if(hypertension == 'Yes'):
            hypertension = 1
        else:
            hypertension = 0
        heart_disease = request.form['heart_disease']
        if(heart_disease == 'Yes'):
            heart_disease = 1
        else:
            heart_disease = 0
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        gender_Male = request.form['gender_Male']
        if(gender_Male == 'Male'):
            gender_Male = 1
            gender_Other = 0
        else:
            gender_Male = 0
            gender_Other = 1
        ever_married_Yes = request.form['ever_married_Yes']
        if(ever_married_Yes == 'Yes'):
            ever_married_Yes = 1
        else: 
            ever_married_Yes = 0
        work_type_Never_worked = request.form['work_type_Never_worked']
        if(work_type_Never_worked == 'Never_worked'):
            work_type_Never_worked = 1
            work_type_Private = 0
            work_type_Self_employed = 0
            work_type_children = 0
        elif(work_type_Never_worked == 'Private'):
            work_type_Never_worked = 0
            work_type_Private = 1
            work_type_Self_employed = 0
            work_type_children = 0
        elif(work_type_Never_worked == 'Self_employed'):
            work_type_Never_worked = 0
            work_type_Private = 0
            work_type_Self_employed = 1
            work_type_children = 0
        elif(work_type_Never_worked == 'Children'):
            work_type_Never_worked = 0
            work_type_Private = 0
            work_type_Self_employed = 0
            work_type_children = 1
        else:
            work_type_Never_worked = 0
            work_type_Private = 0
            work_type_Self_employed = 0
            work_type_children = 0
        Residence_type_Urban = request.form['Residence_type_Urban']
        if(Residence_type_Urban == 'Urban'):
            Residence_type_Urban = 1
        else:
            Residence_type_Urban = 0
        smoking_status_formerly_smoked = request.form['smoking_status_formerly_smoked']
        if(smoking_status_formerly_smoked == 'Formerly'):
            smoking_status_formerly_smoked = 1
            smoking_status_never_smoked = 0
            smoking_status_smokes = 0
        elif(smoking_status_formerly_smoked == 'Never'):
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 1
            smoking_status_smokes = 0
        elif(smoking_status_formerly_smoked == 'Currently'):
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 0
            smoking_status_smokes = 1
        else:
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 0
            smoking_status_smokes = 0
        
        input_features = [age, hypertension, heart_disease, avg_glucose_level, bmi, gender_Male, gender_Other, ever_married_Yes,
                                     work_type_Never_worked, work_type_Private,  work_type_Self_employed, work_type_children, Residence_type_Urban,
                                     smoking_status_formerly_smoked, smoking_status_never_smoked, smoking_status_smokes]
        final_features = np.array(input_features)
        final_features = final_features.reshape(1, -1)
        prediction = model.predict(final_features)
        if(prediction == 1):
            return render_template('index.html', prediction_text = "High possibility of having a stroke")
        else:
            return render_template('index.html', prediction_text = "Low possibility of having a stroke")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

