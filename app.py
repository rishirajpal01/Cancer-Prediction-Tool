#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

class lung_cancer:
    def fun(self, re):
        df=pd.read_csv("lung_cancer.csv")

        #removing unnessary columns
        df = df[['Age', 'Smokes',  'Smokes_years', 'Smokes_packs_per_year', 'AreaQ', 'Alkhol', 'family history', 'Result']]

    

        #splitting dataset into values VS result
        x = df[['Age', 'Smokes',  'Smokes_years', 'Smokes_packs_per_year', 'AreaQ', 'Alkhol', 'family history']]
        y = df[['Result']]

        #Logistic regression model
        model = LogisticRegression()

        model.fit(x, y)

        prediction = model.predict(re)

        return(prediction)

    

class breast_cancer:
    def fun(self, re):
        #UPDATE FROM HERE
        df=pd.read_csv("data.csv")

        #B - benign (noncancerous) - 0
        #M - malignant (cancerous) - 1
        for i in range(0, len(df['diagnosis'])):
            if(df['diagnosis'][i] == 'M'):
                df['diagnosis'][i] = 1
            else:
                df['diagnosis'][i] = 0

        #Speficing the type
        df['diagnosis'] = df['diagnosis'].astype(str).astype(float)

        #removing unnessary columns
        df = df[['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]   


        #removing co-related columns whos corelation > 9.0
        df = df.drop(['perimeter_mean', 'area_mean', 'radius_worst', 
              'area_worst', 'texture_worst', 'area_mean', 'perimeter_worst', 
              'concavity_mean', 'concave points_worst', 'perimeter_se', 'area_se'], axis = 1)

        #removing co-related columns whos corelation > 7.0
        df = df.drop(['concave points_mean', 'smoothness_worst', 'compactness_se', 'compactness_worst', 'concavity_worst', 
              'fractal_dimension_worst', 'concavity_se', 'concave points_se', 'fractal_dimension_se', 
              'concave points_se', 'fractal_dimension_se', 'concavity_worst', 'fractal_dimension_worst'], axis = 1)

        #removing co-related columns whos corelation > 6.5
        df = df.drop(['radius_se', 'compactness_mean', 'symmetry_worst'], axis = 1)

        #splitting dataset into values VS result
        x = df[['radius_mean', 'texture_mean', 'smoothness_mean',
       'symmetry_mean', 'fractal_dimension_mean', 'texture_se',
       'smoothness_se', 'symmetry_se']]

        y = df[['diagnosis']]

        #SVM model
        svm = SVC(random_state=42, kernel='poly')
        svm.fit(x,y)

        prediction = svm.predict(re)

        return(prediction)

class prostate_cancer:
    def fun(self,  re):
        Cancer = pd.read_csv('Prostate_Cancer.csv')

        # We don't care id of the columns. So, we drop that!
        Cancer.drop(['id'],axis=1,inplace=True)

        #B - benign (noncancerous) - 0
        #M - malignant (cancerous) - 1
        Cancer.diagnosis_result = [1 if each == 'M' else 0 for each in Cancer.diagnosis_result]

        #splitting dataset into values VS result
        y = Cancer.diagnosis_result.values
        x_data = Cancer.drop(['diagnosis_result'],axis=1)

        #Decision Tree model
        # Creating the classifier object 
        clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=5, min_samples_leaf=7) 
  
        # Performing training 
        clf_gini.fit(x_data, y) 
    
        y_predition = clf_gini.predict(re)

        return(y_predition)





from flask import Flask
from flask import render_template
from flask import request
app=Flask(__name__)

app.static_folder='static'

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/prostateCancer')
def prostateCancer():
    return render_template('prostateCancer.html')

@app.route('/breastCancer')
def breastCancer():
    return render_template('breastCancer.html')

@app.route('/lungCancer')
def lungCancer():
    return render_template('lungCancer.html')

@app.route('/result_lung_cancer', methods=['POST'])
def result_lung_cancer():

    RESULT = ['NO_CANCER','CANCER']
    
    re = pd.DataFrame(columns=['Age', 'Smokes', 'Smokes_years', 'Smokes_packs_year', 'AreaQ', 'Alkhol', 'family_history'])
    
    re = re.append({'Age': float(request.form['Age']), 
                    'Smokes': float(request.form['Smokes']),
                    'Smokes_years': float(request.form['Smokes_years']),
                    'Smokes_packs_year': float(request.form['Smokes_packs_year']),
                    'AreaQ': float(request.form['AreaQ']),
                    'Alkhol': float(request.form['Alkhol']),
                    'family_history': float(request.form['family_history'])}, ignore_index=True                                        
                    )
    
    
    n = lung_cancer()
    p = n.fun(re)
    testresult=RESULT[int(p)]

    if(testresult=="CANCER"):
        return render_template('resultCancer.html')
    if(testresult=="NO_CANCER"):
        return render_template('resultNotCancer.html')
    

    

#BreastCancer
@app.route('/result_breast_cancer', methods=['POST'])
def result_breast_cancer():

    RESULT = ['NO_CANCER','CANCER']
    
    re = pd.DataFrame(columns=['radius_mean', 'texture_mean', 'smoothness_mean',
       'symmetry_mean', 'fractal_dimension_mean', 'texture_se',
       'smoothness_se', 'symmetry_se'])
    
    re = re.append({'radius_mean': float(request.form['radius_mean']), 
                    'texture_mean': float(request.form['texture_mean']),
                    'smoothness_mean': float(request.form['smoothness_mean']),
                    'symmetry_mean': float(request.form['symmetry_mean']),
                    'fractal_dimension_mean': float(request.form['fractal_dimension_mean']),
                    'texture_se': float(request.form['texture_se']),
                    'smoothness_se': float(request.form['smoothness_se']),
                    'symmetry_se': float(request.form['symmetry_se'])}, ignore_index=True                                        
                    )
    
    
    n = breast_cancer()
    p = n.fun(re)
    testresult=RESULT[int(p)]

    if(testresult=="CANCER"):
        return render_template('resultCancer.html')
    if(testresult=="NO_CANCER"):
        return render_template('resultNotCancer.html')
    




#Prostate Cancer
@app.route('/result_prostate_cancer', methods=['POST'])
def result_prostate_cancer():

    RESULT = ['NO_CANCER','CANCER']
    
    re = pd.DataFrame(columns=['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness',
       'symmetry','fractal_dimension'])
    
    re = re.append({'radius': float(request.form['radius']), 
                    'texture': float(request.form['texture']),
                    'perimeter': float(request.form['perimeter']),
                    'area': float(request.form['area']),
                    'smoothness': float(request.form['smoothness']),
                    'compactness': float(request.form['compactness']),
                    'symmetry': float(request.form['symmetry']),
                    'fractal_dimension': float(request.form['fractal_dimension'])}, ignore_index=True                                        
                    )
    
    
    n = prostate_cancer()
    p = n.fun(re)
    testresult=RESULT[int(p)]

    if(testresult=="CANCER"):
        return render_template('resultCancer.html')
    if(testresult=="NO_CANCER"):
        return render_template('resultNotCancer.html')
    
    