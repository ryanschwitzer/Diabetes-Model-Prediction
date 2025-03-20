import pandas as pd
import numpy as np

import joblib

import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import cross_val_score



########################################################################################################



def prep_data():
    #Load DF
    dataset = pd.read_csv("C:/Users/rjsch/Desktop/app/diabetes_prediction_dataset.csv")
    
    #Normalize Data with Standardizing
    scaler = StandardScaler()
    dataset['age'] = np.log1p(dataset['age'])
    nor_df = pd.DataFrame(scaler.fit_transform(dataset[['bmi', 'HbA1c_level', 'blood_glucose_level']]))

    #Get dummies for gender and smoking history
    df_dum = pd.get_dummies(dataset[['gender', 'smoking_history']], drop_first=True)

    #grab variabled data
    df_valued = dataset[['hypertension', 'heart_disease']]

    #Add data THAT IS normalized to df
    df = pd.concat([dataset['age'], nor_df, df_dum, df_valued, dataset['diabetes']], axis=1)

    #Rename pandas columns for readability
    df = df.rename(columns={0: 'bmi', 1: 'HbA1c_level', 2: 'blood_glucose_level', 'gender_Other': 'gender_Female'})

    df_nor = df

    df = pd.read_csv("C:/Users/rjsch/Desktop/app/diabetes_prediction_dataset.csv")

    df_nor = df_nor.astype(float)

    pickle.dump(scaler, open('scaler.pkl', 'wb'))

    model_development(df, df_nor, dataset, scaler)
    return df, df_nor, scaler, dataset



def model_development(df, df_nor, dataset, scaler):
    #Log transform exponential features
    df['age'] = np.log1p(df['age'])
    #Polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(df_nor[['bmi', 'HbA1c_level', 'blood_glucose_level', ]])

    #transform
    X = np.hstack((df[['age', 'hypertension', 'heart_disease']], X_poly))
    y = df["diabetes"]

    #split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=15)

    #train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    #Flask addition
    pickle.dump(model, open('model.pkl', 'wb'))
    pickle.dump(poly, open('poly.pkl', 'wb'))


    model_accuracy(X_train, X_test, y_train, y_test, X, y, model, poly, dataset, scaler)
    return X_train, X_test, y_train, y_test, model, X, y, poly, model



def model_accuracy(X_train, X_test, y_train, y_test, X, y, model, poly, dataset, scaler):
    #Show accuracy of model
    accuracy = model.score(X_test, y_test)
    print(f'Accuracy of Model: {accuracy}')

    #prepare classification report
    y_prob = model.predict_proba(X_test)[:, 1]
    threshold = 0.4
    y_pred = (y_prob >= threshold).astype(int)

    #Classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

    #Precision Recall F1 Score
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}")


    gather_data(model, poly, dataset, scaler)



def gather_data(model, poly, dataset, scaler):
    #Gather info
    age = float(input("Enter age: "))
    bmi = float(input("Enter BMI: "))
    HbA1c_level = float(input("Enter HbA1c level: "))
    blood_glucose_level = float(input("Enter blood glucose level: "))
    hypertension = int(input('Do you have hypertension? (0/1): '))
    heart_disease = int(input('Do you posses any heart diseases? (0/1): '))


    #Put gathered info in dataframe
    input_data = pd.DataFrame({
        'age': age,
        'bmi': bmi,
        'HbA1c_level': HbA1c_level,
        'blood_glucose_level': blood_glucose_level,
        'hypertension': hypertension,
        'heart_disease': heart_disease
    }, index=[0])

    model_prediction(input_data, model, poly, scaler)
    return input_data



def model_prediction(input_data, model, poly, scaler):
    print(f"Original input_data:\n{input_data}")
    
    input_data['age'] = np.log1p(input_data['age'])
    
    #Check log transformation result
    print(f"Log-transformed input_data:\n{input_data}")
    
    #Normalize new data
    input_data[['bmi', 'HbA1c_level', 'blood_glucose_level']] = pd.DataFrame(scaler.transform(input_data[['bmi', 'HbA1c_level', 'blood_glucose_level']]))
    
    #Check the normalized data
    print(f"Normalized input_data:\n{input_data}")
    
    #Polynomial features for new data
    X_poly_pred = poly.transform(input_data[['bmi', 'HbA1c_level', 'blood_glucose_level']])
    
    #Combine the log-transformed features with the polynomial features
    X_pred = np.hstack((input_data[['age', 'hypertension', 'heart_disease']].values, X_poly_pred))

    # Get model prediction probabilities
    y_pred = model.predict_proba(X_pred)[:, 1]
    
    #Apply threshold for final prediction
    threshold = 0.35
    y_pred_done = (y_pred[0] >= threshold).astype(int)

    print(f"y_pred_done: {y_pred_done}") 
    
    #Final Prediction
    if y_pred_done == 0:
        prediction = 'No Diabetes'
        print(prediction)
    elif y_pred_done == 1:
        prediction = 'Diabetes'
        print(prediction)
        


prep_data()