# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 17:24:33 2023
Creates Random Forest Cross Validation Tree
@author: jjavi
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def getTreeObject(X_train, y_train):
    # Returns model of tree given split data
    
    # number of trees used
    number_of_trees=np.arange(50,501,50)
    # Getting number of features range
    number_of_features=np.arange(1,3)
    # Setting up cv grid
    param_grid_rf = {  'n_estimators': number_of_trees, 
     'max_features': number_of_features                        
     } 
    gridSearch_rf = GridSearchCV(RandomForestRegressor(), param_grid_rf, cv=5,scoring='neg_mean_squared_error')
    gridSearch_rf.fit(X_train, y_train)
    # Features used
    features = gridSearch_rf.best_params_['max_features'] 
    trees_used = gridSearch_rf.best_params_['n_estimators'] 
    # constructing tree using best parameters
    rf_model = RandomForestRegressor(n_estimators= trees_used , max_features=features, random_state=1)
    rf_model = rf_model.fit(X_train, y_train) 
    return rf_model, features, trees_used
    
def get_prediction(tm, job, experience, size): 
     # Returns the prediction 
     # Given tree object and predictors chosen
     # Equation job predictor logic 
     job_list = ['Data_Analyst', 'Data_Scientist', 'Data_Engineer', 'ML_Engineer']
     da = 0
     ds = 0
     de = 0
     ml = 0
     # assign 1 to job of interest for prediction
     for i in job_list:
         if i == job and i == job_list[0]:
             da = 1
         elif i == job and i == job_list[1]:
             ds = 1
         elif i == job and i == job_list[2]:
             de = 1  
         elif i == job and i == job_list[3]:
             ml = 1    
             
     data_to_predict = pd.DataFrame({'experience_level': [experience], 'company_size': [size], job_list[0]: [da], job_list[1]:[ds], job_list[2]:[de], job_list[3]:[ml]})
     prediction = tm.predict(data_to_predict)
     return int(prediction * 1000) 
    
    
def getMetricDataframe(tm, X_test, y_test,features, trees):
    # computing the Rmse
    # Returns dataframe with metrics formatted along with explanation
    #tree model and dataframe split as parameters
    predicted_values = tm.predict(X_test) 
    rmse = np.sqrt(mean_squared_error(y_test, predicted_values)) 
   
    data = pd.DataFrame(
   {
      "Purpose": ['Average difference between predicted values and actual values',
                  'Max features used', 'Number of trees used'
                  ],
      "Measure": [rmse, features, trees]  
   },
   index=['RMSE', 'Features', 'Trees']
)  
    return data
    
    
    
    