# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 15:24:00 2023
Constructs forest bagging tree
@author: jjavi
"""


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

def getTreeObject(X_train, y_train):
    # Returns model of tree given split data
    
    # number of trees used
    number_of_trees=np.arange(50,501,10)
    # will return the best collection of trees with the lowest mse
    mse_bagging_oob_scores=[]
    for i in number_of_trees:
        bag_loop= RandomForestRegressor(n_estimators = i, oob_score= True, max_features=None, random_state=1)
        bag_loop.fit(X_train, y_train)
        mse_bagging_oob_scores.append(mean_squared_error(y_train, bag_loop.oob_prediction_))
    # index with lowest oob mse
    indexmin_bagging= mse_bagging_oob_scores.index(min(mse_bagging_oob_scores))
    # numbers of trees that resulted in lowest mse
    trees = number_of_trees[indexmin_bagging]
    # Building model with selected trees
    bagging_forest= RandomForestRegressor(n_estimators= trees, max_features=None, random_state=1)
    tree_model = bagging_forest.fit(X_train, y_train)
    return tree_model, trees
    
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
 
def getMetricDataframe(tm, X_test, y_test, trees):
    # computing the Rmse
    # Returns dataframe with metrics formatted along with explanation
    #tree model and dataframe split as parameters
    predicted_values = tm.predict(X_test) 
    rmse = np.sqrt(mean_squared_error(y_test, predicted_values)) 
   
    data = pd.DataFrame(
   {
      "Purpose": ['Average difference between predicted values and actual values',
                  'All features used', 'Number of trees used'
                  ],
      "Measure": [rmse, 'max', trees]  
   },
   index=['RMSE', 'Features', 'Trees']
)  
    return data
    