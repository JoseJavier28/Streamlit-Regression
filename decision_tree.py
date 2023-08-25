# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 19:14:59 2023
Constructs a decision tree 
@author: jjavi
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def getTreeObject(X_train, y_train):
    # Returns model of tree given split data
         hyperparam_grid = {
    'max_depth': np.arange(2,11), # testing depth from 2 to 10
    'min_samples_split':[0.1, 0.15, 0.2],
    'min_samples_leaf':[0.05, 0.1, 0.15], 
    'min_impurity_decrease': [0, 0.0005, 0.001, 0.005, 0.01, 0.05]
    }   
   
         #Getting cv object
         gridSearch = GridSearchCV(DecisionTreeRegressor(), hyperparam_grid, cv=5, scoring='neg_mean_squared_error')
         gridSearch.fit(X_train, y_train)
         # best parameters with lowest MSE
         max_depth = gridSearch.best_params_['max_depth']
         min_split = gridSearch.best_params_['min_samples_split']
         min_leaf = gridSearch.best_params_['min_samples_leaf'] 
         min_impure = gridSearch.best_params_['min_impurity_decrease']
         # Getting model
         decision_tree = DecisionTreeRegressor(max_depth= max_depth, min_samples_split= min_split, min_samples_leaf= min_leaf, min_impurity_decrease= min_impure)
         tree_model = decision_tree.fit(X_train, y_train)
         return tree_model
         
        
def get_prediction(tm, job, experience, size): 
     # Returns the prediction 
     # Given tree object and predictors chosen
     # Equation job predictor logic 
     job_list = ['Data_Analyst', 'Data_Scientist', 'Data_Engineer', 'ML_Engineer']
     da = 0
     ds = 0
     de = 0
     ml = 0
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
   

def getMetricDataframe(tm, X_test, y_test):
    # computing the Rmse
    # Returns dataframe with metrics formatted along with explenation
    #tree model and dataframe split as parameters
    predicted_values = tm.predict(X_test) 
    rmse = np.sqrt(mean_squared_error(y_test, predicted_values)) 
    leafs = tm.get_n_leaves()
    depth = tm.get_depth()
    params = tm.get_params() 
    
    data = pd.DataFrame(
   {
      "Purpose": ['Average difference between predicted values and actual values',
                  'Number of terminal nodes',
                  'Layers of internal nodes', 'Minumum number of samples required to be a leaf node',
                  'Minimum number of samples required to split internal node',
                  'Minimum decrease in residual sum of squares that is deemed acceptable',
                  'Algorithm used to split each node'
                  ],
      "Measure": [rmse, leafs, depth, params['min_samples_leaf'], params['min_samples_split'], params['min_impurity_decrease'], params['splitter'] ]  
   },
   index=['RMSE', 'Number of Leaves', 'Max Depth', 'Min Samples Leaf', 'Min Samples Split', 'Min Impurity Decrease', 'Splitter']
)  
    return data



     
        

       
           