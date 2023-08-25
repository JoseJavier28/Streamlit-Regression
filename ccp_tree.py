# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 13:44:04 2023
Constructs CCP Tree
@author: jjavi
"""
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

def getTreeObject(X_train, y_train, X_test, y_test):
    # Returns model object of tree given split data
    reg_tree_unprunned= DecisionTreeRegressor(random_state=1)
    # Getting alpha paths
    path= reg_tree_unprunned.cost_complexity_pruning_path(X_train, y_train)
    alphas= path['ccp_alphas']    
    # Getting mse scores based on alphas for the trees
    mse_scores=[]
    for i in alphas:
        treeloop= DecisionTreeRegressor(ccp_alpha=i,random_state=1)
        treeloop.fit(X_train, y_train)
        y_test_predicted=treeloop.predict(X_test)
        # Saving mse scores
        mse_scores.append(mean_squared_error( y_test,y_test_predicted))
        # getting index where lowest mse happens
    indexmin=mse_scores.index(min(mse_scores))
        # best tree based on the lowest mse
    reg_tree_prunned= DecisionTreeRegressor(ccp_alpha= alphas[indexmin], random_state=1)
    ccp_tree_object = reg_tree_prunned.fit(X_train, y_train)
    return ccp_tree_object, alphas[indexmin]


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
 
def getMetricDataframe(tm, X_test, y_test, alpha):
    # computing the Rmse
    # Returns dataframe with metrics formatted along with explanation
    #tree model and dataframe split as parameters
    predicted_values = tm.predict(X_test) 
    rmse = np.sqrt(mean_squared_error(y_test, predicted_values)) 
    leafs = tm.get_n_leaves()
    depth = tm.get_depth()
    data = pd.DataFrame(
   {
      "Purpose": ['Average difference between predicted values and actual values',
                  'Prunning intensity value that determines the size of the tree',
                  'Number of terminal nodes', 'Layers of internal nodes'
                  
                  ],
      "Measure": [rmse, alpha, leafs, depth, ]    
   },
   index=['RMSE', 'Alpha', 'Number of Leaves', 'Max Depth']
)  
    return data