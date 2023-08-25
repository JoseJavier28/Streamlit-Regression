# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 19:14:55 2023
Performs multi linear equation and each function returns a piece of the process
@author: jjavi
"""

import statsmodels.formula.api as smf
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

def dummy_selection(df, job):
     # Takes a dataframe and a job_converted to filter for dummy dropped
     # Returns new dataframe for only linear regression model
    column_dropped = 'Data_Engineer'
    alternative = 'ML_Engineer'
    if job != column_dropped:
      newDf = df.drop(column_dropped, axis = 1)
    else:
      newDf = df.drop(alternative, axis = 1)   
    return newDf

def get_model(df, job_chosen):
    # Returns multi linear regression model object
    # df is the new dataframe returned previously with the dropped job dummy
    # job is the job chosen 
    # Isolating columns for analysis
    colnames = df.columns.values
    colnames = list(colnames[3:6])
    # creating formula of predictors by concatenating
    columns = 'salary_in_usd~experience_level+company_size'
    for col in colnames:
        columns+='+' + col
            
     # Getting the multiple regression object
    regression_object_multiple= smf.ols(columns, data=df)       
    # Fitting the values into the model
    regression_linear_model= regression_object_multiple.fit()        
    return regression_linear_model        
            
    
def get_prediction(lm, job_chosen, experience, size, otherJob1, otherJob2): 
    # Returns the prediction 
    # Given linear object and predictors chosen
    prediction = lm.predict({'experience_level': experience, 'company_size': size, job_chosen: 1, otherJob1: 0, otherJob2: 0})
    return int(prediction * 1000) 



def getEquation(lm, job_chosen):
    # Retrieving coefficients to output full equation on stremlit
    #manipulate data frame
    params = lm.params
    # index for job chosen param 
    if job_chosen == 'Data_Analyst':
        index = 3
    elif job_chosen == 'Data_Engineer':
        index = 4
    elif job_chosen == 'Data_Scientist':
        index = 4
    elif job_chosen == 'ML_Engineer':
        index = 5
    # concatenates coef with the chosen job coef
    current_equation = 'Estimated Salary = {:.2f} + {:.2f}(Experience Level) + {:.2f}(Company Size) + {:.2f}({}=1)'.format(params[0], params[1], params[2], params[index], job_chosen)
    return current_equation   

def getMetricDataframe(df, lm):
    # computing the Residual Standard Error of the equation and Rmse
    # Returns dataframe with metrics formatted along with explenation
    #gets new dataframe and linear models as parameters
    predicted_values = lm.predict() 
    actual_values = df['salary_in_usd']
    sse= np.sum((actual_values-predicted_values)**2)
    rse = np.sqrt (sse/ (len(df['salary_in_usd'])-1-1))
    
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    coe_variation = rse/np.mean(actual_values) * 100
    data = pd.DataFrame(
   {
      "Purpose": ['Average deviation between real values and the true regression line(The Mean)','Average difference between predicted values and actual values', 'Percenatge Level of dispersion around the mean'],
      "Measure": [rse, rmse, coe_variation]  
   },
   index=['Residual Standard Error', 'Root Mean Squared Error', 'Magnitude']
)  
    return data




