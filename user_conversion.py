# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 19:49:08 2023
Matches predictors to user input
@author: jjavi
"""
#

def convert_jobs(job):
    # Takes in job chosen and assigns it dataframe equivalent
    # returns dataframe job chosen equivalent
    job_converted = ''
    if job == 'Data Analyst':
        job_converted = 'Data_Analyst'
    elif job == 'Data Scientist':
        job_converted = 'Data_Scientist'
    elif job == 'Data Engineer':
        job_converted = 'Data_Engineer'
    elif job == 'ML Engineer':
        job_converted = 'ML_Engineer'
    return job_converted 

def convert_experience(experience):
    # Takes in experience chosen and matches it to dataframe equivalent
    # returns dataframe chosen equivalent
    exp_converted = -1
    if experience == 'Junior':
        exp_converted = 0
    elif experience == 'Intermediate':
        exp_converted = 1
    elif experience == 'Senior':
        exp_converted = 2
    return exp_converted 

def convert_companySize(size):
    # Takes in the user size and matches it to dataframe equivalent 
    # Returns dataframe chosen equivalent 
    size_converted = -1
    if size == 'Small':
        size_converted = 0
    elif size == 'Medium':
        size_converted = 1
    elif size == 'Large':
        size_converted = 2
    return size_converted 