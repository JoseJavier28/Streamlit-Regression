# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:31:21 2023

@author: jjavi
"""
# Importing Libraries
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from matplotlib import pyplot as plt

import squarify
import statsmodels.formula.api as smf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import country_converter as coco

# Custom Files
import linear_regression as lr
import user_conversion as uc
import decision_tree as dt
import ccp_tree as ccp
import forest_bagging as fb
import rf_cv as rf

# Reading both dataframes
df = pd.read_csv("Data_Science_Salaries.csv")
df2 = pd.read_csv("Data_Science_Salaries_addon.csv")
                              

                                    # Data Cleaning
# Dropping redundant columns
df = df.drop(['Unnamed: 0', 'salary', 'salary_currency'], axis = 1) 

#Converting country codes to full country name
df['employee_residence'] = coco.convert(names = df['employee_residence'], to = 'name_short')
df['company_location'] = coco.convert(names = df['company_location'], to = 'name_short')

# Using cleaned dataframe for after uploading map
kept_countries = ('United States', 'United Kingdom', 'Canada', 'Germany', 'India', 'France', 'Spain', 'Greece', 'Japan')
clean_df = df[df.employee_residence.isin(kept_countries)]

# Keeping only a subset of the jobs
kept_jobs = ('Data Scientist', 'Data Engineer', 'Data Analyst',  'Machine Learning Engineer', 'Data Science Manager', 'Research Scientist', 'Data Architect', 'Data Science Consultant',  'Director of Data Science')
clean_df= clean_df[df.job_title.isin(kept_jobs)]

# Filtered dataframe for regression models
jobs = ('Data Analyst', 'Data Scientist', 'Data Engineer', 'Machine Learning Engineer')
filtered_df = clean_df[clean_df['job_title'].isin(jobs)]
# filtering out expert category
filtered_df = filtered_df[filtered_df['experience_level'] != 'EX']
# filtering for only the US
filtered_df = filtered_df[filtered_df['company_location'] == 'United States']
# only keeping desired columns for statistical models
filtered_df = filtered_df[['job_title', 'experience_level', 'company_size', 'salary_in_usd']]

# Merged Kaggle df with add on df using concat to append values
merged_df = pd.concat([df2, filtered_df], axis=0)
# removing outliers on the merged df
merged_df = merged_df[merged_df['salary_in_usd'] > 40000]
merged_df = merged_df[merged_df['salary_in_usd'] < 300000]

                # Converting to dummy variables without dropping column
# converting jobs using one hot encoding
df_dummies=pd.get_dummies(merged_df,columns=['job_title'], drop_first = False)
# converting salaries to the thousands 
df_dummies['salary_in_usd'] = df_dummies['salary_in_usd'] / 1000
# renaming job title columns
df_dummies.rename (columns ={'job_title_Data Analyst':'Data_Analyst','job_title_Data Engineer':'Data_Engineer', 'job_title_Data Scientist': 'Data_Scientist', 'job_title_Machine Learning Engineer': 'ML_Engineer'},inplace=True)

# encoding categorical variables and replacing them with integer
conversion = {'company_size': {'L': 2, 'M': 1, 'S':0}}
df_dummies =  df_dummies.replace(conversion)
conversion = {'experience_level': {'SE': 2, 'MI': 1, 'EN':0}}
df_dummies =  df_dummies.replace(conversion)


                            # Main Body Dashboard
st.title('Data Science :green[$alaries] :chart_with_upwards_trend:' )
st.markdown('Predict Salaries Based on Career, Experience Level and Company Size')


# tab creation 
tab_prediction, tab_data, tab_visuals, tab_summary = st.tabs(['Prediction :arrows_counterclockwise:', 'Data', 'Visuals', 'Summary :clipboard:'])

                            # Data tab
with tab_data:
    st.subheader('Data Science Job Salaries')
    st.write('https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries')
    st.write(clean_df.head(10))
    st.subheader('Data Collected')
    st.write(df2.head(10))
    st.subheader('Merged Dataframe') 
    # Data for the metrics
    data = {'Count': [merged_df.shape[0]], 'Columns': [merged_df.shape[1]], 'Min': merged_df['salary_in_usd'].min(), 'Average': merged_df['salary_in_usd'].mean().astype(int), 'Max': merged_df['salary_in_usd'].max()}
    df_metrics = pd.DataFrame(data) # Creating metrics dataframe
    st.write(df_metrics)
    st.write(merged_df.head(10))
    st.subheader('Encoded')
    st.write(df_dummies.head(10))

                            # visuals tab
with tab_visuals:

   
# Dual Axis graph creation
    
    df_dual = filtered_df[['job_title', 'experience_level', 'salary_in_usd']].groupby(['job_title', 'experience_level']).mean().reset_index()
    # Title
    st.write('Average Salary Based on Experience in the U.S.')
    fig, ax1 = plt.subplots(figsize = (10, 5.5)) 
    ax1.grid(False) # removing second grid line
# Two separate graphs that have the same axis
    g =  sns.barplot(data = df_dual, x = 'job_title', y = 'salary_in_usd', color = 'lightgrey', ax = ax1, errorbar = None) # plots the first set of data, and sets it to ax1. 
    l = sns.lineplot(data = df_dual, x = 'job_title', y = 'salary_in_usd', marker = 'o', hue = 'experience_level', ax = ax1, palette = 'Set1', estimator= None, linewidth = 3, style = 'experience_level') # plots the second set, and sets to ax2. 
    l.tick_params(right=False)
    sns.set(rc = {'figure.figsize':(15,15)})
    sns.set(font_scale=.8)
    ax1.set_xlabel('Job Type')
    ax1.set_ylabel('Average Salary', color='white')
    sns.set_style('dark')
    plt.style.use('dark_background')
    sns.move_legend(l, "upper left")
    st.pyplot(plt, use_container_width=True); # shows the plot.
  
    

    # Removing outliers for map salaries
    map_df = df[df['salary_in_usd'] < 200000] 
    # Grouping by country to get the average of salaries
    df_salary_percountry = map_df[['company_location', 'salary_in_usd']].groupby(['company_location']).mean().reset_index()
    # Converting to no decimal places
    df_salary_percountry['salary_in_usd'] = df_salary_percountry['salary_in_usd'].astype(int)

# Map creation using original dataframe  
    fig = px.choropleth(df_salary_percountry, 
                    color='salary_in_usd', 
                     locations='company_location', 
                     locationmode='country names', 
                     title='Average Salaries Per Country',
                     projection = 'natural earth',
                     color_continuous_scale='algae_r')
        # updating background color       
    fig.update_layout(geo_bgcolor='black')
    st.plotly_chart(fig, use_container_width=True)  

# Scatterplot Creation
    st.write('Salary Based on Job and Company Size')
    # Filtering values into a new data frame
    df_scatter = filtered_df[['job_title', 'company_size', 'salary_in_usd']]
    # Replacing column names
    df_scatter.replace('Machine Learning Engineer', 'ML Engineer', inplace = True)
    # removing outlier
    df_scatter = df_scatter[df_scatter['salary_in_usd'] < 300000]
     #  a scatter plot with the swarm plot style
    s = sns.catplot(data=df_scatter, kind="swarm", x="job_title", hue_order = ['S', 'M', 'L'], y="salary_in_usd", hue="company_size", height = 6,palette = 'RdPu')
    sns.despine(right = False, top = False)
    plt.xlabel('Job Title')
    plt.ylabel('Salary')    
    plt.xticks(rotation = 45, fontsize = 9)
    plt.show()
    st.pyplot(plt, use_container_width=True); # shows the plot.
    
# Box plot Creation 
    st.write('Salary Based on Jobs and Year')
    df_box = df[['job_title', 'work_year', 'salary_in_usd']]
    # filtering plot dataframe
    df_box = df_box[df_box['job_title'].isin(jobs)]
    # removing outlier
    df_box = df_box[df_box['salary_in_usd'] < 300000]
    # Replacing column names
    df_box.replace('Machine Learning Engineer', 'ML Engineer', inplace = True)
    # box plot syntax 
    sns.catplot(data=df_box, x="salary_in_usd", y="job_title", hue="work_year", kind="box", height=9, hue_order = [2020, 2021, 2022])
    sns.despine(right = False, top = False)
    plt.xlabel('Salary')
    plt.ylabel('Jobs')
    st.pyplot(plt, use_container_width=True); # shows the plot.

# Heatmap creation
    # filtering dataframe
    st.write('Remote Rate for Job Types')
    # filtering jobs for heatmap
    
    df_heatmap = clean_df.groupby(['work_year', 'job_title']).mean()['remote_ratio'].reset_index()
   
    
    # creating labels for remote ratio into category to later make it numerical
    df_heatmap['remote'] = pd.qcut( df_heatmap['remote_ratio'], 5,labels = ['Non Remote', 'Some Remote', 'Half Remote', 'Mostly Remote', 'Fully Remote'])
    # Creating codes for categories to use in heat map
    df_heatmap['remote_code'] = df_heatmap['remote'].cat.codes
    # pivoting values for better readability
    df_heatmap = df_heatmap.pivot(index='work_year', columns='job_title', values='remote_code')
    # renaming job title columns
    df_heatmap.rename (columns ={'Data Science Consultant':'DS Consultant', 'Machine Learning Engineer': 'ML Engineer', 'Data Science Manager': 'DS Manager', 'Director of Data Science': 'DS Director', 'Research Scientist': 'DS Researcher'},inplace=True)
    # replacing NAN values with 0
    df_heatmap.iloc[0,1] = 0
    df_heatmap.iloc[2,3] = 0
    #heatmap syntax
    plt.figure(figsize=(9,5));
    ax = sns.heatmap(df_heatmap, cmap = 'Blues')
    plt.xlabel('Job Title')
    plt.ylabel('Work Year')
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0, 1, 2, 3, 4]) # choosing categories for remote type
    colorbar.set_ticklabels(['Non Remote', 'Some Remote', 'Half Remote', 'Mostly Remote', 'Fully Remote'])
    # Rotating x axis
    plt.xticks(rotation = 35)
    st.pyplot(plt, use_container_width=True); # shows the plot.

    
                                # Prediction Tab
with tab_prediction:                             
# categories to choose from
    job_options = ('Data Analyst', 'Data Scientist', 'Data Engineer', 'ML Engineer')         
    companySize_options = ('Small', 'Medium', 'Large')
    experience_options = ('Junior', 'Intermediate', 'Senior')
    model_options = ('Linear Regression', 'Decision Tree', 'CCP Tree', 'Forest(Bagging)','RF-CV' )
    
# saving display
    job_user = st.selectbox('Career',job_options)
    experience_user = st.selectbox('Experience Level', experience_options)
    companySize_user = st.selectbox('Company Size', companySize_options)
# Regression models selection
    models_selected = st.multiselect('Regression Models', model_options) 
# for comparison to run
    l = 'Linear Regression'   
    t = 'Decision Tree'
    c = 'CCP Tree'
    b = 'Forest(Bagging)'
    fc = 'RF-CV'
# Calls user conversion functions
    job_predictor = uc.convert_jobs(job_user) 
    experience_predictor = uc.convert_experience(experience_user)
    size_predictor = uc.convert_companySize(companySize_user)

# Run application 
    start = st.button('Predict Salary')

# Boolean values to show summary for regression models chosen 
# for summary tab
    linearR = False
    decisionT = False
    ccpTree = False
    bagging = False
    randomF = False
    job_list = ['Data_Analyst', 'Data_Scientist', 'Data_Engineer', 'ML_Engineer']
    # Prediction Logic
    if start:
        # Linear Model
        if l in models_selected:
           linearR = True # Linear regression has been selected for summary
           newdf = lr.dummy_selection(df_dummies, job_predictor) # new dropped column df
           # Get jobs not selected by looping to dropped dummy dataframe and getting not selected jobs
           not_selected1 = ''
           not_selected2 = ''
           counter = 0      
           for i in job_list:
               if  counter < 4 and i != job_predictor and i in newdf.columns and not_selected1 =='':
                   not_selected1 = i                 
               elif  counter < 4 and i != job_predictor and i in newdf.columns and not_selected2 =='':
                   not_selected2 = i
               counter +=1 
           lr_model = lr.get_model(newdf, job_predictor) # get linear model function 
           #  return the predicted value using getprediction function
           prediction_lr = lr.get_prediction(lr_model, job_predictor, experience_predictor, size_predictor, not_selected1, not_selected2)
           st.write('_Linear Regression_ Salary Prediction is :green[${:,}] '.format(prediction_lr))
        # Dataframe for trees will only run once
        if t in models_selected or c in models_selected or b in models_selected  or fc in models_selected:
            # Dividing test and training data for tree setup if the corresponding
            # regression models are selected
             predictors = [ 'experience_level','company_size', 'Data_Analyst', 'Data_Scientist', 'Data_Engineer', 'ML_Engineer']
             X_train, X_test, y_train, y_test= train_test_split (df_dummies[predictors],df_dummies['salary_in_usd'], test_size=0.2, random_state=1)  
        # Decision Tree
        if t in models_selected:
            decisionT = True
            # Getting tree model
            tree_model = dt.getTreeObject(X_train, y_train) 
            # Getting prediction from function
            prediction_tree = dt.get_prediction(tree_model, job_predictor, experience_predictor, size_predictor)
            st.write('_Decision Tree_ Salary Prediction is :green[${:,}] '.format(prediction_tree)) 
        # CCP tree
        if c in models_selected:
            ccpTree = True
            # Getting the tree model
            ccp_model, alpha = ccp.getTreeObject(X_train, y_train, X_test, y_test)
            ccp_prediction = ccp.get_prediction(ccp_model, job_predictor, experience_predictor, size_predictor)
            st.write('_CCP Tree_ Salary Prediction is :green[${:,}] '.format(ccp_prediction)) 
        if b in models_selected:
            bagging = True
            fb_model, trees =fb.getTreeObject(X_train, y_train)
            fb_prediction = fb.get_prediction(fb_model, job_predictor, experience_predictor, size_predictor)
            st.write('_Non Random Forest Bagging Method_ Salary Prediction is :green[${:,}] '.format(fb_prediction))
        if fc in models_selected:
            randomF = True
            rf_model, features, trees = rf.getTreeObject(X_train, y_train)
            rf_prediction = rf.get_prediction(rf_model, job_predictor, experience_predictor, size_predictor)
            st.write('_Random Forest CV_ Salary Prediction is :green[${:,}] '.format(rf_prediction))
            
            
                                   # Summary Tab
with tab_summary:
    if linearR:
        st.subheader('Linear Regression')
        # Getting equation
        st.write(':orange[Equation]')
        equation =lr.getEquation(lr_model, job_predictor)
        st.write('###### {}'.format(equation)) 
        st.write(':orange[Evaluation Metrics]')
        # Getting metrics summary
        st.write(lr.getMetricDataframe(newdf, lr_model))
        st.write(lr_model.summary()) 
    if decisionT:
         st.subheader('Decision Tree')
         st.write(':orange[Evaluation Metrics]')
         # Getting metrics summary
         st.write(dt.getMetricDataframe(tree_model, X_test, y_test))
         # Plotting Tree   
         plt.figure(figsize = (15,15))
         tree.plot_tree(tree_model, rounded= True, feature_names=X_train.columns)
         st.pyplot(plt, use_container_width=True)
    if ccpTree:
        st.subheader('Cost Complexity Prunned Tree')
        st.write(':orange[Evaluation Metrics]') 
        st.write(ccp.getMetricDataframe(ccp_model, X_test, y_test, alpha)) 
        # Plotting Tree   
        plt.figure(figsize = (15,15))
        tree.plot_tree(ccp_model, rounded= True, feature_names=X_train.columns, fontsize = 10)
        st.pyplot(plt, use_container_width=True)
    if bagging:
       st.subheader('Non Random Forest(Bagging)') 
       st.write(':orange[Evaluation Metrics]') 
       st.write(fb.getMetricDataframe(fb_model, X_test, y_test, trees))
    if randomF:
        st.subheader('Random Forest Cross Validation')
        st.write(':orange[Evaluation Metrics]') 
        st.write(rf.getMetricDataframe(rf_model, X_test, y_test, features, trees))
        
        