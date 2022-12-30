#!/usr/bin/env python
# coding: utf-8

# <img src = "https://github.com/barcelonagse-datascience/academic_files/raw/master/bgsedsc_0.jpg">
# $\newcommand{\bb}{\boldsymbol{\beta}}$
# $\DeclareMathOperator{\Gau}{\mathcal{N}}$
# $\newcommand{\bphi}{\boldsymbol \phi}$
# $\newcommand{\bx}{\boldsymbol{x}}$
# $\newcommand{\bu}{\boldsymbol{u}}$
# $\newcommand{\by}{\boldsymbol{y}}$
# $\newcommand{\whbb}{\widehat{\bb}}$
# $\newcommand{\hf}{\hat{f}}$
# $\newcommand{\tf}{\tilde{f}}$
# $\newcommand{\ybar}{\overline{y}}$
# $\newcommand{\E}{\mathbb{E}}$
# $\newcommand{\Var}{Var}$
# $\newcommand{\Cov}{Cov}$
# $\newcommand{\Cor}{Cor}$

# ## Customer Churn Analysis and Classification
# 
# With the rapid development of telecommunication industry, the service providers are inclined more towards expansion of the subscriber base. To meet the need of surviving in the competitive environment, the retention of existing customers has become a huge challenge. It is stated that the cost of acquiring a new customer is far more than that for retaining the existing one. Therefore, it is imperative for the telecom industries to use advanced analytics to understand consumer behavior and in-turn predict the association of the customers as whether or not they will leave the company.
# 
# You are given a dataset: each row represents a customer and each column contains attributes related to customer as described:
# 
# + Churn (target): 1 if customer cancelled service, 0 if not
# + AccountWeeks: number of weeks customer has had active account
# + ContractRenewal: 1 if customer recently renewed contract, 0 if not
# + DataPlan: 1 if customer has data plan, 0 if not
# + DataUsage: gigabytes of monthly data usage
# + CustServCalls: number of calls into customer service
# + DayMins: average daytime minutes per month
# + DayCalls: average number of daytime calls
# + MonthlyCharge: average monthly bill
# + OverageFee: largest overage fee in last 12 months
# + RoamMins: average number of roaming minutes
# 
# 
# You are asked to **develop an algorithm** to be able to assess which are the customers that have the highest probability to churn. Besides that you will be asked to answer the 3 following questions:
# 
# 1. **What variables are contributing to customer churn?** 
# 2. **Who are the customers more likely to churn?**
# 3. **What actions can be taken to stop them from leaving?**
# 
# 
# You can follow those **steps** in your first implementation:
# 1. *Explore* and understand the dataset. 
# 2. Create extra variables if needed/possible
# 3. *Build* your model and test it on the same input data
# 4. Assess expected accuracy using *cross-validation*
# 5. Tune the hyperparameters of your model
# 6. Repeat steps 4 and 5 until you find the best model possible
# 7. Answer the questions asked

# ## Main criteria for grading
# + Algorithm implemented
# + AUC score given
# + At least Random Forest and Xgboost are used
# + Data preparation and exploration
# + Hyperparameter optimization 
# + Cross-validation used
# + Code is combined with neat and understandable commentary, with some titles and comments 

# In[1]:


# import libraries
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import *
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_text

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x: '%.2f' % x)
np.set_printoptions(suppress=True)


# #### 1. *Explore* and understand the dataset. 

# In[2]:


# read data
df = pd.read_csv('churn_data.csv', sep=';')


# In[3]:


df.head()


# In[44]:


df['Churn']


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


churn_count_df = df.Churn.value_counts().reset_index().rename(columns={'Churn': 'Count', 'index': 'Churn'})
churn_count_df.Churn = churn_count_df.Churn.astype(object)


# In[7]:


fig = px.bar(churn_count_df, x="Churn", y="Count", color='Churn', title="Customer Churn")
fig.show()


# Based on barplot above we can observe that 483 customers have churned from total customer base of 3333, which is around 14.5%. Although data is imbalanced still the propotion is not too critical and we can balance it in order to get better performance.

# 
# #### 2. Create extra variables if needed/possible
# 

# In[8]:


# define feature and target objects
X = df.drop(columns=['Churn'])
y = df.Churn


# In[9]:


# # train test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)


# #### 3. *Build* your model and test it on the same input data

# # Decision Tree

# In[10]:


# train the model using DecisionTree classifier
clf_tree = DecisionTreeClassifier(max_depth=4, random_state=1)
clf_tree.fit(X, y)

# predict test set labels
y_pred_dt = clf_tree.predict(X)


# In[11]:


print('AUC Decision Tree: ', roc_auc_score(y, y_pred_dt))


# In[12]:


print('Accuracy Score Decision Tree: ', accuracy_score(y, y_pred_dt))


# In[13]:


print(classification_report(y, y_pred_dt))


# In[14]:


from sklearn.tree import plot_tree
plt.figure(figsize=(25,10))
plot_iris = plot_tree(clf_tree,
                      feature_names=X.columns.tolist(), 
                      class_names=['not churned', 'churned'], 
                      filled=True, 
                      rounded=True, 
                      fontsize=14)


# # Random Forest

# In[15]:


# Instantiate rf
rf = RandomForestClassifier(max_depth=9, random_state=0)
             
# Fit rf to the training set    
rf.fit(X, y) 
 
# Predict test set labels
y_pred_rf = rf.predict(X)


# In[16]:


print('AUC Random Forest: ', roc_auc_score(y, y_pred_rf))


# In[17]:


print('Accuracy Score: ', accuracy_score(y, y_pred_rf))


# In[18]:


print(classification_report(y, y_pred_rf))


# In[19]:


# Create a pd.Series of features importances
importances = pd.Series(data=rf.feature_importances_,
                        index= X.columns.tolist())
 
# Sort importances
importances_sorted = importances.sort_values()
 
# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()


# In[20]:


# Extract single tree
estimator = rf.estimators_[1]

plt.figure(figsize=(25,10))
plot_cancer = plot_tree(estimator,
                      feature_names=X.columns.tolist(), 
                      class_names=['not churned', 'churned'], 
                      filled=True, 
                      rounded=True, 
                      fontsize=14)


# # XGBoost

# In[21]:


xgb_model = xgb.XGBClassifier()

# Fit ada to the training set
xgb_model.fit(X, y)

# Compute the probabilities of obtaining the positive class
y_pred_xg = xgb_model.predict(X)


# In[22]:


print('AUC XGBoost: ', roc_auc_score(y, y_pred_xg))


# In[23]:


print('Accuracy Score: ', accuracy_score(y, y_pred_xg))


# In[24]:


print(classification_report(y, y_pred_rf))


# In[25]:


print('AUC Decision Tree: ', roc_auc_score(y, y_pred_dt))
print('AUC Randome Forest: ', roc_auc_score(y, y_pred_rf))
print('AUC XGBoost: ', roc_auc_score(y, y_pred_xg))


# ### We have trained 3 model from tree-based family of algorithms. Decision Tree has achieved AUC of 78%, while Random Forest ahs outperformed DT by 10%, which is expected RF includes not just one tree as DT but 'ensemble' of trees. Highest AUC was achieved with XGBoost around 99.5% but this performance measure if for the same dataset we used to train so it cannot represent the level of performance we may get on real production generated data. In the next step we will conduct cross validation and again measure performance to identify more realistic performance of the models

# ### 4. Assess expected accuracy using *cross-validation*

# In[26]:


clf_tree = DecisionTreeClassifier(max_depth=4, random_state=1)
scores_dt = cross_val_score(clf_tree, X, y, cv=5, scoring='roc_auc')


# In[27]:


rf = RandomForestClassifier(max_depth=9, random_state=0)
scores_rf = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')


# In[28]:


xgb_model = xgb.XGBClassifier(n_estimators=10000, max_depth=20, eval_metric='auc')
scores_xgb = cross_val_score(xgb_model, X, y, cv=5, scoring='roc_auc')


# In[29]:


print('Cross Validated AUC Decision Tree: ', scores_dt.mean())
print('Cross Validated AUC Randome Forest: ', scores_rf.mean())
print('Cross Validated AUC XGBoost: ', scores_xgb.mean())


# ### Interestingly, performance of XGBoost model has dropped in case we use Cross Validation. This was expected and currently Random Forest model showcases the highest performance with AUC of 90.2%. It seems RF model generalizes better and XGBoost overfits when we ccompare results from Cross Validation to the result from previous step.

# ### 5. Tune the hyperparameters of your model

# # As we  have observed from previous steps and having general knowledge that DT is not as powerful algorigthm as RF or XGBoost we will no longer consider it in our future analysis and with conduct hyperparameter optimization for RF and XGBoost

#  # Random Forest

# In[30]:


# Define the dictionary 'params_rf'
param_grid = {
    'n_estimators':[100, 300, 600, 800, 1000, 1100],
    'max_features':['log2', 'auto', 'sqrt'],
    'min_samples_leaf':[2, 5, 10, 15, 20],
}

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV
 
# Create a based model
rf = RandomForestClassifier()

# Instantiate the grid search model
grid_search_rf = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2, scoring='roc_auc')

# Fit the grid search to the data
grid_search_rf.fit(X, y) 


# In[31]:


grid_search_rf.best_params_


# In[32]:


grid_search_rf.best_score_


# ## Performance of RF has increased slightly by utilizing Grid Search by around 0.1%

# # XGBoost

# In[33]:


# Define the dictionary 'params_rf'
param_grid = {
    'max_depth': range (4, 16, 2),
    'n_estimators': range(100, 1000, 100),
    'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001],
    'eval_metric': ['auc']
}

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV
 
# Create a based model
xgb_model = xgb.XGBClassifier()

# Instantiate the grid search model
grid_search_xgb = GridSearchCV(estimator = xgb_model, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2, scoring='roc_auc')

# Fit the grid search to the data
grid_search_xgb.fit(X, y) 


# In[34]:


grid_search_xgb.best_params_


# In[35]:


grid_search_xgb.best_score_


# # Grid Search has increased performance of XGBoost as well as we can observe. Currently, AUC of XGBoost is 90.2% and it is much closer to RF AUC.

# ### 6. Repeat steps 4 and 5 until you find the best model possible

# ## Optimization Step 4 repeat

# ## Random Forest

# In[36]:


rf_s6 = RandomForestClassifier(max_depth=9, max_features='auto', min_samples_leaf=5, n_estimators=90, random_state=0)
scores_rf_s6 = cross_val_score(rf_s6, X, y, cv=5, scoring='roc_auc')


# In[37]:


print('Cross Validated AUC Randome Forest: ', scores_rf_s6.mean())


# ## XGBoost

# In[38]:


xgb_model_s6 = xgb.XGBClassifier(n_estimators=350, max_depth=4, learning_rate=0.004, eval_metric='auc')
scores_xgb_s6 = cross_val_score(xgb_model_s6, X, y, cv=5, scoring='roc_auc')


# In[39]:


print('Cross Validated AUC XGBoost: ', scores_xgb_s6.mean())


# # We have improved performance of RF and XGBoost models by utilizing information from optimal hypermateres and achieved superior performance compared to step 4

# ##### Grid Search Optimization Step 5 repeat

# ## Now let's narrow down the parameter grid to the values identified in previous run for hyperparameter tuning step 5. Fine-tuned hyperparameters identified for RF were {'max_features': 'sqrt', 'min_samples_leaf': 10, 'n_estimators': 100}

# In[40]:


# Define the dictionary 'params_rf'
param_grid = {
    'n_estimators':[50, 60, 70, 80, 90, 100, 110, 120, 130],
    'max_features':['log2', 'auto', 'sqrt'],
    'min_samples_leaf':[5, 7, 8, 9, 10, 11, 12, 13, 14, 15],
}
 
# Create a based model
rf = RandomForestClassifier()

# Instantiate the grid search model
grid_search_rf_s6 = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2, scoring='roc_auc')

# Fit the grid search to the data
grid_search_rf_s6.fit(X, y) 


# In[41]:


grid_search_rf_s6.best_params_


# In[42]:


grid_search_rf_s6.best_score_


# # After narrowing down Grid Search for RF we have achieved higher performance of around 90.5% which is by more than 0.2%

# ### XGBoost

# ## Let's try to optimize more XGBoost Grid Search from optimal hyperparameters found in step 5 {'eval_metric': 'auc', 'learning_rate': 0.005, 'max_depth': 4, 'n_estimators': 400}

# In[43]:


# Define the dictionary 'params_rf'
param_grid = {
    'max_depth': range (4, 16, 2),
    'n_estimators': range(300, 500, 25),
    'learning_rate': [ 0.01, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001],
    'eval_metric': ['auc']
}
 
# Create a based model
xgb_model = xgb.XGBClassifier()

# Instantiate the grid search model
grid_search_xgb_s6 = GridSearchCV(estimator = xgb_model, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2, scoring='roc_auc')

# Fit the grid search to the data
grid_search_xgb_s6.fit(X, y) 


# In[ ]:


grid_search_xgb_s6.best_params_


# In[ ]:


grid_search_xgb_s6.best_score_


# ## Again narrowing down hyperparameter range has helped as we have improved AUC of XGBoost model by more around 0.1% from step 5

# ## 7. Answer the questions asked

# 1. **What variables are contributing to customer churn?** 

# ## RF

# In[ ]:


rf_s6_f = rf_s6.fit(X, y)
# Create a pd.Series of features importances
importances = pd.Series(data=rf_s6_f.feature_importances_,
                        index= X.columns.tolist())
 
# Sort importances
importances_sorted = importances.sort_values()
 
# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()


# # XGBoost

# In[ ]:


xgb_model_s6_f = xgb_model_s6.fit(X, y)
# Create a pd.Series of features importances
importances = pd.Series(data=xgb_model_s6_f.feature_importances_,
                        index= X.columns.tolist())
 
# Sort importances
importances_sorted = importances.sort_values()
 
# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()


# ### Interestingly, the most dominant feature in case of RF seems to be average daytime minutes per month, whereas XGBoost has identified number of calls into customer service as the one ahving most predicitive power for churn prediction. However, average daytime minutes per month seems to be real important as well as XGBoost has detected it to be 2nd most important feature in the levels of the customer churn.
# 
# ### Average monthly bill and average number of roaming minutes also seems to have high impact on the customer churn.

# 2. **Who are the customers more likely to churn?**

# ### To understand which customers are more likely to churn we need to visualize sample of trees to have some understanding how each feature affect probability of churning i.e. increase/decrease 

# # RF

# In[ ]:


# Extract single tree
estimator = rf_s6.estimators_[2]

plt.figure(figsize=(25,10))
plot_cancer = plot_tree(estimator,
                      feature_names=X.columns.tolist(), 
                      class_names=['not churned', 'churned'], 
                      filled=True, 
                      rounded=True, 
                      fontsize=10,
                      max_depth=4)


# In[ ]:


tree_rules = export_text(rf_s6.estimators_[2], feature_names=list(X.columns),  max_depth=3, show_weights=True)


# In[ ]:


print(tree_rules)


# ## Above we can observe both decision path of RF 2nd estimator or tree. It has started from DayMins (average daytime minutes per month) which is consitent from previous steps and if average daytime minutes per month <= 261.35 we go to left branch of the tree where proportion of customer not churned is 2774 and churned 354. While is average daytime minutes per month > 261.35 dominant class is churned 106 customer have churned and and 99 not churned. If we continue with last branch if gigabytes of monthly data usage <= 0.325 then customer will be more likely to churn 101 churned vs 48 not churned. However, customer who had gigabytes of monthly data usage > 0.325 are much likely to churn as dominant class there is not churned 51 customer

# In[ ]:





# In[ ]:





# In[ ]:





# 3. **What actions can be taken to stop them from leaving?**

# + Churn (target): 1 if customer cancelled service, 0 if not
# + AccountWeeks: number of weeks customer has had active account
# + ContractRenewal: 1 if customer recently renewed contract, 0 if not
# + DataPlan: 1 if customer has data plan, 0 if not
# + DataUsage: gigabytes of monthly data usage
# + CustServCalls: number of calls into customer service
# + DayMins: average daytime minutes per month
# + DayCalls: average number of daytime calls
# + MonthlyCharge: average monthly bill
# + OverageFee: largest overage fee in last 12 months
# + RoamMins: average number of roaming minutes

# # To be done!!!
# 
# -  2. Create extra variables if needed/possible
# -  7. Answer the questions asked

# In[ ]:





# In[ ]:





# In[ ]:





# ### maybe useful>> https://towardsdatascience.com/visualizing-decision-trees-with-python-scikit-learn-graphviz-matplotlib-1c50b4aa68dc

# In[ ]:




