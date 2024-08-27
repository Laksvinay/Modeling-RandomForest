#!/usr/bin/env python
# coding: utf-8

# In[866]:


from platform import python_version
print("\n python version for Random forest regression analysis is ",python_version())


# In[867]:


print("Research Question: To what extent can patient lifestyle choices along with demographic and medical factors, be used to predict total hospital charges during their initial days in hospital using a Random Forest Regressor?")


# In[869]:


import os
import pandas as pd
import numpy as np
dmrgdata=pd.read_csv("C:/Users/laksh/Desktop/WGU-MS Data analytics/MSDA-D209-Datamining1/Task2/medical_clean.csv")
dmrgdata.head()


# In[870]:


dmrgdata.shape


# In[871]:


dmrgdata.info()


# In[872]:


# Renaming survey variables in a meaningful way
dmrgdata.rename(columns={
    "Item1": "Timely_admission",
    "Item2": "Timely_treatment",
    "Item3": "Timely_visits",
    "Item4": "Reliability",
    "Item5": "Treatment_hours",
    "Item6": "Options",
    "Item7": "Courteous_staff",
    "Item8": "Active_listening"
}, inplace=True)

print(dmrgdata.columns)


# In[873]:


# Check for null values
dmrgdata.isna().sum()


# In[874]:


# Check for duplicates
(dmrgdata.duplicated().sum())


# In[875]:


dmrgdata=dmrgdata.drop(['Customer_id','Interaction','UID','City','State','County','Zip','Lat','Lng','Population','TimeZone','Job'],axis=1)


# In[876]:


dmrgdata.info()


# In[877]:


#categorical features
categorical_features = dmrgdata.select_dtypes(include=['object']).columns.tolist()
print("Categorical Features:", categorical_features)


#quantitative variables
numeric_features= dmrgdata.drop(categorical_features,axis=1)
print("\n\nNumeric features:",numeric_features.columns)


# In[878]:


#Quantitative features heatmap
import seaborn as sns
sns.heatmap(data=numeric_features.corr(),annot=True,cmap='coolwarm')


# In[879]:


for feature in categorical_features:
    unique_values = dmrgdata[feature].unique()
    print(f"Unique values for '{feature}':")
    print(unique_values)


# In[880]:


# one-hot encoding for binary categorical variables 
binary_vars = ['ReAdmis','Soft_drink','HighBlood','Diabetes','Hyperlipidemia','Stroke','Overweight','Arthritis','BackPain','Anxiety','Allergic_rhinitis','Reflux_esophagitis','Asthma']
dmrg_encoded = pd.get_dummies(dmrgdata,columns=binary_vars, drop_first=True,dtype=int)



# In[882]:


dmrg_encoded.info()


# In[855]:


print(dmrg_encoded['HighBlood_Yes'].value_counts())
print("\n\n",dmrg_encoded['HighBlood_Yes'].head())


# In[883]:


# Display the updated DataFrame
print(dmrg_encoded.head)


# In[884]:


print(dmrg_encoded.head())


# In[833]:


dmrg_encoded.info()


# In[885]:


#Removing whitespaces
dmrg_encoded['Marital']=  dmrg_encoded['Marital'].str.replace(' ','')
dmrg_encoded['Gender']=  dmrg_encoded['Gender'].str.replace(' ','')
dmrg_encoded['Initial_admin']=  dmrg_encoded['Initial_admin'].str.replace(' ','')
dmrg_encoded['Complication_risk']=  dmrg_encoded['Complication_risk'].str.replace(' ','')
dmrg_encoded['Services']=  dmrg_encoded['Initial_admin'].str.replace(' ','')


print("\n\n",dmrg_encoded['Marital'].value_counts())
print("\n\n",dmrg_encoded['Gender'].value_counts())
print("\n\n",dmrg_encoded['Initial_admin'].value_counts())
print("\n\n",dmrg_encoded['Complication_risk'].value_counts())
print("\n\n",dmrg_encoded['Services'].value_counts())


# In[886]:


#Ordinal encoding for feature values that are in order
from sklearn.preprocessing import OrdinalEncoder
oenc=OrdinalEncoder()
dmrg_encoded['Complication_risk']=oenc.fit_transform(dmrg_encoded[['Complication_risk']])
dmrg_encoded['Area']=oenc.fit_transform(dmrg_encoded[['Area']])


# In[887]:


print(dmrg_encoded['Complication_risk'].head())
print(dmrg_encoded['Area'].head())


# In[888]:


# One-hot encoding for nominal categorical variables with more than two levels
dmrg_encoded = pd.get_dummies(dmrg_encoded,columns=['Marital','Gender','Initial_admin', 'Services'], drop_first=True,dtype=int)


# In[889]:


dmrg_encoded.info()


# In[890]:


#Export  preprocessed data
dmrg_encoded.to_csv("C:/Users/laksh/Desktop/WGU-MS Data analytics/MSDA-D209-Datamining1/Performance Assessment D209/task2/D209-task2-encodeddata.csv",index=False)


# In[891]:


#Examine the correlations in the dataset by using a heatmap

sns.heatmap(data=dmrg_encoded.corr(),annot=True,cmap='coolwarm')


# # Feature selection using Lasso regression 

# In[892]:


import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Prepare data
X1 = dmrg_encoded.drop("TotalCharge", axis=1).values
y1 = dmrg_encoded["TotalCharge"].values
names = dmrg_encoded.drop("TotalCharge", axis=1).columns

# Split the data into training and testing sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X1_train_scaled = scaler.fit_transform(X1_train)
X1_test_scaled = scaler.transform(X1_test)

# Feature selection using Lasso
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X1_train_scaled, y1_train).coef_

# Plotting the Lasso coefficients
plt.figure(figsize=(12, 8))
plt.bar(names, lasso_coef)
plt.xticks(rotation=45, ha='right')

# Add labels and title for clarity
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Lasso Regression Coefficients')
plt.show()


# # Feature selection using SelectKBest

# In[894]:


#Feature selection to reduce the dimensionality, based on statistical tests using selectKBest

from sklearn.feature_selection import SelectKBest, f_regression

# Separate features and target
X2 = dmrg_encoded.drop(columns=['TotalCharge'])
y2 = dmrg_encoded['TotalCharge']

# f_regression univariate linear regression test that computes the F-statistic for each feature
# k='all'  indicates that all features will be scored, but no features are removed

selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(X2, y2)

# Get the scores and p-values for each feature
scores = selector.scores_
p_values = selector.pvalues_

# Create a data frame to display feature scores and p-values
feature_scores_df = pd.DataFrame({
    'Feature': X2.columns,
    'Score': scores,
    'P-Value': p_values
}).sort_values(by='P-Value')

# Filter the DataFrame for features with p-value < 0.05
filtered_df1 = feature_scores_df[feature_scores_df['P-Value'] < 0.05]
filtered_df2 = feature_scores_df[(feature_scores_df['P-Value'] > 0.05) & (feature_scores_df['P-Value'] < 1) ]
print(filtered_df1)
print(filtered_df2)


# # Feature selection using Lasso regression on unscaled data

# In[725]:


#Feature selection using lasso

X3=dmrg_encoded.drop("TotalCharge",axis=1).values
y3=dmrg_encoded["TotalCharge"].values
names=dmrg_encoded.drop("TotalCharge",axis=1).columns
# Increase the figure size
plt.figure(figsize=(12, 8))
lasso2=Lasso(alpha=0.1)
lasso_coef=lasso.fit(X3,y3).coef_
plt.bar(names,lasso_coef)
plt.xticks(rotation=45, ha='right')
# Add labels and title for clarity
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Lasso Regression Coefficients')
plt.show()


# In[895]:


dmrg_encoded.info()


# # Features for analysis

# In[896]:


#features selected after SelectKBest
#features_to_select=['TotalCharge','Initial_days','ReAdmis_Yes','Initial_admin_EmergencyAdmission','Complication_risk','Services_ObservationAdmission','BackPain_Yes','Arthritis_Yes','Anxiety_Yes','Reflux_esophagitis_Yes','HighBlood_Yes']

#Features selected after Lasso with unscaled data
#features_to_select=['TotalCharge','Initial_days','Complication_risk','Initial_admin_EmergencyAdmission','BackPain_Yes','Hyperlipidemia_Yes','Arthritis_Yes','Diabetes_Yes','Allergic_rhinitis_Yes','Anxiety_Yes','Reflux_esophagitis_Yes','HighBlood_Yes']

#Features selected after Lasso after scaled and split data
#features_to_select=['TotalCharge','Initial_days','Complication_risk','Initial_admin_EmergencyAdmission','Hyperlipidemia_Yes','Anxiety_Yes']

#features based on practical significance
#features_to_select=['TotalCharge','Initial_days','Complication_risk','Initial_admin_EmergencyAdmission','HighBlood_Yes','Diabetes_Yes']

#Select the variables for analysis
features_to_select=['TotalCharge','Initial_days','Complication_risk','ReAdmis_Yes','Initial_admin_EmergencyAdmission','BackPain_Yes','Hyperlipidemia_Yes','Arthritis_Yes','Diabetes_Yes','Allergic_rhinitis_Yes','Anxiety_Yes','Reflux_esophagitis_Yes','HighBlood_Yes']



rfdata = dmrg_encoded[features_to_select]
print(rfdata.head())


# In[897]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
# Calculate VIF (variance Inflation Factor) for each selected feature
vif_stddata = pd.DataFrame()
vif_stddata["Feature"] = rfdata.columns
vif_stddata["VIF"] = [variance_inflation_factor(rfdata.values, i) for i in range(rfdata.shape[1])]

print(vif_stddata)


# In[729]:


rfdata.info()


# In[898]:


#Outliers detection
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=rfdata['Initial_days'], color="olivedrab")
plt.title('Boxplot of Initial_days')


# In[899]:


#Outliers detection
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=rfdata['Initial_days'], color="olivedrab")
plt.title('Boxplot of Initial_days')


# In[900]:


#Data ditsribution of target variable

# Assuming y is your target variable
plt.figure(figsize=(8, 6))
plt.hist(y, bins=30, color='olivedrab', edgecolor='black', alpha=0.7)
plt.title('Histogram of Target Variable')
plt.xlabel('TotalCharge')
plt.ylabel('Frequency')
plt.show()


# In[901]:


rfdata.describe()


# In[902]:


#Export  preprocessed data
rfdata.to_csv("C:/Users/laksh/Desktop/WGU-MS Data analytics/MSDA-D209-Datamining1/Performance Assessment D209/task2//Task2-RF-preprocesseddata.csv",index=False)


# # Split the dataset into test and train

# In[903]:


#To split the dataset into training and test set
from sklearn.model_selection import train_test_split

#response and predictors
X=rfdata.drop(['TotalCharge'],axis=1)
y=rfdata['TotalCharge']


print("Actual predictors",X.shape)
print("Actual outcome",y.shape)


#splitting training and test data
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=30)

# 0.3 means that 30% of the data will be used for testing
#  random_state=21 ensures that each time we run the code, the split will be the same

frames_train=[X_train,y_train]
rf_train=pd.concat(frames_train,axis=1)

frames_test=[X_test,y_test]
rf_test=pd.concat(frames_test,axis=1)

print("Training set",rf_train.shape)
print("Test set",rf_test.shape)




# In[904]:


#Export the training and test data to CSV

rf_train.to_csv("C:/Users/laksh/Desktop/WGU-MS Data analytics/MSDA-D209-Datamining1/Performance Assessment D209/task2/traindata.csv",index=False)
rf_test.to_csv("C:/Users/laksh/Desktop/WGU-MS Data analytics/MSDA-D209-Datamining1/Performance Assessment D209/task2/testdata.csv",index=False)



# # Initial model

# In[906]:


from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import  r2_score,mean_absolute_error

#Initializing and training model
rf=RandomForestRegressor(n_estimators=200, random_state=30,oob_score=True)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)

# n_estimators=200 is the number of trees in the forest
#random_state=30 ensures reproducibility by controlling the randomness of the model
# oob_score=True, Out-of-Bag score, an internal validation score used to estimate the performance.


#Mean squared error, calculates the average of the squared differences between the actual and predicted values.
print("Mean squared error is: ", MSE(y_test, y_pred))


#Evaluate the test set RMSE
rmse_test=MSE(y_test,y_pred)**(1/2)
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

# R Squared, determines the amount of variance in the model caused by the input variables
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

#Mean absolute error, calculates the average absolute difference between predicted and actual values 
print("Mean absolute error is:",mean_absolute_error(y_test, y_pred))


# OOB Score,out-of-bag score is calculated using the samples that are not used in the training of the model,
oob_score = rf.oob_score_
print(f'Out-of-Bag Score: {oob_score}')


# In[907]:


paramgrid = rf.get_params()
print(paramgrid)


# # cross validation

# In[908]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

# Define the parameter grid

param_grid = {
    'n_estimators': [100,500],
    'max_depth': [None, 10, 20],
    'max_features': ['auto', 'sqrt']
}
#parametsers are set to balance thorough exploration of hyperparameters with computational efficiency

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
#cv=5 is 5 fold cross validation
# n_jobs=-1 uses all available processors for parallel run
#verbose=2 providing more information during execution
# 'scoring' is used to evaluate model performance

# Fit the model
grid_search.fit(X_train, y_train)  

# Get the best parameters from the optimized model
best_params = grid_search.best_params_
print("Best Parameters:", best_params)




# In[910]:


# Print the best highest accuracy

print("Best performance: ",grid_search.best_score_)


# # Evaluate the model

# In[911]:


# Get the best model
best_rf = grid_search.best_estimator_
#most effective model based on the grid search's evaluation of different hyperparameter settings.


# Predict on the test set
y_pred = best_rf.predict(X_test)

# Evaluate the performance on the test set
mae_best=mean_absolute_error(y_test, y_pred)
print("Test set MAE:",mae_best)
rmse_best=np.sqrt(MSE(y_test, y_pred))
print("Test set RMSE:", rmse_best)
r2_best=r2_score(y_test, y_pred)
print("Test set R-squared:",r2_best)
mse_best= MSE(y_test, y_pred)
print("Test set Mean squared error is: ",mse_best)


# In[912]:


# Display the predicted values
print("Predicted Values:", y_pred)


# In[915]:


# Create a DataFrame to compare actual vs predicted values
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Display the comparison
print(comparison_df)


# In[916]:


# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals (RandomForest Regression)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

#A normally distributed frequency plot of residuals is one sign of a well-chosen, well-fitted model.


# In[917]:


# Plot Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.05)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('RandomForest model: Actual vs Predicted')
plt.show()



# In[920]:


# Model evaluation graph


metrics = {'Metric': ['MSE', 'RMSE', 'R²','MAE'],
           'Value': [mse_best, rmse_best, r2_best,mae_best]}

# Convert to DataFrame
metrics_df = pd.DataFrame(metrics)

# Plotting
plt.figure(figsize=(8, 5))
plt.bar(metrics_df['Metric'], metrics_df['Value'], color=['steelblue', 'red', 'green','gold'])

# Adding labels and title
plt.ylabel('Value')
plt.title('Model Performance Metrics')

# Show plot
plt.show()


# # Model consistency 

# In[919]:


# Training in different random state to check consistency 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)
# slightly different data splits, to tets the consistancy of model performance


# Create the model with the best parameters
best_model_opt = RandomForestRegressor(max_depth=None, n_estimators=100, random_state=56)

# Fit the model
best_model_opt.fit(X_train, y_train)

y_pred = best_model_opt.predict(X_test)

# Calculate MSE
mse_opt = MSE(y_test, y_pred)

# Calculate RMSE
rmse_opt = np.sqrt(mse_opt)

# Calculate R²
r2_opt = r2_score(y_test, y_pred)

print(f"MSE: {mse_opt}")
print(f"RMSE: {rmse_opt}")
print(f"R²: {r2_opt}")
print("Mean abolute error is:",mean_absolute_error(y_test, y_pred))


# In[ ]:




