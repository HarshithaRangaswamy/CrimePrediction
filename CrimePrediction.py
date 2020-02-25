#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pandas as pd
import numpy as np
import os
from sklearn import tree
from IPython.display import Image  
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pydotplus
from  sklearn.linear_model import LinearRegression
from sklearn.model_selection import validation_curve
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from __future__ import division
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[3]:


#importing dataset downloaded from UCI machine learning repository
get_ipython().run_line_magic('matplotlib', 'inline')
dfd=pd.read_csv("C:/Users/harsh/Documents/CrimePrediction/communities-crime-full.csv")
df=pd.read_csv("C:/Users/harsh/Documents/CrimePrediction/communities-crime-clean.csv")


# In[7]:


plt.hist(dfd['communityname'])
plt.show()


# In[5]:


#creating new column in cleaned dataframe where binary values are used to represent the values of ViolentCrimesPerPop
#for df['ViolentCrimesPerPop']>0.1 df[highcrime] 1 is assigned else 0
df['highCrime'] = np.where(df['ViolentCrimesPerPop']>0.1, 1, 0)
df.head()


# In[4]:


#finding highly correlated attributes
plt.matshow(df.corr())


# In[5]:


corr = df.corr()
#print(corr)
c1=df.corr()['ViolentCrimesPerPop'].sort_values().drop_duplicates().abs()
print(c1.sort_values(ascending = False))


# In[6]:


#finding top 10 highly correlated atributes with increase in crime rate
df.corr()['ViolentCrimesPerPop'].abs().nlargest(5)


# In[7]:


#plotting histogram for 'PctIlleg'
plt.hist(df['PctIlleg'],bins=30)
plt.xlabel('Percentage of kids born to never married')
plt.grid(True)
#df['PctIlleg'].hist(log=True)
#plt.show()


# In[8]:


plt.hist(df['PctPopUnderPov'],bins=30)
plt.xlabel('Percentage of people under poverty level')
plt.grid(True)


# In[9]:


df['NumStreet'].hist(log=True)
plt.xlabel('Number of homeless people in street')
plt.show()


# ##### 1. K-means

# In[10]:


# Drop the columns and build the model using Kmeans
X = df.drop('ViolentCrimesPerPop',1).drop('highCrime',1).drop('communityname',1)
Y = df['highCrime']
#Computing the K-means for the given data
scaler = StandardScaler()
X_scaled = scaler.fit_transform( X )


# In[11]:


#Setting the cluster number form 1 to 10
from sklearn.metrics import silhouette_samples, silhouette_score

cluster_number = range( 1, 10 )
cluster_errors = []

for n_clusters in cluster_number:
    clusters = KMeans( n_clusters )
    clusters.fit( X_scaled )
    #silhouette_avg = silhouette_score(X_scaled , a)
    cluster_errors.append( clusters.inertia_ )


# In[12]:


clusters_df = pd.DataFrame( { "num_clusters":cluster_number, "cluster_errors": cluster_errors } )
clusters_df.head()


# In[13]:


#Finding the optimal number of clusters with the Elbow method
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "8" )
plt.grid(True)


# In[14]:


#calculating accuracy for different k values
np.random.seed(1)
k2 = KMeans(n_clusters = 2)
accu_k2 = cross_val_score(k2, X, Y, cv=10, scoring='accuracy').mean()

print ('Accuracy is', accu_k2)


# In[15]:


np.random.seed(1)
k3 = KMeans(n_clusters = 3)
accu_k3 = cross_val_score(k3, X, Y, cv=10, scoring='accuracy').mean()

print ('Accuracy is', accu_k3)


# In[17]:



k4 = KMeans(n_clusters = 4)
accu_k4 = cross_val_score(k4, X, Y, cv=10, scoring='accuracy').mean()

print ('Accuracy is', accu_k4)


# In[14]:


np.random.seed(1)
#train and test data split
from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(X,Y,test_size=0.2)
#number of rows in training dataset
print(train_X.shape,train_Y.shape)
#number of rows in test dataset
print(test_X.shape,test_Y.shape)


# ##### 2.Decision Trees

# In[15]:


#ploting histogram for population attribute
df.population.hist()


# In[16]:


#Store dataframe to a new variable before dropping the variables and applying the Decision tree 
#classifier
tree_df=df
tree_df = tree_df.drop('communityname', 1).drop('ViolentCrimesPerPop', 1).drop('fold', 1).drop('state', 1).drop('highCrime',1)
Y = df['highCrime']
dtree = tree.DecisionTreeClassifier(max_depth= 3)
dtree = dtree.fit(tree_df, Y)


# In[22]:


#finding the accuracy precission and recall of the model
acc_dt = cross_val_score(dtree, tree_df, Y, scoring='accuracy').mean()
pre_dt = cross_val_score(dtree, tree_df, Y, scoring='precision').mean()
rec_dt = cross_val_score(dtree, tree_df, Y, scoring='recall').mean()
f1_dt = cross_val_score(dtree, tree_df, Y, scoring='f1').mean()


# In[23]:


print("The Accuracy of the Decision tree classifier is:",acc_dt)


# In[24]:


print("The Precision of the Decision tree classifier is:",pre_dt)


# In[25]:


print("The recall of the Decision tree classifier is:",rec_dt)


# In[26]:


print("The F1-score of the Decision tree classifier is:",f1_dt)


# ### 3 Linear Classification

# ##### GaussianNB

# In[68]:


#building the model using GaussianNB
predictors = df.drop('ViolentCrimesPerPop',1).drop('highCrime',1).drop('communityname',1)
response = df['highCrime']
gnb_mod = GaussianNB([0,1])


# In[69]:


#Finding the accuracy precision and recall values
np.seterr(divide = 'ignore')
acc_gnb = cross_val_score(gnb_mod, predictors, response, cv=10, scoring='accuracy').mean()
pre_gnb = cross_val_score(gnb_mod, predictors, response, cv=10, scoring='precision').mean()
rec_gnb = cross_val_score(gnb_mod, predictors, response, cv=10, scoring='recall').mean()


# In[70]:


print("The Accuracy of GaussianNB model:",acc_gnb)


# In[71]:


print("The Precision of GaussianNB model:",pre_gnb)


# In[72]:


print("The Recall of GaussianNB model:",rec_gnb)


# ### 4 Non Linear Classification

# ##### Logistic Regression

# In[114]:


#building the model using Logistic Regression
from sklearn.feature_selection import RFE
predictors = df.drop('ViolentCrimesPerPop',1).drop('highCrime',1).drop('communityname',1)
response = df['highCrime']
lr = LogisticRegression()
lr.fit(train_X, train_Y)
Y_predict=lr.predict(test_X)


# In[126]:


print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(lr.score(test_X, test_Y)))


# In[115]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(test_Y, Y_predict)
print(confusion_matrix)


# In[116]:


from sklearn.metrics import classification_report
print(classification_report(test_Y, Y_predict))


# In[122]:


#Finding the accuracy precision and recall value
from sklearn.metrics import mean_squared_error
acc_lr = cross_val_score(lr, predictors, response, cv=10, scoring='accuracy').mean()
pre_lr = cross_val_score(lr, predictors, response, cv=10, scoring='precision').mean()
rec_lr = cross_val_score(lr, predictors, response, cv=10, scoring='recall').mean()
mse_lr = cross_val_score(lr, predictors, response,None,scoring='neg_mean_squared_error',cv=10).mean()
#mean_squared_error(response , y_pred)
print(abs(mse_lr))


# In[123]:


print("The Accuracy of Logistic Regression model:",acc_lr)


# In[124]:


print("The Precision of Logistic Regression model:",pre_lr)


# In[125]:


print("The Recall of Logistic Regression model:",rec_lr)


# In[35]:


#selecting the predective feature as High Crime
df_predection = df.drop('communityname',1)
df_predection.head()


# ##### Finding all the features which are responsible for high crime rate

# In[36]:


#Computing the mean and variance for the highCrime predictive feature and calculating the  meanTrue  meanFalse varTrue  varFalse 
count = 0
dict_pred = {}
for col in df_predection:
    meanTrue = df_predection[df_predection['highCrime'] == 1][col].mean()
    meanFalse = df_predection[df_predection['highCrime'] == 0][col].mean()
    varTrue = df_predection[df_predection['highCrime'] == 1][col].var()
    varFalse = df_predection[df_predection['highCrime'] == 0][col].var()
    if(col != 'highCrime'):
        predScore = abs((meanTrue - meanFalse))/(varFalse+varTrue)
        dict_pred[col] = predScore
print(sorted(dict_pred.items(), key=lambda x: x[1]))
count = count+1


# In[37]:


#printing top 10 features which affect crime rate
top_features = sorted(dict_pred.items(), key=lambda x: x[1])[-11:-1]
for i in top_features:
    print(i)


# ##### 5 Linear Regression

# In[38]:


predictor= df
predictor = predictor.drop('communityname', 1).drop('ViolentCrimesPerPop', 1).drop('fold', 1).drop('state', 1).drop('highCrime',1)
response = df['highCrime']
lrg= LinearRegression().fit(predictor,response)
scores = cross_val_score(lrg, predictor, response,None,scoring='neg_mean_squared_error',cv=10).mean()

print(abs(scores))


# In[39]:


#variable with large coefficient value are more important in predicting response value
print('Coefficients: \n', lrg.coef_)
ls_lrg_coeff = np.array(lrg.coef_)
max_feature = np.where(ls_lrg_coeff == ls_lrg_coeff.max())


# In[40]:


#column name with high coefficient value
predictor.columns[np.argmax(ls_lrg_coeff)]


# In[41]:


#column name with low coefficient value
predictor.columns[np.argmin(ls_lrg_coeff)]


# ##### 5 Ridge Regression

# In[86]:


# Tuning the Aplha for Ridge Regression
parameter_range = np.logspace(-2, 1, 10)
train_scores, test_scores = validation_curve(
    linear_model.Ridge (), predictor, response, param_name="alpha", param_range=parameter_range,
    cv=10, scoring="neg_mean_squared_error", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)


# In[90]:


print("MSE",abs(test_scores_mean.mean()))
print("train-MSE",abs(train_scores_mean.mean()))


# In[55]:


plt.title("Validation Curve for tuning alpha")
plt.xlabel("Alpha")
plt.ylabel("Mean Squared error")

lw = 3
plt.semilogx(parameter_range, train_scores_mean, label="Training score",
             color="green", lw=lw)
plt.fill_between(parameter_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="green", lw=lw)
plt.semilogx(parameter_range, test_scores_mean, label="test score",
             color="navy", lw=lw)
plt.fill_between(parameter_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="lower right")
plt.show()


# In[ ]:




