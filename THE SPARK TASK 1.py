#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION
# 

# ### TASK 1- PREDICTION USING SUPERVISED MACHINE LEARNING
# 
# 

# ### DESCRIPTION - PREDICT THE PERCENTAGE OF A STUDENT BASED ON THE NUMBER OF STUDY HOURS
# 

# ### NAME - SATYAM KUMAR
# 

# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


dataset_url = 'http://bit.ly/w-data'
s_data = pd.read_csv(dataset_url)


# In[20]:


s_data.head()


# #check the shape of student data
# 

# In[21]:


s_data.shape


# #Checking data types

# In[22]:


s_data.dtypes


# #getting information from student data

# In[23]:


s_data.info()


# #calculating statistical data

# In[24]:


s_data.describe().transpose()


# #checking if there are any null values in s_data

# In[25]:


s_data.isnull().sum()


# #plot the data points on 2-D graph

# In[26]:


s_data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[27]:


X = s_data.iloc[:, :-1].values
y = s_data.iloc[:, 1].values


# In[28]:


#spliting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[29]:


from sklearn.linear_model import LinearRegression
#creating model
regressor = LinearRegression() 
#fit the model
regressor.fit(X_train, y_train)


# In[30]:


y_pred = regressor.predict(X_test)
y_pred


# In[32]:


print(regressor.coef_)


# In[33]:


print(regressor.intercept_)


# In[34]:


y_new = regressor.intercept_ + X*regressor.coef_


# In[35]:


plt.scatter(X,y,color='r')
plt.plot(X, y_new, color='b')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Hours vs Scores')
plt.grid()
plt.show()


# In[36]:


data = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
data.head()


# In[37]:


regressor.score(X_train, y_train)


# In[38]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[39]:


from sklearn import metrics
print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))


# ##### finally calculating the task (hours=9.25)

# In[40]:


hours = [[9.25]]
pred_value = regressor.predict(hours)
print('Number of hours = {}'.format(hours))
print('Predicted Score = {}'.format(pred_value[0]))


# He will pass the exam.

# In[ ]:




