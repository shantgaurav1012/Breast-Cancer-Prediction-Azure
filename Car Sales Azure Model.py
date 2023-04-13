#!/usr/bin/env python
# coding: utf-8

# <h2 style='color:purple' align='center'>Training And Testing Available Data</h2>

# <p><b>We have a dataset containing prices of used BMW cars. We are going to analyze this dataset
# and build a prediction function that can predict a price by taking mileage and age of the car
# as input. We will use sklearn train_test_split method to split training and testing dataset</b></p>

# In[1]:


import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import r2_score
df = pd.read_csv("carprices.csv")
df.head()


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# **Car Mileage Vs Sell Price ($)**

# In[3]:


plt.scatter(df['Mileage'],df['Sell Price($)'])


# **Car Age Vs Sell Price ($)**

# In[4]:


plt.scatter(df['Age(yrs)'],df['Sell Price($)'])


# **Looking at above two scatter plots, using linear regression model makes sense as we can clearly see a linear relationship between our dependant (i.e. Sell Price) and independant variables (i.e. car age and car mileage)**

# <p style='color:purple'><b>The approach we are going to use here is to split available data in two sets</b></p>
#     <ol>
#         <b>
#         <li>Training: We will train our model on this dataset</li>
#         <li>Testing: We will use this subset to make actual predictions using trained model</li>
#         </b>
#      </ol>
# <p style='color:purple'><b>The reason we don't use same training set for testing is because our model has seen those samples before, using same samples for making predictions might give us wrong impression about accuracy of our model. It is like you ask same questions in exam paper as you tought the students in the class.
# </b></p>

# In[5]:


X = df[['Mileage','Age(yrs)']]


# In[6]:


y = df['Sell Price($)']


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3) 


# In[8]:


X_train


# In[9]:


X_test


# In[10]:


y_train


# In[11]:


y_test


# **Lets run linear regression model now**

# In[12]:


from sklearn.linear_model import LinearRegression
clf = LinearRegression()
a = clf.fit(X_train, y_train)


# In[13]:


X_test


# In[14]:


y_predict = clf.predict(X_test)
y_predict


# In[15]:


y_test


# In[16]:


clf.score(X_test, y_test)


# In[17]:


scored =r2_score(y_test,y_predict)
print(scored)


# In[18]:


meanAbErr = metrics.mean_absolute_error(y_test, y_predict)
meanSqErr = metrics.mean_squared_error(y_test, y_predict)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_predict))
print('R squared: {:.2f}'.format(scored*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)


# In[19]:


pred_df=pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_predict,'Difference':y_test-y_predict})
pred_df


# In[20]:


sns.regplot(x="Actual Value", y="Predicted Value",data=pred_df , scatter_kws={"color": "green"}, line_kws={"color": "red"});


# In[ ]:




