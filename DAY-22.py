#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pandas import read_csv
import seaborn as sns
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_regression,f_classif
dataframe=sns.load_dataset('tips')
dataframe


# In[3]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
dataframe['sex']=lb.fit_transform(dataframe['sex'])
dataframe['smoker']=lb.fit_transform(dataframe['smoker'])
dataframe['day']=lb.fit_transform(dataframe['day'])
dataframe['time']=lb.fit_transform(dataframe['time'])


# In[4]:


X = dataframe.drop('tip',axis=1)
Y = dataframe.tip


# In[5]:


X


# In[6]:


Y


# In[7]:


test=SelectKBest(score_func=f_regression, k=4).fit(X,Y)
test


# In[ ]:




