#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


from scipy.stats import zscore
data=sns.load_dataset('iris')
df=data.copy()


# In[3]:


z_scores=np.abs(zscore(df.drop('species',axis=1)))


# In[4]:


df


# In[5]:


df.describe()


# In[6]:


zscore(df.drop('species',axis=1))  ##zscore is used to identify the outlier


# In[7]:


np.abs(zscore(df.drop('species',axis=1)))  ## by applying np.abs all the number will become the positive number


# In[8]:


z_scores


# In[9]:


z_scores.loc[10:15]


# In[14]:


z_scores<3


# In[15]:


non_outliers=(z_scores < 3).all(axis=1)
df_no_outliers=df[non_outliers]


# In[16]:


df_no_outliers


# In[18]:


(z_scores<3).all()


# In[19]:


(z_scores<3).all(axis=1)


# In[20]:


outliers=(z_scores > 3).any(axis=1)
outlier_rows=df[non_outliers]
outlier_rows                 ##outlier data


# In[22]:


x[mean+3 std,mean-3 std]


# In[23]:


from sklearn.ensemble import IsolationForest
import seaborn as sns
import pandas as pd
import numpy as np


# In[24]:


data=sns.load_dataset("iris")
data


# In[25]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
data['species']=lb.fit_transform(data['species'])


# In[28]:


data.head(3)


# In[27]:


clf=IsolationForest(random_state=10)
clf.fit(data)


# In[29]:


clf.predict(data)


# In[30]:


clf=IsolationForest(random_state=10,contamination=0.01)
clf.fit(data)


# In[ ]:




