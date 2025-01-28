#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=sns.load_dataset('iris')
df


# In[3]:


df.isnull().sum()


# In[4]:


df.info


# In[7]:


df['species'].unique()


# In[12]:


oh_species=pd.get_dummies(df['species'])
df=pd.concat([df.drop('species',axis=1),oh_species],axis=1)


# In[13]:


df


# In[ ]:




