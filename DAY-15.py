#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[7]:


df=sns.load_dataset('titanic')
df


# In[3]:


df.duplicated()


# In[4]:


df.duplicated().sum()


# In[5]:


df[df.duplicated()]


# In[15]:


df.drop_duplicates(inplace=True)


# In[9]:


df.shape


# In[8]:


df.age.isnull().sum()


# In[10]:


df['deck'].isnull().sum()


# In[11]:


sns.heatmap(df.isnull())
plt.show()


# In[12]:


df.dropna()


# In[14]:


(df.isnull().sum()/df.shape[0])*100


# In[16]:


df.drop('deck',axis=1,inplace=True)


# In[17]:


df.columns


# In[18]:


df.isnull().sum()


# In[22]:


df['age']   ##numeric == median 
df['embarked']  ##object==mode
df['embark_town']   ##object==mode


# In[23]:


df['embark_town'].unique()


# In[27]:


df['embark_town'].mode()


# In[28]:


df['embark_town'].mode()[0]


# In[29]:


df['embark_town'].isnull().sum()


# In[31]:


#filling missing position

df['embark_town'].fillna('Southampton')


# In[33]:


df['embark_town'].fillna('Southampton',inplace=True)


# In[34]:


df['embark_town'].isnull().sum()


# In[35]:


df['age'].isnull().sum()


# In[36]:


df['age'].median()


# In[37]:


df['age'].fillna(df['age'].median(),inplace=True)


# In[38]:


df['age'].isnull().sum()


# In[43]:


df['embarked'].mode()[0]


# In[44]:


df['embarked'].isnull().sum()


# In[46]:


print('Missing values before imputation - ',df['embarked'].isnull().sum())
df['embarked'].fillna(df['embarked'].mode()[0],inplace=True)
print('Missing values after imputation - ',df['embarked'].isnull().sum())


# In[ ]:




