#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import numpy as np
import pandas as pd


# In[2]:


df=sns.load_dataset("tips")
df


# In[3]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()



# In[4]:


df['smoker']=lb.fit_transform(df['smoker'])
df['sex']=lb.fit_transform(df['sex'])
df['time']=lb.fit_transform(df['time'])
df['day']=lb.fit_transform(df['day'])


# In[5]:


df.dtypes


# In[6]:


df.head(2)


# In[7]:


df.corr()


# In[8]:


import matplotlib.pyplot as plt


# In[16]:


sns.heatmap(np.abs(df.corr()),cmap='Blues')
plt.show()


# In[9]:


sns.heatmap(np.abs(df.corr()),cmap='Blues',annot=True)
plt.show()


# In[19]:


sns.heatmap(np.abs(df.corr())>0.7,cmap='Blues')
plt.show()


# In[10]:


sns.heatmap(np.abs(df.corr())>0.5,cmap='Blues')
plt.show()


# In[11]:


df


# In[22]:


df.describe()

Scaling techniques
1.) standradization  -->  mean 0 and std of 1

2.) Normalization    ---> range of 0 to 1 inclusivedf.describe()
# In[12]:


df.describe()


# In[13]:


from sklearn.preprocessing import StandardScaler
std_scaler=StandardScaler()


# In[14]:


scale_array=std_scaler.fit_transform(df)


# In[15]:


scale_array.shape


# In[16]:


import numpy as np


# In[18]:


from sklearn.preprocessing import MinMaxScaler
mx_scaler=MinMaxScaler()


# In[19]:


mx_array=mx_scaler.fit_transform(df)


# In[20]:


type(mx_array)


# In[21]:


mx_array


# In[24]:


pd.DataFrame(mx_array,columns=df.columns)


# In[25]:


mx_df=pd.DataFrame(mx_array,columns=df.columns)
mx_df


# In[26]:


df.describe()


# In[27]:


mx_df.describe()


# In[29]:


import statsmodels.api as sm
import matplotlib.pyplot as plt

# Create a Q-Q plot
sm.qqplot(df['tip'], line='s')
plt.show()


# In[ ]:




