#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[19]:


ds= pd.read_csv ('/Users/niveditaarvind/Downloads/Life Expectancy Data.csv')


# In[20]:


ds.columns


# In[21]:


ds.shape


# In[22]:


ds1=ds.drop ('Country', axis=1)


# In[23]:


ds1.columns


# In[24]:


ds2=ds1.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
ds2
# drop missing values 


# In[25]:


### Independant Variable: Life Expectancy 
ds_X= ds2.drop('Status', axis=1)
ds_y= ds2["Status"]
print(ds_X.columns)
print(ds_y)


# In[26]:


ds.corr()


# In[27]:


import seaborn as sns
fig = plt.subplots (figsize = (15,15))
sns.heatmap (ds.corr (), square = True, cbar = True, annot = True, annot_kws = {'size': 10})
plt.show ()


# In[28]:


ds_X.value_counts ()
ds_y.value_counts ()


# In[29]:


## setting the x and y values for our analyses 
X = ds_X
y = ds_y


# In[30]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 


# In[31]:


### scaling the dataset to units of variance and on a standard normal distribution so it can be used for the Linear Regression later on
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)  


# In[32]:


from sklearn.decomposition import PCA

pca = PCA()  
X_train = pca.fit_transform(X_train)  
X_test = pca.transform(X_test)  


# In[33]:


#float type array which contains variance ratios for each principal component.
explained_variance = pca.explained_variance_ratio_ 
explained_variance


# In[34]:


## The changes in the curve regarding variance-- as seen with the variance, 7 variable account for almost 75% of the variance 
pca = {'PC1','PC2','PC3','PC4', 'PC5', 'PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13', 'PC14', 'PC15','PC16','PC17','PC18','PC19','PC20'}
plt.plot(explained_variance)
plt.title("variance")


# In[35]:


### running a PCA with a 0.75 threshold  or retaining atleast 75% of the information 
from sklearn.decomposition import PCA

pca = PCA(0.75)  
X_train = pca.fit_transform(X_train)  
X_test = pca.transform(X_test)  


# In[37]:


from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit (X_train, y_train)


# In[38]:


print(logisticRegr.intercept_)


# In[39]:


print(logisticRegr.coef_)


# In[40]:


## goodness of fit-- the model can explain almost 70% of the variance in this model (r2)
logisticRegr.score(X_train, y_train)


# In[60]:


y_pred = (logisticRegr.predict(X_test))


# In[76]:


from sklearn.metrics import precision_score
precision_score(y_test, y_pred, average='macro')

##Precision Score 


# In[77]:


from sklearn.metrics import recall_score
recall_score(y_test, y_pred, average='macro')
## Recall Score 


# In[78]:





# In[69]:





# In[74]:





# In[61]:





# In[53]:





# In[ ]:




