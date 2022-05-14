#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
data= pd.read_csv ('/Users/niveditaarvind/Downloads/Life Expectancy Data.csv')


# In[2]:


data.isnull().sum()


# In[3]:


x = data['Life expectancy '].mean()
data['Life expectancy '].fillna(x, inplace=True)

x = data['Adult Mortality'].mean()
data['Adult Mortality'].fillna(x, inplace=True)

x = data['Alcohol'].mean()
data['Alcohol'].fillna(x, inplace=True)

x = data['Hepatitis B'].mean()
data['Hepatitis B'].fillna(x, inplace=True)

x = data[' BMI '].mean()
data[' BMI '].fillna(x, inplace=True)

x = data['Polio'].mean()
data['Polio'].fillna(x, inplace=True)

x = data['Total expenditure'].mean()
data['Total expenditure'].fillna(x, inplace=True)

x = data['Diphtheria '].mean()
data['Diphtheria '].fillna(x, inplace=True)

x = data['GDP'].mean()
data['GDP'].fillna(x, inplace=True)

x = data['Population'].mean()
data['Population'].fillna(x, inplace=True)

x = data[' thinness  1-19 years'].mean()
data[' thinness  1-19 years'].fillna(x, inplace=True)

x = data[' thinness 5-9 years'].mean()
data[' thinness 5-9 years'].fillna(x, inplace=True)

x = data['Income composition of resources'].mean()
data['Income composition of resources'].fillna(x, inplace=True)

x = data['Schooling'].mean()
data['Schooling'].fillna(x, inplace=True)


# In[4]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['Country'] = le.fit_transform(data['Country'])

le = LabelEncoder()
data['Status'] = le.fit_transform(data['Status'])


# In[5]:


data.info()


# In[6]:


### Independant Variable: Life Expectancy 
df_X = data.drop('Life expectancy ', axis=1)
df_y = data['Life expectancy ']


# In[7]:


print(df_X.columns)
print(df_y)


# In[8]:


data.corr()


# In[9]:


import seaborn as sns
fig = plt.subplots (figsize = (15,15))
sns.heatmap (data.corr (), square = True, cbar = True, annot = True, annot_kws = {'size': 10})
plt.show ()


# In[10]:


df_X.value_counts ()
df_y.value_counts ()


# In[11]:


## setting the x and y values for our analyses 
X = df_X
y = df_y


# In[14]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 


# In[15]:


### scaling the dataset to units of variance and on a standard normal distribution so it can be used for the Linear Regression later on
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)  


# In[16]:


from sklearn.decomposition import PCA

pca = PCA()  
X_train = pca.fit_transform(X_train)  
X_test = pca.transform(X_test)  


# In[17]:


#float type array which contains variance ratios for each principal component.
explained_variance = pca.explained_variance_ratio_ 
explained_variance


# In[18]:


## The changes in the curve regarding variance-- as seen with the variance, 7 variable account for almost 75% of the variance 
pca = {'PC1','PC2','PC3','PC4', 'PC5', 'PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13', 'PC14', 'PC15','PC16','PC17','PC18','PC19','PC20','PC21','PC22'}
plt.plot(explained_variance)
plt.title("variance")


# In[19]:


### running a PCA with a 0.75 threshold  or retaining atleast 75% of the information 
from sklearn.decomposition import PCA

pca = PCA(0.95)  
X_train = pca.fit_transform(X_train)  
X_test = pca.transform(X_test)  


# In[20]:


from sklearn.tree import DecisionTreeRegressor


# In[21]:


DecisionTreeRegressor = DecisionTreeRegressor()
DecisionTreeRegressor.fit (X_train, y_train)


# In[22]:


y_pred = (DecisionTreeRegressor.predict(X_test))


# In[23]:


Data_Final= pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})


# In[24]:


Data_Final.head()


# In[ ]:




