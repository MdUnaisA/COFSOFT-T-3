#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


iris = pd.read_csv("IRIS.csv")


# In[3]:


iris.head()


# In[4]:


iris.info()


# In[5]:


iris.describe()


# In[6]:


iris.value_counts()


# In[7]:


iris.count()


# In[8]:


iris['species'].value_counts()


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


sns.set()


# In[11]:


sns.countplot(x = 'species', data = iris,  )


# In[12]:


iris['species'].value_counts().plot(kind='barh', color=['red','green','yellow'], title='species count')


# In[14]:


sns.swarmplot(x= 'species', y='petal_length', data = iris, size = 3)


# In[15]:


sns.scatterplot(x ='sepal_length', y = 'sepal_width', hue ='species', data = iris)


# In[16]:


iris.replace({'species':{'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}}, inplace = True)


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


x = iris.drop(columns='species', axis= 1 )


# In[19]:


y = iris['species']


# In[20]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=4)


# In[23]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(max_iter=1000)


# In[24]:


LR.fit(x_train,y_train)


# In[25]:


LogisticRegression()


# In[26]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report 


# In[27]:


y_pred = LR.predict(x_test)
acc_LR = accuracy_score(y_pred, y_test)
print(acc_LR)


# In[28]:


from sklearn import metrics


# In[29]:


score = round(LR.score(x_test, y_test)*100,2)
score


# In[30]:


cls_report = classification_report(y_pred, y_test)


# In[31]:


print('accuracy score for the logistic regression model is:', score)

print('classification report for our model is:', cls_report)


# In[32]:


from sklearn.svm import SVC


# In[33]:


svc = SVC()
svc.fit(x_train,y_train)


# In[34]:


y_pred_svc = svc.predict(x_test)
acc_svc = accuracy_score(y_pred_svc,y_test)
score_svc = round(svc.score(x_test, y_test)*100,2)
score_svc


# In[35]:


cls_svc = classification_report(y_pred_svc, y_test)


# In[36]:


print('accuracy score of svc model is :', acc_svc)
print('classification report of svc model is:', cls_svc)


# In[37]:


print(x)
print(y)


# In[39]:


import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# In[40]:


LR.predict([[6.7, 3.0, 5.2, 2.3]])
svc.predict([[6.7, 3.0, 5.2, 2.3]])

