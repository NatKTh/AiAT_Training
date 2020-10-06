#%%
import numpy as np # linear algebra
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score
#%%
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# %%
train_data.info()
train_data.isnull().sum()
# %%
test_data.info()
test_data.isnull().sum()
# %%
train_data.nunique()
# %%
len(train_data[train_data['Fare'] < 5])
# %%
