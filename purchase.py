# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:44:38 2024

@author: CSE PL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sns
dataset = pd.read_csv(r"C:\Users\CSE PL\Downloads\Purchase_status.csv")
print(dataset)
dataset.info()
dec=dataset.describe()
print(dec)
X=dataset.iloc[:,[2,3]].values
print(X)


y=dataset.iloc[:,4].values
print(y)
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
X_train = sc.fit_transform(X_train)
X_test =sc.transform(X_test)
print(X_train)
print(X_test)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
y_pred =classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
import seaborn as sns
sns.heatmap(cm,annot=True)






