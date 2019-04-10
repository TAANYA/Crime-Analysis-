#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 02:16:25 2018

@author: p2
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/home/p2/Desktop/major/MonthData/January_Data.csv',sep ='\t')

dataset.columns.values

dataset.drop(['Unnamed: 0' , 'Block' , 'Description' , 'Location Description'
              , 'Arrest' ,'Domestic' , 'Beat' ,'Month' , 'Latitude' ,'Longitude' ,'Day'] ,axis=1 ,inplace =True)



dataset.columns.values

dataset.head(5)

dataset.shape[0]

year = []
crimeRate = []

for i in range(2001,2019):
    year.append(i)
    crimeRate.append((dataset[dataset.Year == i].shape[0]/dataset.shape[0])*100)

crimeRateYearlyData = pd.DataFrame({'Year':year ,'Crime Rate':crimeRate})


X = crimeRateYearlyData.iloc[:15, 1:2].values
y = crimeRateYearlyData.iloc[:15, 0].values


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(sc_X.transform(np.array([[2020]])))
y_pred = sc_y.inverse_transform(y_pred)
print(y_pred)

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


