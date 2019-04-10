#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 15:25:51 2018

@author: p2
"""

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

# Importing the datasetJanuary
datasetJanuary = pd.read_csv('/home/p2/Desktop/major/MonthData/January_Data.csv',sep ='\t')
datasetFebuary = pd.read_csv('/home/p2/Desktop/major/MonthData/Febuary_Data.csv',sep ='\t')
datasetMarch = pd.read_csv('/home/p2/Desktop/major/MonthData/March_Data.csv',sep ='\t')
datasetApril = pd.read_csv('/home/p2/Desktop/major/MonthData/April_Data.csv',sep ='\t')
datasetMay = pd.read_csv('/home/p2/Desktop/major/MonthData/May_Data.csv',sep ='\t')
datasetJune = pd.read_csv('/home/p2/Desktop/major/MonthData/June_Data.csv',sep ='\t')
datasetJuly = pd.read_csv('/home/p2/Desktop/major/MonthData/July_Data.csv',sep ='\t')
datasetAugust= pd.read_csv('/home/p2/Desktop/major/MonthData/August_Data.csv',sep ='\t')
datasetSeptember = pd.read_csv('/home/p2/Desktop/major/MonthData/September_Data.csv',sep ='\t')
datasetOctober = pd.read_csv('/home/p2/Desktop/major/MonthData/October_Data.csv',sep ='\t')
datasetNovember = pd.read_csv('/home/p2/Desktop/major/MonthData/November_Data.csv',sep ='\t')
datasetDecember = pd.read_csv('/home/p2/Desktop/major/MonthData/December_Data.csv',sep ='\t')

MonthlyData = []
MonthlyData.append(datasetJanuary)
MonthlyData.append(datasetFebuary)
MonthlyData.append(datasetMarch)
MonthlyData.append(datasetApril)
MonthlyData.append(datasetMay)
MonthlyData.append(datasetJune)
MonthlyData.append(datasetJuly)
MonthlyData.append(datasetAugust)
MonthlyData.append(datasetSeptember)
MonthlyData.append(datasetOctober)
MonthlyData.append(datasetNovember)
MonthlyData.append(datasetDecember)

year = []
crimeRate = []
district = []

for i in range(0,12):
    MonthlyData[i].drop(['Unnamed: 0' , 'Block' , 'Description' , 'Location Description'
              , 'Arrest' ,'Domestic' , 'Beat' ,'Month' , 'Latitude' ,'Longitude' ,'Day'] ,axis=1 ,inplace =True)





for j in range(2001 ,2019):
    yearSum=0
    for i in range(0,12):    
        yearSum=yearSum+MonthlyData[i][MonthlyData[i].Year==j].shape[0]
    year.append(j)
    crimeRate.append(((MonthlyData[0][MonthlyData[0].Year==j].shape[0])/yearSum)*100)


for j in range(2001 ,2019):
    yearSum=0
    for d in range(1,32):
        for i in range(0,12):    
            yearSum=yearSum+MonthlyData[i][(MonthlyData[i].Year==j )&( MonthlyData[i].District==d)].shape[0]
        year.append(j)
        district.append(d)
        crimeRate.append(((MonthlyData[0][(MonthlyData[0].Year==j) & (MonthlyData[i].District==d)].shape[0])/yearSum)*100)




crimeRateYearlyDataJanuary = pd.DataFrame({'Year':year ,'Crime Rate':crimeRate})


X = crimeRateYearlyDataJanuary.iloc[:15, 1:2].values
y = crimeRateYearlyDataJanuary.iloc[:15, 0].values



#District Wise
crimeRateYearlyDataJanuary = pd.DataFrame({'Year':year ,'District':district,'Crime Rate':crimeRate})

crimeRateYearlyDataJanuary=crimeRateYearlyDataJanuary.sort_values(['Year'], ascending=[1])

X = crimeRateYearlyDataJanuary.iloc[:482, 1:3].values
y = crimeRateYearlyDataJanuary.iloc[:482, 0].values


# Feature Scaling required for SVR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting SVR to the datasetJanuary
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(sc_X.transform(np.array([[1,2016]])))
y_pred = sc_y.inverse_transform(y_pred)
print(y_pred)

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.xlabel('Year')
plt.ylabel('Crime Rate')
plt.show()


# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.xlabel('Year')
plt.ylabel('Crime Rate')
plt.show()



# Fitting Polynomial Regression to the datasetJanuary
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

lin_reg_2.predict(poly_reg.fit_transform([[1,2016],[1,2017],[20,2018],[15,2019],[12,2020],[1,2025]]))

import plotly.plotly as py
import plotly.graph_objs as go
import plotly

Xsplit = np.split(X ,2 , axis =1)
a = np.reshape(Xsplit[0] , (Xsplit[0].shape[0],))
 b = np.reshape(Xsplit[1] , (Xsplit[1].shape[0],))
trace1 = go.Scatter3d(
    x=a,
    y=b,
    z=y,
    mode='markers',
    marker=dict(
        size=12,
        color=y,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig)
py.iplot(fig, filename='3d-scatter-colorscale')





# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(2016)
print(y_pred)
y_pred = regressor.predict(2017)
print(y_pred)
y_pred = regressor.predict(2018)
print(y_pred)
y_pred = regressor.predict(2019)
print(y_pred)
y_pred = regressor.predict(2020)
print(y_pred)

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()





