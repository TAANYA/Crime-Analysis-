#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 15:30:46 2018
@author: p2
"""

import pandas as pd 

data = pd.read_csv("/home/p2/Desktop/major/Crimes_-_2001_to_present.csv",nrows=500000)

#checking the available data
print(data.shape)

print(data.head(5))

print(data.columns.values)

print("No of columns in data dataset: "+str(len(data.columns.values)))

#removing unnessecary columns from the data set 
data.drop(['ID' ,'Case Number' ,'IUCR' ,'FBI Code' ,'X Coordinate', 
           'Y Coordinate' ,'Updated On' ,'Location', 'Beat' , 'Community Area' 
           ,'District' ,'Ward','Block' ] ,axis=1 ,inplace =True)

print(data.columns.values)

print(data.head(3))

print("No of columns in data dataset: "+str(len(data.columns.values)))

print(data.isnull().sum())

"""
print(data.Ward.unique())
print(len(data.Ward.unique()))
"""

#drop null values 
data = data.dropna(axis=0 ,subset=['Latitude' , 'Longitude'])

#checking the reamining null values 
data = data.dropna(axis=0 , subset= ['Location Description'])

print(data.shape)

print(data.isnull().sum())

#shuffling data and reseting the index
from sklearn.utils import shuffle
data = shuffle(data)
data=data.reset_index(drop=True)

"""
data2017 =data[data['Year']>2016]
print(data2017.shape) 
"""
data=data[data['Year']<2017]
print(data.shape) 


#reseting the index 
data=data.reset_index(drop=True)
#data2017 = data2017.reset_index(drop=True)

# splitting date into three columns  of month , day & year
#data['Date'][0].split(" ")[0].split("/")

month = []
day = []
year = []
for i in range(0 , len(data)):
    month.append(int(data['Date'][i].split(" ")[0].split("/")[0]))
    day.append(int(data['Date'][i].split(" ")[0].split("/")[1]))
    year.append(int(data['Date'][i].split(" ")[0].split("/")[2]))
    
#dateDataFrame = pd.DataFrame({'Month':month ,'Day':day , 'CrimeYear' : year})

#joining the splitted value of date back to the data frame
data = pd.concat([data,pd.DataFrame({'Month':month ,'Day':day , 'CrimeYear' : year})]  , axis=1)
print(data.shape)

#removing the unnecesssary date entries
data.drop(['Date','CrimeYear'] ,axis = 1 ,inplace =True)

print(data.shape)


#predicting the primary type from the dataset 
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

#handling categorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
lableencoder_X = LabelEncoder()
X[: , 0]= lableencoder_X.fit_transform(X[: ,0])

lableencoder_X = LabelEncoder()
X[: , 1]= lableencoder_X.fit_transform(X[: ,1])

lableencoder_X = LabelEncoder()
X[: , 2]= lableencoder_X.fit_transform(X[: ,2])

lableencoder_X = LabelEncoder()
X[: , 3]= lableencoder_X.fit_transform(X[: ,3])

onehotencoder = OneHotEncoder(categorical_features=[0,1 ,4,7,8])
X= onehotencoder.fit_transform(X).toarray()

lableencoder_y = LabelEncoder()
y=lableencoder_y.fit_transform(y) 


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



#feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)


#fitting traning set to logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train , y_train)

#predicting the test set
y_pred = classifier.predict(X_test)

print(y_pred.size)

#evaluating the result via confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

from sklearn.externals import joblib
joblib.dump(classifier, 'crime5L.pkl') 














