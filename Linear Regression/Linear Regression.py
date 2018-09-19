# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 19:26:33 2018

@author: vinso
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#note : this data actually cant use for predict because its not an income or salary or either else but its look like continues ,so why not :) just for a education
#u can change the data,then change the x and y
#i just take it from internet 


#preprocessing

data = pd.read_excel("gov_finance.xlsx") #read the excel file
x = data.iloc[:,0:1].values #im take the year column
y = data.iloc[:,-1].values #the value column

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)
#pslit into xtrain,... then the best size is 0.25 u can experiment 
#note !>49 && !<16

reg = LinearRegression()
reg.fit(x_train,y_train)

#our model done 
#predict = reg.predict(x_test)


plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,reg.predict(x_train),color = 'blue')
plt.xlabel("Years")
plt.ylabel("Income")
plt.show()
#and the data show the diagram plot is going down 
#you can test another result 
plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_test,reg.predict(x_test),color = 'blue')
plt.xlabel("Years")
plt.ylabel("Income")
plt.show()





