#python3 linearmultivar.py

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error

x = np.array([[3,2],[2,1],[5,5],[1,1],[2,3],[4,3],[4,4],[3,1],[3,3],[6,1]])
y = np.array([3100,2100,5500,1100,2300,4300,4400,3100,3300,6100])
#1 room = 1000
#1 bathroom 100

def ridge(lamb,theta):
	return lamb * theta**2

def lasso(lamb,theta):
	return lamb * theta
def elastic(lamb,theta):
	las = lasso(lamb,theta)
	ridge = ridge(lamb,theta)
	return las+ridge


def mse(y,pred):
	loss = 0
	for i in range(len(y)):
		loss += (y[i]-pred[i])**2

	return loss/len(y)

def GradientDescent(x,y,n,epochs,rate):
	theta = theta2 =  0
	c = 0
	lamb = 1500
	x1 = [x[i][0] for i in range(n)]
	x2 = [x[i][1] for i in range(n)]
	x1 = x1-np.mean(x1)/np.std(x1)
	x2 = x2-np.mean(x2)/np.std(x2)
	for i in range(epochs):
		predict = (theta * x1) + (theta2 * x2) + c
		loss = mse(y,predict)
		
		ridges = ridge(lamb,theta2)
		dt = -(2/n) * sum(x1*(y-predict))
		dt2 = -(2/n) * sum(x2*(y - predict))+ ridges
		dc = -(2/n) * sum((y-predict))

		theta = theta - (rate  * dt)
		theta2 = theta2 - (rate * dt2)
		c = c - (rate * dc)

		print("Loss : {} , Theta1 : {} , Theta2 : {} , C : {}".format(loss,theta,theta2,c))


	return [theta,theta2,c]



def linear(x,y,epochs,rate):
	n = len(x)
	gradient = GradientDescent(x,y,n,epochs,rate)

	return gradient	





model = linear(x,y,1000,0.0001)

def predict(m,x):
	n = len(x)
	x1 = [x[i][0] for i in range(n)]
	x2 = [x[i][1] for i in range(n)]
	prediction = [(m[0] * x1[i]) + (m[1] * x2[i]) + m[2] for i in range(n)]
	return prediction

prediction = predict(model,x)
loss  = mse(y,prediction)
print(model)
print("actual y : ",y)
print("prediction : ",prediction)
print("loss : ",loss)


x1 = [x[i][0] for i in range(len(x))]
plt.scatter([x[i][0] for i in range(len(x))],y,color = "red") 
plt.plot([x[i][0] for i in range(len(x))],prediction)
plt.title("Linear Regression")
plt.xlabel("Rooms")
plt.ylabel("Price")
plt.show()


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x,y)
pred2 = reg.predict(x)
"""
from sklearn.linear_model import Ridge
ridgeReg = Ridge(alpha=0.05, normalize=True)
ridgeReg.fit(x,y)
pred2 = ridgeReg.predict(x)
print(ridgeReg)
print(pred2)


from sklearn.linear_model import Lasso

lassoReg = Lasso(alpha=0.3, normalize=True)

lassoReg.fit(x_train,y_train)

pred = lassoReg.predict(x_cv)


from sklearn.linear_model import ElasticNet

ENreg = ElasticNet(alpha=1, l1_ratio=0.5, normalize=False)

ENreg.fit(x_train,y_train)

pred_cv = ENreg.predict(x_cv)

"""
from sklearn.linear_model import ElasticNet

ENreg = ElasticNet(alpha=1, l1_ratio=0.5, normalize=False)

ENreg.fit(x,y)

pred_cv = ENreg.predict(x)
print("yp ",pred_cv)
plt.scatter(x1,y,color = "red") 
plt.plot(x1,pred_cv.predict(x))
plt.title("Linear Regression22")
plt.xlabel("Rooms")
plt.ylabel("Price")
plt.show()

