import numpy as np
import matplotlib.pyplot as plt
#x =np.array([15,17,14,13,14,21,22,25,18,21])
#y = np.array([3700,4400,5200,5500,5600,6000,6700,6900,7200,8000])

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1200,1800,2100,2400,2500,3200,3600,4000,5000,6000])


def mse(y,pred):
	loss = 0
	for i in range(len(y)):
		loss += (y[i]-pred[i])**2

	return loss/len(y)

def GradientDescent(x,y,iteration,learning_rate):
	theta = 0
	b = 0
	leng = len(x)
	x = x-np.mean(x)/np.std(x)


	for i in range(iteration):
		predict = (theta*x)+b
		np.seterr(all='warn')
		loss = mse(y,predict)
		new_theta = -(2/len(x)) * sum(x*(y-predict))
		new_b = -(2/len(x)) * sum((y-predict))
		
		theta = theta - (learning_rate*new_theta)
		b = b - (learning_rate * new_b)

		print("loss {} , theta {}, b {}".format(loss,theta,b))
	return [theta,b]

def gradientDescent(X, y, theta,num_iters ,alpha):
    """
       Performs gradient descent to learn theta
    """
    m = y.size  # number of training examples
    for i in range(num_iters):
        y_hat = np.dot(X, theta)
        theta = theta - alpha * (1.0/m) * np.dot(X.T, y_hat-y)
        print("theta : ",theta)
    return theta
	
def linearRegression(x,y,learning_rate,iteration):

	#OLS METHOD
	x1 = (sum(x)/len(x))-x
	y1 = (sum(y)/len(y))-y
	a1 = x1*y1
	x2 = x1**2

	m = sum(a1)/sum(x2)
	b = (sum(y)/len(y)) - (m*(sum(x)/len(x)))

	gradient = GradientDescent(x,y,iteration,learning_rate)
	
	return gradient



def predict(m,x):
	return (m[0]*x)+m[1]




model = linearRegression(x,y,0.001,1000)
prediction = predict(model,x)
loss = mse(y,prediction)


print(model)
print("actual y : ",y)
print("prediction : ",prediction)
print("loss : ",loss)

plt.scatter(x, y,color = "red") 
plt.plot(x,prediction)
plt.title("Linear Regression")
plt.xlabel("Age")
plt.ylabel("Years of success")
plt.show()














