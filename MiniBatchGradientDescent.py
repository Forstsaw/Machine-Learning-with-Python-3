import numpy as np 
from sklearn.utils import shuffle

#mini batch gradient descent
def mini(x,y,rate,epochs):
        
	batch_size = len(x)/epochs
	theta1 = theta2 = 0
	
	if batch_size <= 1:
		while True:
			if batch_size < 1:
				batch_size *= 10
			else:
				break		
	if batch_size < epochs:
		m = int(epochs/batch_size)
		
	else:
		m = int(batch_size/epochs)

	for i in range(epochs):
		x_shuffled,y_shuffled = shuffle(x,y)

		for j in range(30):

			x_j = x_shuffled[j:j+30]
			y_j = y_shuffled[j:j+30]

			predict = theta1 + theta2*x_j

			n_t2 = (1/m)*(x_j*(predict-y_j))
			n_t1 = (1/m)*((predict-y_j))

			theta1 = theta1-(rate*n_t1)
			theta2 = theta2-(rate*n_t2)

	return theta2,theta1

X = 2 * np.random.rand(100,1)
y = 4 +3 * X+np.random.randn(100,1)
a = mini(X,y,0.01,1000)



print(a[1])
#1.7164927


#1.7164927+1.40829101*1.52928396
