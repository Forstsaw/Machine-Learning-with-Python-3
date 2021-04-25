import numpy as np
x  = 32
s = 2 #standard derivation
u = 28 #mean

def normal_dst(x,s,u):
	return 1/(s*np.sqrt(2*np.pi))*np.exp(-(x-u)**2/(2*s**2))

x1 = normal_dst(x,s,u)
x2 = normal_dst(34,s,u)
res = x1+x2
print(x1,x2,res)

