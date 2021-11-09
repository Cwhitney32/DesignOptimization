import numpy as np
import math 
import matplotlib.pyplot as plt
from line_search_bt import *

x_i=np.array([0,0])
x_init=x_i
convergence_criteria=.001
alpha=1
conv=1
error=1
err=[]
itt=[]
count=1
xstar=[-0.14248091,0.7854586]

def func(x):
    return 5*x[0]**2+12*x[0]*x[1]-8*x[0]+10*x[1]**2-14*x[1]+5

def gradient(x):
   return np.array([10*x[0]+12*x[1]-8,12*x[0]+20*x[1]-14])

while conv > convergence_criteria:

    x_k=x_i

    alpha=line_search_bt(x_i,func,gradient,alpha,.5,.5)
    
    x_i=x_k-alpha*gradient(x_k)

    error=func(x_k)-func(xstar)

    conv=np.linalg.norm(gradient(x_i))
    
    err.append(float(error))

  

x=np.array([-2*x_i[0]-3*x_i[1]+1,x_i[0],x_i[1]])

print(x)

plt.plot(err)
plt.yscale("log")