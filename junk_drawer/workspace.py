import numpy as np
import matplotlib.pyplot as plt
from line_search_bt import *

x_i=np.array([1,1])
convergence_criteria=.0001
alpha=1
error=1
err=[]
itt=[]
count=1

def func(x):
    return 5*x[0]**2+12*x[0]*x[1]-8*x[1]+10*x[1]**2-14*x[1]+5

def gradient(x):
   return np.array([10*x[0]+12*x[1]-8,12*x[0]+20*x[1]-14])

while error > convergence_criteria:
    x_k=x_i
    x_i=x_k-alpha*gradient(x_k)
    error=abs(max(x_i-x_k))
    err.append(float(error))

    alpha=line_search_bt(x_i,func,gradient,alpha,.8,.8)

x=np.array([-2*x_i[0]-3*x_i[1]+1,x_i[0],x_i[1]])
print(x)

plt.plot(err)
plt.yscale("log")