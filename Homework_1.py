import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint

##bounds=Bounds([-10,10],[-10,10],[-10,10],[-10,10],[-10,10])
bounds=Bounds([-10,-10,-10,-10,-10],[10,10,10,10,10])

linear_constraint = LinearConstraint([[1,3,0,0,0],[0,0,1,1,-2],[0,1,0,0,-1]],[1,1,1], [1,1,1])

def f(x):
    return (x[0]-x[1]**2 + (x[1]+x[2]-2)**2 + (x[3]-1)**2+(x[4]-1)**2)

x0 = np.array([1, 2, 3, 4, 5])

res = minimize(f, x0,constraints=linear_constraint, bounds=bounds)

print(res)