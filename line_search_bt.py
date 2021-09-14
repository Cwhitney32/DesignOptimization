import numpy as np


def line_search_bt(x,func,grad,alpha,t,rho):
   
    alpha=1

    direction=-grad(x)

    
    f1=func(x+alpha*direction)

    f2=func(x)+t*alpha*np.matmul(np.transpose(grad(x)),direction)
    
    while f1>f2:

     
        f1=func(x+alpha*direction)

        f2=func(x)+t*alpha*np.matmul(np.transpose(grad(x)),direction)
    
        alpha=alpha*rho
    

    return alpha
    
