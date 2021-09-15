import numpy as np


def line_search_bt(x_init,x,func,grad,alpha,t,rho):
   
    alpha=1

    direction=-1*grad(x)
   

    in1=x+alpha*np.matmul(np.transpose(grad(x)),direction)
    
    f1=func(in1)
    f2=func(x)+t*alpha*np.matmul(np.transpose(grad(x)),direction)
    
    while f1>f2:

        in1=x+alpha*np.matmul(np.transpose(grad(x)),direction)
        f1=func(in1)
        f2=func(x)+t*alpha*np.matmul(np.transpose(grad(x)),direction)

        alpha=alpha*rho
    

    return alpha
    
