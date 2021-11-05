import numpy as np


def line_search_bt(x,func,grad,alpha,t,rho,direction):
   
    alpha=1

    f1=func(x+alpha*direction)

    f2=func(x)+t*alpha*np.matmul(np.transpose(grad),direction)

    while f1>f2:

        f1=func(x+alpha*direction)

        f2=func(x)+t*alpha*np.matmul(np.transpose(grad),direction)
    
        alpha=alpha*rho
    

    return alpha
    
