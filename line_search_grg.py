import numpy as np


def line_search_grg(f,dfdd,sk,dk):
   
    alpha=1

    b=0.5

    t=0.3

    f1=func(x+alpha*direction)

    f2=func(x)+t*alpha*np.matmul(np.transpose(grad),direction)

    while f1>f2:

        f1=func(x+alpha*direction)

        f2=func(x)+t*alpha*np.matmul(np.transpose(grad),direction)
    
        alpha=alpha*rho
    

    return alpha
    
