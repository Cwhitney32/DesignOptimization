import numpy as np


def sk_solve(h,dk,sk):
   
    alpha=1

    f1=func(x+alpha*direction)

    f2=func(x)+t*alpha*np.matmul(np.transpose(grad),direction)

    while f1>f2:

        f1=func(x+alpha*direction)

        f2=func(x)+t*alpha*np.matmul(np.transpose(grad),direction)
    
        alpha=alpha*rho
    

    return sk_new
    
