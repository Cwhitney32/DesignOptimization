import numpy as np


def sk_solve(h,dk,sk,e,phps):
   
    h_sol=h(sk,dk)+phps(sk)*sk

    while h_sol>e:
    
        phps_sol=phps(sk)

        sk=sk-np.linalg.inv((phps(sk))*h(sk)

        h_sol = h(sk,dk)

return sk
    
