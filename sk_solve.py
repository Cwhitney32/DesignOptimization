import numpy as np


def sk_solve(h,dk,sk,e,phps):
   
    h_sol=h(np.concatenate([sk,dk])) + np.matmul(phps(np.concatenate([sk,dk])),sk)  

    while np.linalg.norm(h_sol)>e:
    
        phps_sol=phps(np.concatenate([sk,dk]))

        sk=sk-np.matmul(np.linalg.inv(phps(np.concatenate([sk,dk]))),h(np.concatenate([sk,dk])))

        h_sol = h(np.concatenate([sk,dk]))

    return sk
    
