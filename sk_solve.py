import numpy as np


def sk_solve(h,dk,sk,e,phps):
   
    h_sol=h(np.concatenate([sk,dk])) + np.matmul(phps(np.concatenate([sk,dk])),sk)  
    conv = np.linalg.norm(h_sol)

    while conv > e:
    
        phps_sol=phps(np.concatenate([sk,dk]))

        h_sol=h(np.concatenate([sk,dk]))

        sk=np.ndarray.transpose(sk)

        sk=np.ndarray.transpose(sk)-np.matmul(np.linalg.inv(phps_sol),h_sol)

        sk=np.ndarray.transpose(sk)
        
        h_sol = h(np.concatenate([sk,dk]))

        conv = np.linalg.norm(h_sol)

    return sk
    
