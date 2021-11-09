import numpy as np


def line_search_grg(func,dfdd,sk,dk,phps,phpd):
   
    alpha=1

    b=0.5

    t=0.3

    f_sk=sk + alpha * np.transpose(np.matmul(np.linalg.inv(phps(np.concatenate([sk,dk]))),phpd(np.concatenate([sk,dk])))*np.transpose(dfdd(np.concatenate([sk,dk]))))
    f_dk=dk-alpha*dfdd(np.concatenate([sk,dk]))

    x_f=np.concatenate([f_sk,f_dk])

    f=func(x_f)

    x_phi=np.concatenate([sk,dk])

    phi=func(x_phi)+t*alpha*dfdd(np.concatenate([sk,dk]))

    while f>phi:

        alpha=alpha*b

        f_sk=sk + alpha * np.transpose(np.matmul(np.linalg.inv(phps(np.concatenate([sk,dk]))),phpd(np.concatenate([sk,dk])))*np.transpose(dfdd(np.concatenate([sk,dk]))))
        f_dk=dk-alpha*dfdd(np.concatenate([sk,dk]))

        x_f=np.concatenate([f_sk,f_dk])

        f=func(x_f)

        x_phi=np.concatenate([sk,dk])

        phi=func(x_phi)+t*alpha*dfdd(np.concatenate([sk,dk]))

    return alpha
    
