import numpy as np


def line_search_grg(func,dfdd,sk,dk,phps,php
   
    alpha=1

    b=0.5

    t=0.3

    x_f=dk-alpha*dfdd(dk),sk+alpha*np.transpose(np.matmul(np.matmul(np.linalg.inv(phps(dk)),phpd(dk)),np.transponse(dfdd(dk))))

    f=func(x_f)

    x_phi=[dk,sk]

    phi=func(x_phi)+t*alpha*dfdd(dk)

    while f>phi:

        alpha=alpha*b

        x_f=dk-alpha*dfdd(dk),sk+alpha*np.transpose(np.matmul(np.matmul(np.linalg.inv(phps(dk)),phpd(dk)),np.transponse(dfdd(dk))))

        f=func(x_f)

        x_phi=[dk,sk]

        phi=func(x_phi)+t*alpha*dfdd(dk)

    return alpha
    
