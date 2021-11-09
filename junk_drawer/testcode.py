import numpy as np
import matplotlib.pyplot as plt
from line_search_grg import*
from sk_solve import*

alpha=1
conv=1
error=1
sol=[]
err=[]
itt=[]


def f(x):
    return x[0]**2+x[1]**2+x[2]**2

sol=f