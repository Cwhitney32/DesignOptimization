import logging
import math
import random
import numpy as np
import time
import torch as t
import torch.nn as nn
from torch import optim
from torch.nn import utils
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

n=10 #random intial states
state=np.zeros((n,6))

for i in range(n):
    state[i-1,:] = [4.,.2,-3.,.2,1.,0.]

state=t.tensor(state)
            
print(state)