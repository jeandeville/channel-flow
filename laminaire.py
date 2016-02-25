# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 22:11:01 2016

@author: jeandeville
"""

from maillage import *;
from laplacien import *;
import matplotlib.pyplot as plt;
import numpy as np


h = 1;
r = 3;

l0 = 0.01;


y = maillage(h,r,l0);

Nb_Pts = np.size(y);


for i in range(Nb_Pts):
    plt.plot([0.0, 10.0], [y[i], y[i]], 'r-', lw=2) 


plt.show()


A = laplacien(y);