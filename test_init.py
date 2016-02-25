# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:30:39 2016

@author: Said
"""


import numpy as np
import matplotlib.pyplot as plt
from maillage import *
from laplacien import *
from Initialisation import *

#   Paramètres
r = 1.01 # taux de croissance du maillage
dy0= 0.01;# pas de la première cellule 
h = 1 # taille de la moitie du canal
gradient=-1.0; # gradient de pression modifié (1/rho * dPe/dx)
Rstar=1000.0; # nombre de rynolds (Rstar=u_tau*h/nu)
u_tau=np.sqrt(h*(-gradient));# vitesse caractéristique de frottement (u_tau=mu.dU/dy|y=0) 
nu=u_tau*h/Rstar; # viscosité dynamique à 0 C
kappa=0.41;
     
     
#création du maillage
y = maillage(h,r,dy0);

IN=initialisation(y,h,Rstar,gradient);

