# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 18:55:28 2016

@author: Said et Jean
"""

import math
import numpy as np
#
## l'ancien laplacien avec la mauvaise dérivée seconde
#def laplacien (y):
#    
#    Nb_Pts = np.size(y);
#    Nmax = Nb_Pts -1;
#    
#    dy = np.zeros(Nmax);
#    
#    for i in range(Nmax):
#        dy[i] = y[i+1] - y[i];
#        
#    A = np.zeros([Nb_Pts,Nb_Pts]);
#    
#    for i in range (1,Nb_Pts-1): #i va de 1 à Nb_Pts - 2
#        dy_moyen = 0.5 * (dy[i-1] + dy[i]);
#        A[i,i] = -2/dy_moyen**2;
#        A[i,i-1] = 1/dy_moyen**2;
#        A[i,i+1] = 1/dy_moyen**2;
#        
#    return A;


def laplacien (y):
    
    Nb_Pts = np.size(y);
    Nmax = Nb_Pts -1;
    
    dy = np.zeros(Nmax);
    
    for i in range(Nmax):
        dy[i] = y[i+1] - y[i];
        
    A = np.zeros([Nb_Pts,Nb_Pts]);
    
    for i in range (1,Nb_Pts-1): #i va de 1 à Nb_Pts - 2
        dy_moyen = 0.5 * (dy[i-1] + dy[i]);
        
        A[i,i] = -2.0/(dy[i-1]*dy[i]);
        A[i,i-1] = 1.0/(dy_moyen*dy[i-1]);
        A[i,i+1] = 1.0/(dy_moyen*dy[i]);
        
    return A;
    
    

#def laplacien_turbulent (y, nu_t,nu,sigma): #Le laplacien turbulent n'est pas vraiment un laplacien, il s'agit du terme en dy((nu + sigma*nu_t) du/dy)
#    
#    Nb_Pts = np.size(y);
#    Nmax = Nb_Pts -1;
#    
#    dy = np.zeros(Nmax);
#    
#    for i in range(Nmax):
#        dy[i] = y[i+1] - y[i];
#        
#    A = np.zeros([Nb_Pts,Nb_Pts]);
#    
#    for i in range (1,Nb_Pts-1): #i va de 1 à Nb_Pts - 2
#        dy_moyen = 0.5 * (dy[i-1] + dy[i]);
#        nu_tsup = 0,5 * (nu_t[i+1] + nu_t[i]) #correspond à "Ci[i+0,5]"
#        nu_tinf = 0,5 * (nu_t[i] + nu_t[i-1]) #correspond à "Ci[i-0,5]"
#        
#        A[i,i] = -(nu+sigma*nu_tsup)/(dy[i+1]*dy_moyen)-(nu+sigma*nu_tinf)/(dy[i-1]*dy_moyen);
#        A[i,i-1] = (nu+sigma*nu_tinf)/(dy_moyen*dy[i-1]);
#        A[i,i+1] = (nu+sigma*nu_tsup)/(dy_moyen*dy[i]);
#        
#    return A;
    
    
def inverse_tdgauche_turbulent (dt, beta, omega): #correspond à l'inverse du terme de gauche (1+dt*Beta*omega(i)); c'est une matrice diagonale

    Nb_Pts = np.size(omega);
    mat_inverse = np.zeros([Nb_Pts,Nb_Pts]);
    
    for i in range(Nb_Pts):
        mat_inverse[i,i] = 1.0/(1.0+dt*beta*omega[i]); 
        
    return mat_inverse;
    

def N_turbulent(U,nu_t,y): #correspond au terme de droite "N" du rapport, correspond au terme en carré de la dérivée de la vitesse

    Nb_Pts = np.size(U); #nombre de points
    Nmax = Nb_Pts - 1; #nombre d'intervalles
    
    dy = np.zeros(Nmax);
    
    for i in range(Nmax):
        dy[i] = y[i+1] - y[i];
        
        
    if np.size(nu_t)==1:
        nu = nu_t * np.ones(Nb_Pts); #coorespond à l'équation de omega: on a une constante gamme au lieu d'un vecteur de nu_t
    else:
        nu = nu_t; #correspond à l'équation de k
        
    N = np.zeros(Nb_Pts);
    
    for i in range(1,Nb_Pts-1): # le premier et le dernier terme reste à 0 !!!!!!
        N[i] = nu[i] * (0.5 *((U[i+1]-U[i])/dy[i]+(U[i]-U[i-1])/dy[i-1])**2);
        
    return N;
    
    
def L_turbulent(nu,nu_t,sigma,y): #correspond au terme de droite "L" du rapport, retourne une matrice qu'il faudra multiplier par Kn ou Wn

    Nb_Pts = np.size(y); #nombre de points
    Nmax = Nb_Pts - 1; #nombre d'intervalles  
    
    dy = np.zeros(Nmax);
    
    for i in range(Nmax):
        dy[i] = y[i+1] - y[i];
        
        
    L = np.zeros([Nb_Pts,Nb_Pts]);
    
    for i in range(1,Nb_Pts-1):
        dymoyen = 0.5 * (dy[i]+dy[i-1]);  
        nu_t_plus = 0.5 * (nu_t[i+1] + nu_t[i]);
        nu_t_moins = 0.5 * (nu_t[i] + nu_t[i-1]);
        
        L[i,i] = -1.0/dymoyen * ((nu + sigma * nu_t_plus)/dy[i] + (nu + sigma*nu_t_moins)/dy[i-1]);
        L[i,i-1] = (nu+sigma*nu_t_moins)/(dymoyen*dy[i-1]);
        L[i,i+1] = (nu+sigma*nu_t_plus)/(dymoyen*dy[i]);
        
    
    return L;














