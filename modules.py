# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 18:55:28 2016

@author: Said et Jean
"""

import math
import numpy as np


def maillage (h,r,l0):
    
    N0=math.floor(math.log(1+0.5*(r-1)*h/l0)/math.log(r))-1;
    N0 = int(N0);
    lmax = l0*r**N0;
    d = h - l0*((1-r**(N0+1))/(1-r));
    N1 = math.floor(d/lmax);
    N1 = int(N1);
    lcste = d/N1;
    
    y = np.zeros(N0+N1+2);
    
    print(N0, N1);
    
    for i in range(N0+2):
        y[i] =l0 * (1-r**i)/(1-r);
        
    for i in range(N0+2,N0+2+N1):
        y[i] = y[N0+1] + (i-N0-1)*lcste;
        
        
    return y;




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
  
  
def L_turbulent(nu,nu_t,sigma,y): #correspond au terme de droite "L" du rapport, retourne une matrice qu'il faudra multiplier par Kn ou Wn

    Nb_Pts = np.size(y); #nombre de points
    Nmax = Nb_Pts - 1; #nombre d'intervalles  
    
    dy = np.zeros(Nmax);
    
    for i in range(Nmax):
        dy[i] = y[i+1] - y[i];
        
        
    L = np.zeros([Nb_Pts,Nb_Pts]);
    
    for i in range(1,Nb_Pts-1): # de 1 à Nb-2 pour prenrdre en compte les bc
        dymoyen = 0.5 * (dy[i]+dy[i-1]);  
        nu_t_plus = 0.5 * (nu_t[i+1] + nu_t[i]);
        nu_t_moins = 0.5 * (nu_t[i] + nu_t[i-1]);
        
        L[i,i] = -1.0/dymoyen * ((nu + sigma * nu_t_plus)/dy[i] + (nu + sigma*nu_t_moins)/dy[i-1]);
        L[i,i-1] = (nu+sigma*nu_t_moins)/(dymoyen*dy[i-1]);
        L[i,i+1] = (nu+sigma*nu_t_plus)/(dymoyen*dy[i]);
        
    
    return L;  
    
    
def M_turbulent (beta, omega): #correspond à  la matrice diag (M(i,i)=Beta*omega(i)); 

    Nb_Pts = np.size(omega);
    mat = np.zeros([Nb_Pts,Nb_Pts]);
    
    for i in range(1,Nb_Pts-1):  # de 1 à Nb-2 pour prenrdre en compte les bc
        mat[i,i] = beta*omega[i]; 
    
    return mat;
    


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
    
    for i in range(1,Nb_Pts-1):  # de 1 à Nb-2 pour prenrdre en compte les bc
        #N[i] = nu[i] * (0.5 *((U[i+1]-U[i])/dy[i]+(U[i]-U[i-1])/dy[i-1])**2);
        N[i] = nu[i] * (((U[i+1]-U[i-1])/(dy[i]+dy[i-1]))**2);
        
    return N;
    
    

def cisaillement_plus(nu,nu_t,uplus,yplus):
    Nb_Pts = np.size(uplus); #nombre de points
    Nmax = Nb_Pts - 1; #nombre d'intervalles
    
    dy = np.zeros(Nmax);
    
    for i in range(Nmax):
        dy[i] = yplus[i+1] - yplus[i];
        
    
    nu_tot_dudy = np.zeros(Nb_Pts);
    
    for i in range(1,Nb_Pts-1):  # de 1 à Nb-2 pour prenrdre en compte les bc
        #N[i] = nu[i] * (0.5 *((U[i+1]-U[i])/dy[i]+(U[i]-U[i-1])/dy[i-1])**2);
        nu_tot_dudy[i] = (1.0+nu_t[i]/nu)* (((uplus[i+1]-uplus[i-1])/(dy[i]+dy[i-1])));
    nu_tot_dudy[0]=(1.0+nu_t[0]/nu)* (((uplus[1]-uplus[0])/(dy[0]))**2);
    nu_tot_dudy[Nmax]=(1.0+nu_t[0]/nu)* (((uplus[Nmax]-uplus[Nmax-1])/(dy[Nmax-1])));
    return nu_tot_dudy;    



def dudy_plus(uplus,yplus):
    Nb_Pts = np.size(uplus); #nombre de points
    Nmax = Nb_Pts - 1; #nombre d'intervalles
    
    dy = np.zeros(Nmax);
    
    for i in range(Nmax):
        dy[i] = yplus[i+1] - yplus[i];
        
    
    dudy_plus = np.zeros(Nb_Pts);
    
    for i in range(1,Nb_Pts-1):  # de 1 à Nb-2 pour prenrdre en compte les bc
        dudy_plus[i] =(((uplus[i+1]-uplus[i-1])/(dy[i]+dy[i-1])));
    dudy_plus[0]=(((uplus[1]-uplus[0])/(dy[0]))**2);
    dudy_plus[Nmax]=(((uplus[Nmax]-uplus[Nmax-1])/(dy[Nmax-1])));
    
    return dudy_plus; 





def thomas (A,b):

    #A = np.array(A);
    #b = np.array(b);    
    
    NA = len(A);
    Nb = len(b);

   
    U = np.zeros(NA);
     
    if (NA != Nb):
        print("ERREUR DE DIMENSION DANS LA MATRICE A ET b");
    
    else:
        A[0,1]=A[0,1]/A[0,0];
        b[0] = b[0]/A[0,0];
        
        for i in range(1,Nb-1):
            A[i,i+1] = A[i,i+1]/(A[i,i]-A[i,i-1]*A[i-1,i]);
            b[i]=(b[i]-A[i,i-1]*b[i-1])/(A[i,i]-A[i,i-1]*A[i-1,i]);

        b[Nb-1]=(b[Nb-1]-A[Nb-1,Nb-2]*b[Nb-2])/(A[Nb-1,Nb-1]-A[Nb-1,Nb-2]*A[Nb-2,Nb-1]);



        U[Nb-1] = b[Nb-1];
        
        for i in range(1,Nb):
            U[Nb-1-i] = b[Nb-1-i] - A[Nb-1-i,Nb-i]*U[Nb-i];
                

    return U;



