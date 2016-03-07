# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 17:16:31 2016

@author: jeandeville
"""

#########################################
#   COUCHE LIMITE LAMINAIRE
#   dU/dt =- 1/rho*dP/dx +d/dy [(nu + nu_t) du:dy]
#   dk/dt = nu_t (du/dy)^2 - beta*omega*k + d/dy[(nu + sigma_k * nu_t) dk/dy]
#   domega/dt = gamma (du/dy)^2 - beta*omega^2 + d/dy[(nu + sigma_omega * nu_t) domega/dy]
#########################################

import numpy as np
import matplotlib.pyplot as plt
from maillage import *;
from laplacien import *;
from Initialisation import *

#   Paramètres

#   Paramètres
r = 1.01 # taux de croissance du maillage
dy0= 0.01;# pas de la première cellule 
h = 1 # taille de la moitie du canal
gradient=-1.0; # gradient de pression modifié (1/rho * dPe/dx)
Rstar=100.0; # nombre de rynolds (Rstar=u_tau*h/nu)
u_tau=np.sqrt(h*(-gradient));# vitesse caractéristique de frottement (u_tau=mu.dU/dy|y=0) 
nu=u_tau*h/Rstar; # viscosité dynamique à 0 C
Niter = 300; # nombre d'itérations
dt = 10.; # pas de calcul

#Paramètres de k-omega
sigma_k = 0.5;
sigma_omega = 0.5;
gamma = 5./9.;
beta = 3./40.;
beta_star = 0.09;

####################
####### IL FAUT RAJOUTER LES CONSTANTES SIGMA_K SIGMA_OMEGA ET GAMMA
#######################

#création du maillage
y = maillage(h,r,dy0);

# Def Nb_Pts
Nb_Pts = np.size(y);
Nmax = Nb_Pts-1;

#Def de Gr
Gr = gradient * np.ones(Nb_Pts)

#Solution laminaire
Ulaminaire = 1/(2*nu)*gradient*np.multiply(y,np.add(y,-2*h));


######
#Initialisation (de k0, omega0, u0)
######

Init=initialisation(y,h,Rstar,gradient);
U0=Init[0,:];
nu_t0=Init[1,:];
K0=Init[2,:];
omega0=Init[3,:];

#plt.plot(y,U0,'r')
#plt.plot(y,Ulaminaire,'g')
#plt.show

#print(u0)
#print(k0)
#print(nu_t0)
#print(omega0)
#

Uprec = Ulaminaire#U0;
Kprec = K0;
omegaprec = omega0;
nu_tprec = nu_t0;

Id_tid = np.eye(Nb_Pts);
Id_tid[Nmax,Nmax-1] =-1;

#plt.plot(y,Ulaminaire);
#plt.plot(y,U0);
#plt.show;



for i in range(100):
    ################## DEBUT BOUCLE ##############
    ######### 
    #Mise à jour du vecteur U_n+1:
    # U^(n+1)=[Id-∆t.L(v_t)]^(-1) (U^n-∆t*Gr)
    ########
    A_u=Id_tid-dt*L_turbulent(nu,nu_tprec,1,y);
    b_u=Uprec-dt * Gr;
    b_u[0]=0.0;
    b_u[Nmax]=0.0;
    U = np.dot(np.linalg.inv(A_u), b_u) #on envoie 1 car formule de L_Turbulent: nu + sigma*nu_t
    
    
    
    ##########
    #Mise à jour vecteur K
    # 〖K^(n+1)〗=[I_d+∆t.M(β^*,ω^n )-∆t.L(υ,υ_t^n,σ_k)]^(-1) (〖∆t*N(u^n,υ_t^n)〗^ +K^n )
    ##########
    A_k=Id_tid+dt*M_turbulent(beta_star, omegaprec)-dt*L_turbulent(nu,nu_tprec,sigma_k,y);
    b_k=dt *  N_turbulent(Uprec,nu_tprec,y)+ Kprec;
    b_k[0]=0.0;
    b_k[Nmax]=0.0;
    K = np.dot(np.linalg.inv(A_k), b_k)
    
    
    
    ##########
    #Mise à jour vecteur W
    # 〖W^(n+1)〗=[I_d+∆t.M(β,ω^n )-∆t.L(υ,υ_t^n,σ_ω)]^(-1) (〖∆t.N(u^n,γ)〗^ +W^n )
    ##########
    A_w=Id_tid+dt*M_turbulent(beta, omegaprec)-dt*L_turbulent(nu,nu_tprec,sigma_omega,y);
    b_w=dt *  N_turbulent(Uprec,gamma,y)+ omegaprec;
    b_w[0] = 10 * 6 * nu / (beta * (dy0)**2 );# condition à la paroi de omega
    b_w[Nmax]=0.0;
    W = np.dot(np.linalg.inv(A_w), b_w)
    
    
    ##########
    #Mise à jour nu_t
    # nu_t = k / omega
    #########
    nu_t = np.multiply(K,np.power(W,-1));
    
    
    
    
    
    
    plt.clf()
    #plt.plot(y,U,'g')
    #plt.plot(y,U0,'r')
    
    u0plus=1.0/u_tau*U0
    yplus=y*Rstar/h
    uplus=1.0/u_tau*U
    #plt.plot(yplus,uplus,'g');
    #plt.plot(yplus,u0plus,'r');
    #plt.xscale('log');
    #plt.ylim((0,30))
    #plt.xlim((1,1000))
    #plt.plot(y,Ulaminaire)
    
    
    plt.plot(yplus,nu_t);
    #plt.plot(yplus,k);
    #plt.plot(W);
    plt.show();
    plt.pause(0.0001)
    
    ###### Stockage du vecteurs actuels dans les vecteurs précs
    Uprec = U;
    Kprec = K;
    omegaprec = W;
    nu_tprec = nu_t;
    
    print(i);
    
    ################## FIN BOUCLE ##############
