# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 17:16:31 2016

@author: jeandeville
"""

#########################################
#   COUCHE LIMITE TURBULENTE
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
r = 1.01 # taux de croissance du maillage

#h = 1 # taille de la moitie du canal
#gradient=-1.0# gradient de pression modifié (1/rho * dPe/dx)
#Rstar=950.0; # nombre de rynolds (Rstar=u_tau*h/nu)
#u_tau=np.sqrt(h*(-gradient));# vitesse caractéristique de frottement (u_tau=mu.dU/dy|y=0) 
#nu=u_tau*h/Rstar; # viscosité dynamique à 0 C
#Niter = 300; # nombre d'itérations
#dt = 10.; # pas de calcul

Rstar=950.0; # nombre de rynolds (Rstar=u_tau*h/nu)
u_tau = 0.045390026;
nu = 1/(20580.0);

h = Rstar*nu / u_tau;
gradient = - u_tau**2/h;


Niter = 300; # nombre d'itérations
dt = 100.; # pas de calcul

#Paramètres de k-omega
sigma_k = 0.5;
sigma_omega = 0.5;
gamma = 5./9.;
beta = 3./40.;
beta_star = 0.09;



#création du maillage
dy0plus = 0.031230073;
dy0=dy0plus*h/Rstar;# pas de la première cellule 
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


#Adimensionalisation
u0plus=1.0/u_tau*U0
yplus=y*Rstar/h
#Lois de paroi
kappa = 0.41;
Cplus = 5.0;
Uwall_sublayer = yplus;
Uwall_loglayer = 1/kappa * np.log(yplus) + Cplus;


Uprec = U0;
Kprec = K0;
omegaprec = omega0;
nu_tprec = nu_t0;

Id_tid = np.eye(Nb_Pts); #correspond à une matrice identité prenant en compte lesCAL c'est à dire dernièreligne à 0...0 -1 1 ("I_tild")
Id_tid[Nmax,Nmax-1] =-1;


#chargement de uplus et y plus DNS NASA
lines = np.loadtxt("Re950.txt")
yplusDNS = lines[0:192,1]
uplusDNS = lines[0:192,2]



for i in range(300):
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
    # W_n+1=[Id+∆t.M(β,ω_n)-∆t.L(nu,nu_tprec,sigma_omega)]^(-1) (∆t.N(u_n,γ) +W_n )
    ##########
    A_w=Id_tid+dt*M_turbulent(beta, omegaprec)-dt*L_turbulent(nu,nu_tprec,sigma_omega,y);
    b_w=dt *  N_turbulent(Uprec,gamma,y)+ omegaprec;
    b_w[0] = 60.0 * nu / (beta * (dy0)**2 );# condition à la paroi de omega
    b_w[Nmax]=0.0;
    W = np.dot(np.linalg.inv(A_w), b_w)    
    
    ##########
    #Mise à jour nu_t
    # nu_t = k / omega
    #########
    nu_t = np.multiply(K,np.power(W,-1));

    ###########
    #Transformation des résultats sont grandeurs adimensionnées
    ###########
    uplus=1.0/u_tau*U    
    
    #plt.plot(yplus,nu_t);
    #plt.plot(yplus,k);
    #plt.plot(W);
    #plt.show();
    #plt.pause(0.0001)
    
    plt.clf()
    plt.plot(yplusDNS,uplusDNS,'g',label='DNS');
    plt.plot(yplus,uplus,'r',label='k-w'); 
    plt.plot(yplus,u0plus,'b',label='mixing length');
    plt.plot(yplus[1:160],Uwall_sublayer[1:160],'y',label='wall law');
    plt.plot(yplus[120:Nmax],Uwall_loglayer[120:Nmax],'y');
    plt.ylim((0,25))
    plt.xscale('log')
    plt.title('U+=f(y+) pour DNS, RANS(k-w), Lmelange')
    #plt.legend(bbox_to_anchor=(1.05,1), loc=-1, borderaxespad=0.)
    plt.legend(loc=2);
    plt.show()
    plt.pause(0.0001)
    
    
    
    ###### Stockage du vecteurs actuels dans les vecteurs préc
    Uprec = U;
    Kprec = K;
    omegaprec = W;
    nu_tprec = nu_t;
    
    print(i); #affichage de l'itération
    
    ################## FIN BOUCLE ##############
