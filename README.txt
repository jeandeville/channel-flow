This k-omega code is composed of three different files
1-The "Initialisation" file initialises k, omega, nu_t and U with a mixing length model for turbulence
2-The "modules" file contains all the modules needed in the main k-omega code in order to have a more compact and structured code.
This includes the mesh generation, the different derivates and laplacien used in the code, and the thomas algorithm used to inverse tridiagonal matrices.
3-The "main" module which is composed of the k-omega algorithm and a module to display different graphs.

In order to use our code, the user only needs to input values in the "main" file. "modules" and "initialisation" must not be modified.
Values that must be entered are as follows: 
r:increase in height of the grid, usually between 1.01 and 1.10
Niter: number of iterations
dt: timestep (the implicit code allows high dt values and thus acceleration of the convergence)
cas: two choices are possible: "Re=950" or "Re=180", corresponding to the two Reynolds number investigated in our project.

Graphs produced by our code are compared to two codes available online: a DNS code produced by the Universidad Politecnica de Madrid (for Reynolds 950)
and an SST code produced by Bertrand Aupoix from ONERA France (for Re=180).

A memoir on our project is available (in French) for further details in the development our code.

Enjoy.
Jean Deville, Said Ouhamou
ISAE Supaero, Toulouse, France, 29 March 2016
End of the year Fluid Dynamics Project