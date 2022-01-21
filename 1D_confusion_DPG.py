# This FEniCS code solves the 1D convection-diffusion equation using the DPG method
# Last tested January 2021

import os
import mshr
import numpy as np
import time
import dolfin
from fenics import *
from dolfin import *
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#If MPI
parameters['ghost_mode']='shared_facet'
comm = MPI.comm_world
rank = comm.Get_rank()



File_name = 'Figures'
# Adaptive data
THETA = 0.5
interpdeg = 8
MAX_ITER = 8
REF_TYPE = 1 ## 1 is uniform 2 is adaptive
TOL = 1E-12


L2vect_TOT = np.zeros(MAX_ITER)
L2vectQ = np.zeros(MAX_ITER)
L2vect = np.zeros(MAX_ITER)
Evect = np.zeros(MAX_ITER)
Dofsvect = np.zeros(MAX_ITER)
L2vectRate = np.zeros(MAX_ITER)
L2vectQRate = np.zeros(MAX_ITER)
EvectRate= np.zeros(MAX_ITER)
hvect = np.zeros(MAX_ITER)
TIMINGVEC= np.zeros(MAX_ITER)

pprimal = 1
p_add = 2
testdeg = pprimal + p_add

epsilon = 0.01



if REF_TYPE ==1:
    TrialTypeDir = os.path.join(File_name, 'Peclet_'+str(1/epsilon)+'_uniform_DPG_Ptrial' + str(pprimal)+ '_Ptest'+ str(testdeg))

if REF_TYPE == 2:
        TrialTypeDir = os.path.join(File_name, 'Peclet_'+str(1/epsilon)+'_adapt_DPG_Ptrial' + str(pprimal)+ '_Ptest'+ str(testdeg))

if not os.path.isdir(TrialTypeDir): os.makedirs(TrialTypeDir)

#Mesh
N = 2
M = UnitIntervalMesh(N)



for level in range(MAX_ITER):
    level_step = level


    if level == 0:
       tstart = time.time()


    # Define the necessary function spaces: 
    # Discontinuous polynomial space for vector valued test function
    #U
    V1 = FiniteElement( "DG", M.ufl_cell() ,pprimal)
    #UHAT - constants on interface
    V2 = FiniteElement( "CG", M.ufl_cell() ,pprimal)['facet']
    #Sigma
    V3 = FiniteElement( "DG", M.ufl_cell() ,pprimal)
    #SIGHAT- constants on interface
    V4 = FiniteElement( "CG", M.ufl_cell() ,pprimal)['facet']
    #Error indicators psi phi
    V5 = FiniteElement( "DG", M.ufl_cell() ,testdeg)
    V6 = FiniteElement( "DG", M.ufl_cell() ,testdeg)

    V = FunctionSpace(M, MixedElement(V1,V2,V3,V4,V5,V6))
    Vtest = FunctionSpace(M, MixedElement(V1,V2,V3,V4))

    # Define the test and trial functions
    u, uhat, sig, sighat, psi, phi  = TrialFunctions(V)
    a, ahat, b, bhat, v, tau = TestFunctions(V)
    

    # To compute the inner product we need the element diameter 
    h = Circumradius(M)

    hmax = M.hmax()

    # n is the element normal vector needed to compute the boundary 
    # "integrals"
    n = FacetNormal(M)


    #Exact soln
    uexact = Expression('(x[0]+(exp(1*x[0]/epsilon)-1)/(1-exp(1/epsilon)))',degree = interpdeg,domain = M,epsilon=epsilon)
    
    # Source term f
    f = -epsilon*uexact.dx(0).dx(0) + uexact.dx(0)


    # Ultraweak form, the other DPG weak forms can be acieved by slight modification
    innerp = ( inner(tau, phi) + inner(v, psi)  + inner(tau.dx(0), phi.dx(0)) + inner(v.dx(0), psi.dx(0))  )*dx

    buv = ( inner(sig,tau) + inner(u*epsilon,tau.dx(0))+ inner(sig,v.dx(0)) - inner(u,v.dx(0)) )*dx + ( -inner(uhat('+'),jump(tau)) - inner(sighat('+'),jump(v)) +inner(uhat('+'),jump(v)) )*dS + ( -inner(uhat,tau) - inner(sighat,v) +inner(uhat,v) )*ds
        
    brs = ( inner(b,phi) + inner(a*epsilon,phi.dx(0))+ inner(b,psi.dx(0)) - inner(a,psi.dx(0)) )*dx + ( -inner(ahat('+'),jump(phi)) - inner(bhat('+'),jump(psi)) +inner(ahat('+'),jump(psi)))*dS + ( -inner(ahat,phi) - inner(bhat,psi) +inner(ahat,psi) )*ds
    

    # Add all contributions to the LHS
    a = buv + brs + innerp

    # Define the load functional
    L = 1*inner(v,f)*dx

        
    def Xzero(x, on_boundary):
        return on_boundary and (near(x[0],0, 1e-14))
    def Xzone(x, on_boundary):
        return on_boundary and (near(x[0],1, 1e-14))

        
    # Dirichlet BC is applied  on trial function uhat
    bc1 = DirichletBC(V.sub(1), Constant(0.0), Xzero)
    # Dirichlet BC is applied on trial function uhat
    bc2 = DirichletBC(V.sub(1), Constant(0.0),Xzone)

    # Define the solution
    sol0 = Function(V)
    print("dofs", sol0.vector().size())
    soltest = Function(Vtest)
    Dofsvect[level] = soltest.vector().size()
    

    
    if sol0.vector().size() > 200000000:
        print("Too many dofs at level %d" %level)
        break


    # Call the solver
    solve(a==L, sol0, [bc1, bc2], solver_parameters = {'linear_solver' : 'mumps'})

    # Split the solution vector
    u0, uhat0, sig0, sighat0, psi0, phi0 = sol0.split(True)

    #compute errors
    e0r = u0 - uexact
    qe0r = sig0 - epsilon*uexact.dx(0)


    # compute error indicators from the DPG mixed variable e
    PWTEST = FunctionSpace(M,"DG", 0)
    c  = TestFunction(PWTEST)
    gplot = Function(PWTEST)
    ge = ( inner(phi0, phi0)*c + inner(psi0, psi0)*c  + inner(psi0.dx(0), psi0.dx(0))*c + inner(phi0.dx(0), phi0.dx(0))*c  )*dx
    g = assemble(ge)
    gplot.vector()[:]=g

    # Compute inner products
    Ee = assemble(( inner(phi0, phi0) + inner(psi0, psi0)  + inner(psi0.dx(0), psi0.dx(0)) + inner(phi0.dx(0), phi0.dx(0))   )*dx)
    L2r = assemble(inner(e0r,e0r)*dx)
    L2qr = assemble(inner(qe0r,qe0r)*dx)
    
    # Compute norms
    E = sqrt(Ee)
    L2e = sqrt(L2r)
    L2qe = sqrt(L2qr)
    L2_err = sqrt(L2r+L2qr)
    

    L2vect[level] = L2e
    L2vectQ[level] = L2qe
    L2vect_TOT[level] = L2_err

    Evect[level] = E
    hvect[level] = hmax
    
    if REF_TYPE ==1 and level>1:
       L2vectRate[level] = ln(L2vect[level]/L2vect[level-1])/ln(hvect[level]/hvect[level-1])
       L2vectQRate[level] = ln(L2vectQ[level]/L2vectQ[level-1])/ln(hvect[level]/hvect[level-1])
       EvectRate[level] = ln(Evect[level]/Evect[level-1])/ln(hvect[level]/hvect[level-1])


    if ((level >= 0)and(rank == 0)):
        print("l2 norm u ", L2vect[level])
        print("l2 norm sigma ", L2vectQ[level])
        print("Energy error", Evect[level])
        if (REF_TYPE==1):
            print(" rate 'Energy'     ",  EvectRate[level])
            print(" rate 'L2u'    ",  L2vectRate[level])
            print(" rate 'L2q'     ",  L2vectQRate[level])
        
    # Plot the solution mesh
    fig, ax1 = plt.subplots()
    cf = plot(M)
    data_filename = os.path.join(TrialTypeDir, 'mesh%s.pdf'%(level))
    fig.savefig(data_filename, format='pdf', transparent=True)
    plt.close()

    # Mark cells for refinement
    cell_markers = MeshFunction("bool", M, M.topology().dim())


    if REF_TYPE == 2:
        g0 = sorted(g, reverse=True)[int(len(g)*THETA)]
        for cellm in cells(M):
            cell_markers[cellm] = g[cellm.index()] > g0

    if REF_TYPE == 1:
    	M = refine(M) #uniform
    if REF_TYPE == 2:
        M = refine(M,cell_markers) #adaptive

    plt.style.use('classic')

  
        
        

    
if REF_TYPE == 2 or REF_TYPE ==1:
    fig, ax1  = plt.subplots()
    ax1.set_ylabel('Error norm')
    ax1.set_xlabel('dofs')
    plt.loglog(Dofsvect[:MAX_ITER],L2vect[:MAX_ITER],linewidth=2.0)
    ax1.legend(['$||u-u^h||_{L^2(\Omega)}$'], loc='best')
    data_filename = os.path.join(TrialTypeDir, 'L2err_final.pdf')
    fig.savefig(data_filename, format='pdf', transparent=True)
    plt.close()


if REF_TYPE == 2 or REF_TYPE ==1:
    fig, ax1  = plt.subplots()
    ax1.set_ylabel('Error norm')
    ax1.set_xlabel('dofs')
    plt.loglog(Dofsvect[:MAX_ITER],L2vectQ[:MAX_ITER],linewidth=2.0)
    ax1.legend([ '$||\sigma-\sigma^h||_{L^2(\Omega)}$ '], loc='best')
    data_filename = os.path.join(TrialTypeDir, 'flux_L2err_final.pdf')
    fig.savefig(data_filename, format='pdf', transparent=True)
    plt.close()


    







tstop = time.time()
if rank == 0: print("\n total time [s] :", tstop-tstart)

