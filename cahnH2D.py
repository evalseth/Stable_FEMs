# This FEniCS code solves the 2D stationary Cahn-Hilliard equation using the AVS-FE method
# Last tested January 2021, EV

#Import used modules and useful stuff
from ufl import nabla_div
from sympy import symbols
import sympy as sp
import os
import mshr
import math
import time
import numpy as np
import time
import dolfin
from fenics import *
# from dolfin import *
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

if has_linear_algebra_backend("Epetra"):
        parameters["linear_algebra_backend"] = "Epetra"
        
parameters['ghost_mode']='shared_facet'
        
comm = MPI.comm_world
rank = comm.Get_rank()  # no AttributeError

# Mesh of 2*N*N triangles on the unit square
N = 1
M = UnitSquareMesh(N, N)
Degree = 8
compdeg = 1
MAX_REFS = 6

File_name = 'CH_Uniform_P' + str(compdeg)


#exact hyperbolic with inner layer
Evect = np.zeros(MAX_REFS)
Dofsvect = np.zeros(MAX_REFS)
hvect = np.zeros(MAX_REFS)
EvectRate = np.zeros(MAX_REFS)

TrialTypeDir = os.path.join(File_name, 'Results')
#print(TrialTypeDir)

if not os.path.isdir(TrialTypeDir): os.makedirs(TrialTypeDir)


for level in range(MAX_REFS):
    
    if level == 0:
       tstart = time.time()

    # Define the necessary function spaces:
    # Discontinuous RT space for vector valued test function
    V1 = VectorElement( "DG", M.ufl_cell() ,compdeg)

    # Discontinuous polynomial space for scalar valued test function
    V2 = FiniteElement( "DG", M.ufl_cell() ,compdeg)

    # Discontinuous RT space for vector valued test function
    V3 = VectorElement( "DG", M.ufl_cell() ,compdeg)

    # Discontinuous polynomial space for scalar valued test function
    V4 = FiniteElement( "DG", M.ufl_cell() ,compdeg)

    # Raviart-Thomas space for the flux (vector valued) trial function r
    V5 = FiniteElement( "RT", M.ufl_cell() , compdeg)

    # Continuous polynomial space for the base variable trial function u
    V6 = FiniteElement( "CG", M.ufl_cell() ,compdeg)

    # Raviart-Thomas space for the flux (vector valued) trial function t
    V7 = FiniteElement( "RT", M.ufl_cell() , compdeg)

    # Continuous polynomial space for the q variable trial function
    V8 = FiniteElement( "CG", M.ufl_cell() ,compdeg)

    # The 'total' space is the product of all four functions spaces
    # defined above
    V = FunctionSpace(M, MixedElement(V1,V2,V3,V4,V5,V6,V7,V8))

    # Define the test functions
    s, v, p, w, cc, aa, dd, bb = TestFunctions(V)

    # Define soolution functions
    soln = Function(V)
    ksi, psi, eta, phi, r, u, t, q=split(soln)

    # To compute the inner product we need the element diameter
    h = Circumradius(M)
    hmax = M.hmax()

    # n is the element normal vector needed to compute the boundary
    # integrals
    n = FacetNormal(M)

    # Analytical solution and gradient
    u_ex = Expression('x[0]*x[1]*tanh((x[0]-0.5*x[1]-0.25)/(sqrt(2*0.0625)))',degree = Degree,domain = M)
    gradu = as_vector([u_ex.dx(0),u_ex.dx(1)])
    
    #Problem parameters
    Gamm  = 1/sqrt(0.0625)
    Dee   = 1.0
    
    #Exact auxilliary variable plus projection
    q_ex = u_ex*u_ex*u_ex-u_ex-Gamm*div(gradu)
    gradq = as_vector([q_ex.dx(0),q_ex.dx(1)])
    PC2 = FunctionSpace(M,"CG", Degree)
    q_exx = project(q_ex,PC2)

    # Source term f
    f= Dee*div(gradq)
    
    # Inner product on V of the "optimal" test functions and the error
    # representation functions
    innerp = (   inner(s, ksi) + inner(p,eta)
      + inner(v, psi) + h**2 * inner(grad(v), grad(psi))
      + inner(w, phi) + h**2 * inner(grad(w), grad(phi)) )*dx


    # Differential form acting on the trial functions (u,q,r,t) and the
    # test functions (v,w,s,p)
    buv =( inner(s, grad(u)-r)+ inner(p, grad(q)-t)
        - Dee*inner(grad(w),t)+ inner(u*u*u,v)
        - inner(u,v) + Gamm*inner(r,grad(v))
        - inner(q,v) )*dx +(
    # Integral over the mesh skeleton
        +(  Dee*dot(t('+'),n('+'))*jump(w)  )*dS
        -( Gamm*dot(r('+'),n('+'))*jump(v)  )*dS
    # Integral over the global boundary
        + (Dee*dot(t,n)*w )*ds
        - (Gamm*dot(r,n)*v)*ds )

    # Gateux derivative of thedifferential form acting on the functions
    # (aa,bb,cc,dd) and the error representation functions (psi,phi,ksi,eta)
    brs = ( inner(ksi, grad(aa)-cc) + inner(eta, grad(bb)-dd)
        - Dee*inner(grad(phi),dd)+ inner(3*u*u*aa,psi)
        - inner(aa,psi)+ Gamm*inner(cc, grad(psi))
        - inner(bb,psi) )*dx +(
    # Integral over the mesh skeleton
        +(  Dee*dot(dd('+'),n('+'))*jump(phi)  )*dS
        -( Gamm*dot(cc('+'),n('+'))*jump(psi)  )*dS
    # Integral over the global boundary
        + (Dee*dot(dd,n)*phi )*ds
        - (Gamm*dot(cc,n)*psi)*ds )

    # Define the load functional
    L = 1*inner(w, f)*dx


    # Add all contributions to the LHS
    aaa = buv + brs + innerp - L

    # Dirichlet BCs
    bc = DirichletBC(V.sub(5), u_ex, "on_boundary")
    bc2 = DirichletBC(V.sub(7), q_exx , "on_boundary")


    # Define a solution variable and  print size of the matrix
    sol0 = Function(V)
    print("dofs", sol0.vector().size())
    Ndofs = sol0.vector().size()

    
    # Call nonlinear solver
    solve(aaa == 0, soln,bcs= [bc,bc2], solver_parameters={"newton_solver":{"relative_tolerance": 1e-14, "absolute_tolerance": 1e-14, "maximum_iterations":200, 'linear_solver' : 'mumps'}})

    # Split the solution vector
    ksi0, psi0, eta0, phi0, r0, u0, t0, q0 = soln.split(True)

    # Residual error
    Ee = assemble(( inner(ksi0, ksi0) + inner(eta0,eta0)
      + inner(psi0, psi0) + h**2 * inner(grad(psi0), grad(psi0))
      + inner(phi0, phi0) + h**2 * inner(grad(phi0), grad(phi0))   )*dx)
    E = sqrt(Ee)

    Evect[level] = E
    Dofsvect[level] = Ndofs
    hvect[level] = hmax
    
    # Compute convergence rate
    if level > 0:
        EvectRate[level] = ln(Evect[level]/Evect[level-1])/ln(hvect[level]/hvect[level-1])

    # Refine mesh
    M =refine(M)


    plt.style.use('classic')

    # Plot the solution u
    fig, ax1 = plt.subplots()
    figure=plot(u0)
    fig.colorbar(figure,ax=ax1)
    data_filename = os.path.join(TrialTypeDir, 'soln_%s.pdf'%(level))
    fig.savefig(data_filename, format='pdf', transparent=True)
    plt.close()

    # Plot the mesh
    fig = plt.figure()
    plot(M)
    data_filename = os.path.join(TrialTypeDir, 'mesh_%s.pdf'%(level))
    fig.savefig(data_filename, format='pdf', transparent=True)
    plt.close()
    
    
    
    if ((level > 1)and(rank == 0)):
        print("Energy error ", Evect[level])

    if level > 0 and rank == 0:
        print("energy rate",EvectRate[level])

        
#Plot errors when done refining
fig, ax1  = plt.subplots()
ax1.set_ylabel('Error Norm')
ax1.set_xlabel('dofs')
plt.loglog(Dofsvect[:MAX_REFS],Evect[:MAX_REFS])
ax1.legend(['Energy norm'], loc='best')
data_filename = os.path.join(TrialTypeDir, 'Error_final.pdf')
fig.savefig(data_filename, format='pdf', transparent=True)
plt.close()


tstop = time.time()

for level in range(MAX_REFS):
    if rank == 0:
        print("Dofs",  Dofsvect[level])
        print("Energy error",  Evect[level])

if rank == 0: print("\n total time [s] :", tstop-tstart)

