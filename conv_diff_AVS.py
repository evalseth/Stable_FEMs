# This FEniCS code solves the 2D convection-diffusion equation using the AVS-FE method
# Last tested January 2021, EV
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


parameters['ghost_mode']='shared_facet'
comm = MPI.comm_world
rank = comm.Get_rank()

M = UnitSquareMesh(1, 1)

File_name = 'Figures'
MM = 500
Degree = 8
MAX_REFS = 6


Evect_AVS = np.zeros(MAX_REFS)
Dofsvect_primal = np.zeros(MAX_REFS)
hvect_AVS = np.zeros(MAX_REFS)
EvectRate_AVS = np.zeros(MAX_REFS)
L2vect_AVS = np.zeros(MAX_REFS)
L2vectRate_AVS = np.zeros(MAX_REFS)
H1vect_AVS = np.zeros(MAX_REFS)
H1vectRate_AVS = np.zeros(MAX_REFS)

pprimal = 2
p_add = 0
testdeg = pprimal + p_add
epsilon = 0.1

TrialTypeDir = os.path.join(File_name, 'Peclet_100_exact_DPG_Ptrial' + str(pprimal)+ '_Ptest'+ str(testdeg))


if not os.path.isdir(TrialTypeDir): os.makedirs(TrialTypeDir)


for level in range(MAX_REFS):

    # Define the necessary function spaces: 
    # Discontinuous RT space for vector valued test function
    V1 = FiniteElement( "DRT", M.ufl_cell() ,pprimal+p_add)

    # Discontinuous polynomial space for scalar valued test function
    V2 = FiniteElement( "DG", M.ufl_cell() ,pprimal+p_add)

    # Raviart-Thomas space for the flux vector valued trial function
    V3 = FiniteElement( "RT", M.ufl_cell() ,pprimal)

    # Continuous polynomial space for the base variable trial function
    V4 = FiniteElement( "CG", M.ufl_cell() ,pprimal) 

    # The 'total' space is the product of all four functions spaces
    V = FunctionSpace(M, MixedElement(V1,V2,V3,V4))
    Vdum = FunctionSpace(M, MixedElement(V3,V4))

    # Define the test and trial functions
    phi, psi, q, u = TrialFunctions(V)
    w, v, s, r = TestFunctions(V)

    # To compute the inner product we need the element diameter 
    h = Circumradius(M)
    hmax = M.hmax()
    n = FacetNormal(M)

    # Analytical solution
    uexact = Expression('(x[0]+(expl(1*x[0]/epsilon)-1)/(1-expl(1/epsilon)))*(x[1]+(expl(1*x[1]/epsilon)-1)/(1-expl(1/epsilon)))',degree = Degree,domain = M,epsilon=epsilon)
    
    b = Constant((1,1))
    
    # Source term f
    f = -epsilon*div(nabla_grad(uexact)) + dot(nabla_grad(uexact),b) 


    # Inner product on V of the test functions (v,w) and the error
    # representation functions (psi, phi)
    innerp = ( inner(w, phi) + inner(v, psi)  + h**2 * inner(nabla_grad(v),nabla_grad(psi)) + h**2 * inner(div(w),div(phi)) )*dx

    # Bilinear form acting on the trial functions (u,q) and the
    # optimal test functions (v,w)
    buv = (  inner(w, q - epsilon*grad(u))
            - inner(v, div(q))
            + inner(v, dot(b, grad(u)))  )*dx


    # Bilinear form acting on the functions (r,s) and the error
    # representation functions (psi, phi)
    brs = (   inner(phi, s-epsilon*grad(r))
            - inner((psi), div(s))
            + inner(psi, dot(b, grad(r)))  )*dx
            
    # Add all contributions to the LHS
    a = buv + brs + innerp

    # Define the load functional
    L = 1*inner(v, f)*dx

    #BCs
    def Xzero(x, on_boundary):
        return on_boundary and (near(x[0],0, 1e-14))
    def Xzone(x, on_boundary):
        return on_boundary and (near(x[0],1, 1e-14))
    def Yzero(x, on_boundary):
        return on_boundary and (near(x[1],0, 1e-14))
    def Yone(x, on_boundary):
        return on_boundary and (near(x[1],1, 1e-14))
        
    # Dirichlet BC is applied only on the space V4 ( on trial function u )
    bc1 = DirichletBC(V.sub(3), Constant(0.0), Xzero)
    # Dirichlet BC is applied only on the space V4 ( on trial function u )
    bc2 = DirichletBC(V.sub(3), Constant(0.0),Xzone)
    # Dirichlet BC is applied only on the space V4 ( on trial function u )
    bc3 = DirichletBC(V.sub(3), Constant(0.0),Yzero)
    # Dirichlet BC is applied only on the space V4 ( on trial function u )
    bc4 = DirichletBC(V.sub(3), Constant(0.0),Yone)


    # Define the solution
    sol0 = Function(V)
    print("dofs", sol0.vector().size())
    
    soldum = Function(Vdum)
    Ndofs_primal = soldum.vector().size()
    print("Primal dofs", soldum.vector().size())
    Dofsvect_primal[level] = Ndofs_primal
    
    # Set mumps solver parameter
    PETScOptions.set("mat_mumps_icntl_14", 50.0)

    # Call the solver
    solve(a==L, sol0, bcs=[bc1,bc2,bc3,bc4], solver_parameters = {'linear_solver' : 'mumps'})

    # Split the solution vector
    phi00, psi00, q0, u0 = sol0.split(True)
    
    #Compute error
    e0r = u0 - uexact
    L2r = assemble(inner(e0r,e0r)*dx) 
    h1r = assemble(inner(e0r,e0r)*dx + inner(grad(e0r),grad(e0r))*dx)


    # Energy error/ residual
    Ee = assemble(( inner(phi00, phi00)+ inner(psi00, psi00) + h**2 * inner(grad(psi00), grad(psi00))+ h**2 * inner(div(phi00), div(phi00))   )*dx)
    
    #Norms
    E = sqrt(Ee)
    L2e = sqrt(L2r)
    H1e = sqrt(h1r)

    

    L2vect_AVS[level] = L2e
    H1vect_AVS[level] = H1e
    Evect_AVS[level] = E
    Dofsvect_primal[level] = Ndofs_primal
    hvect_AVS[level] = hmax
    
    if level>1:
       L2vectRate_AVS[level] = ln(L2vect_AVS[level]/L2vect_AVS[level-1])/ln(hvect_AVS[level]/hvect_AVS[level-1])
       EvectRate_AVS[level] = ln(Evect_AVS[level]/Evect_AVS[level-1])/ln(hvect_AVS[level]/hvect_AVS[level-1])
       H1vectRate_AVS[level] = ln(H1vect_AVS[level]/H1vect_AVS[level-1])/ln(hvect_AVS[level]/hvect_AVS[level-1])

    
    if ((level >= 0)and(rank == 0)):
        print("h1 error u", H1vect_AVS[level])
        print("l2 error u", L2vect_AVS[level])
        print("Energy error", Evect_AVS[level])
        
    # uniform refinement
    M = refine(M)

    plt.style.use('classic')

    # Plot the solution u
    fig, ax1 = plt.subplots()
    plotU = plot(u0)
    fig.colorbar(plotU,ax=ax1)
    data_filename = os.path.join(TrialTypeDir, '2D_soln_%s.pdf'%(level))
    fig.savefig(data_filename, format='pdf', transparent=True)
    plt.close()


    

fig, ax1  = plt.subplots()
ax1.set_ylabel('Error norm')
ax1.set_xlabel('dofs')
plt.loglog(Dofsvect_primal[:MAX_REFS],L2vect_AVS[:MAX_REFS],linewidth=2.0)
ax1.legend(['$||u-u^h||_{L^2(\Omega)}$'], loc='best')
data_filename = os.path.join(TrialTypeDir, 'L2err_final.pdf')
fig.savefig(data_filename, format='pdf', transparent=True)
plt.close()


    

fig, ax1  = plt.subplots()
ax1.set_ylabel('Error norm')
ax1.set_xlabel('dofs')
plt.loglog(Dofsvect_primal[:MAX_REFS],L2vect_AVS[:MAX_REFS],linewidth=2.0)
plt.loglog(Dofsvect_primal[:MAX_REFS],Evect_AVS[:MAX_REFS],linewidth=2.0)
ax1.legend(['$||u-u^h||_{L^2(\Omega)}$', 'Energy norm'], loc='best')
data_filename = os.path.join(TrialTypeDir, 'Errors_final.pdf')
fig.savefig(data_filename, format='pdf', transparent=True)
plt.close()



    
if rank == 0 :
    for level in range(MAX_REFS):
        print("dofs  ",  Dofsvect_primal[level])
        print("L2 error u   ",  L2vect_AVS[level])
        print("H1 error u   ",  H1vect_AVS[level])
        print("Energy error ",  Evect_AVS[level])

        if level == MAX_REFS-1:
            print("final rate 'L2u'  ",  L2vectRate_AVS[level])
            print("final rate 'h1u'  ",  H1vectRate_AVS[level])
            print("final energy rate'   ",  EvectRate_AVS[level])



