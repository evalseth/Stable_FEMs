

# This FEniCS code solves the 2D linear elasticity problem for a composite material  using the AVS-FE method and adaptive mesh refinement
# see https://doi.org/10.1002/nme.6743 for details
# Last tested January 2021, EV

from ufl import nabla_div
import os
import mshr
from mshr import *
import numpy as np
import time
import dolfin
from fenics import *
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


if has_linear_algebra_backend("Epetra"):
        parameters["linear_algebra_backend"] = "Epetra"
        
parameters['ghost_mode']='shared_facet'



File_name = 'Incompress'
# Adaptive data
RATIO = 0.25
Degree = 8
MAX_REFS = 10
REFINEMENT = 1
comp_degree = 1

#Youngs modulus and poisson's ratio for incompressible
Youngs = 1500
Poisson = 0.49
#Lame parameters
mu_1 = Youngs/(2*(1+Poisson))
lam_1 = Poisson*Youngs/((1+Poisson)*(1-2*Poisson))

#Youngs modulus and poisson's ratio for stiff material
Youngs_2 = 10000
Poisson_2 = 0.3
#Lame parameters
mu_2 = Youngs_2/(2*(1+Poisson_2))
lam_2 = Poisson_2*Youngs_2/((1+Poisson_2)*(1-2*Poisson_2))


LevelStep = np.zeros(MAX_REFS)
L2vect = np.zeros(MAX_REFS)
L2vectQ = np.zeros(MAX_REFS)
Evect = np.zeros(MAX_REFS)
Dofsvect = np.zeros(MAX_REFS)
hvect = np.zeros(MAX_REFS)


TrialTypeDir = os.path.join(File_name, 'Circular inclusion center adaptive')
#print(TrialTypeDir)

if not os.path.isdir(TrialTypeDir): os.makedirs(TrialTypeDir)



#Define strain and stress
def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
    #return sym(nabla_grad(u))

def sigma(u):
    return lam*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)
    

#Set up domain
domain = Rectangle(Point(0, 0), Point(1.0, 1.0))
inclusion = Circle(Point(0.5, 0.5), 0.1,100)
domain.set_subdomain(1, inclusion)

# Create mesh
M = generate_mesh(domain, 5)

#Define meshfunctions
markers = MeshFunction('size_t', M, 2, M.domains())
boundaries = MeshFunction('size_t', M, 1, M.domains())
interfaces =  MeshFunction('size_t', M, 1, M.domains())

dx = Measure('dx', domain=M, subdomain_data=markers)
ds = Measure('ds', domain=M, subdomain_data=boundaries)
dS = Measure('dS', domain=M, subdomain_data=interfaces)



for level in range(MAX_REFS):
    level_step = level

    
    # Define the necessary function spaces:
    # Discontinuous polynomial space for tensor valued test function
    V1 = TensorElement( "DG", M.ufl_cell() ,comp_degree)
    
    # Discontinuous polynomial space for vector valued test function
    V2 = VectorElement( "DG", M.ufl_cell() ,comp_degree)

    # Raviart-Thomas space for the flux (tensor valued) trial function
    V3 = VectorElement( "RT", M.ufl_cell() ,comp_degree)

    # Continuous polynomial space for the base variable trial function
    V4 = VectorElement( "CG", M.ufl_cell() ,comp_degree)

    # The 'total' space is the product of all four functions spaces
    # defined above
    V = FunctionSpace(M, MixedElement(V1,V2,V3,V4))

    # Define the test and trial functions
    phi, psi, q, u = TrialFunctions(V)
    w, v, s, r = TestFunctions(V)
    # Dummy soln to keep track of primary dofs
    trialV = FunctionSpace(M, MixedElement(V3,V4))
    dummysol = Function(trialV)

    d = u.geometric_dimension()

    
    
    #Change material in circle
    mu = Expression('(x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) <= 0.1*0.1 + 1e-14 ? k_1 : k_2', degree=0,domain = M, k_1=mu_1, k_2=mu_2)
    lam =  Expression('(x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) <= 0.1*0.1 + 1e-14 ? k_1 : k_2', degree=0,domain = M, k_1=lam_1, k_2=lam_2)
   
    print("lam1", lam_1)
    print("lam2", lam_2)
    print("mu1", mu_1)
    print("mu2", mu_2)


    # To compute the inner product we need the element diameter
    h = Circumradius(M)
    hmax = M.hmax()

    # n is the element normal vector needed to compute the boundary
    # integrals
    n = FacetNormal(M)
    x, y = SpatialCoordinate(M)

    #define conditionals
    Gamma_right = conditional(x > 1.0-1e-14, 1,0)
    Gamma_left = conditional(x < 1e-14, 1,0)

    

    # Inner product on V of the  test functions (v,w) and the error
    # representation functions (psi, phi)
    innerp = ( inner(w, phi) + inner(v, psi)  + h**2 * inner(nabla_grad(v),
               nabla_grad(psi)) )*dx

    # Bilinear form acting on the trial functions (u,q) and the
    # optimal test functions (v,w)
    buv = (  inner(w, q - lam*nabla_div(u)*Identity(d) - 2*mu*epsilon(u))
            + inner(nabla_grad(v), q)  )*dx +(
    # Integral over the mesh skeleton
            - (inner(dot(q('+'),n('+')),v('+')-v('-')))*dS
    # Integral over the global boundary
            - (Gamma_left*inner(dot(q,n),v))*ds )

    # Bilinear form acting on the functions (r,s) and the error
    # representation functions (psi, phi)
    brs = (   inner(phi, s-lam*nabla_div(r)*Identity(d) - 2*mu*epsilon(r))
            + inner(nabla_grad(psi), s) )*dx +(
    # Integral over the mesh skeleton
            -(inner(dot(s('+'),n('+')),psi('+')-psi('-')))*dS
    # Integral over the global boundary
            -(Gamma_left*inner(dot(s,n),psi))*ds )

    # Add all contributions to the LHS
    a = buv + brs + innerp
    
    #Only load is traction on right surface
    flux = as_vector([100,0])

    # Define the load functional
    L =  Gamma_right*inner(flux,v)*ds
        
        
        
    # Only BC on a portion of left side
    def leftpoint(x, on_boundary):
        return on_boundary and ((near(x[0], 0, 1e-14)) and (near(x[1], 0.5,0.15)) )
        
    bclin = DirichletBC(V.sub(3),Constant((0.0,0.0)),leftpoint)


    # Define the solution
    sol0 = Function(V)
    print("dofs", dummysol.vector().size())
    print("total dofs", sol0.vector().size())
    print(" num of elems", M.num_cells())

    Ndofs = dummysol.vector().size()


    # Set mumps inputs
    PETScOptions.set("mat_mumps_icntl_14", 1000)
    PETScOptions.set("mat_mumps_icntl_24", 1)
    PETScOptions.set("mat_mumps_icntl_1", 6)
    PETScOptions.set("mat_mumps_icntl_2", 0)
    PETScOptions.set("mat_mumps_icntl_3", 6)
    PETScOptions.set("mat_mumps_icntl_4", 2)

    # Call the solver
    solve(a==L, sol0, bcs=[bclin], solver_parameters = {'linear_solver' : 'mumps'})
    
    # Split the solution vector
    phi0, psi0, q0, u0 = sol0.split(True)
    q11, q22= q0.split(True)

    # compute error indicators from the error representation function
    INDSPACE = FunctionSpace(M,"DG", 0)
    c  = TestFunction(INDSPACE)
    gplot = Function(INDSPACE)
    ge = ( inner(phi0, phi0)*c+ inner(psi0, psi0)*c + h**2 * inner(nabla_grad(psi0), nabla_grad(psi0))*c )*dx
    g = assemble(ge)
    gplot.vector()[:]=g


    # Compute energy error
    Ee = assemble(( inner(phi0, phi0)+ inner(psi0, psi0)+ h**2 * inner(nabla_grad(psi0), nabla_grad(psi0)) )*dx)

    E = sqrt(Ee)


    Evect[level] = E
    Dofsvect[level] = Ndofs
    hvect[level] = hmax
        


    # Plot the solution u
    fig, ax1 = plt.subplots()
    plotting = plot(u0)
    fig.colorbar(plotting,ax=ax1)
    data_filename = os.path.join(TrialTypeDir, 'Displ_2D_%s.pdf'%(level))
    fig.savefig(data_filename, format='pdf', transparent=True)
    plt.close()

    ux,uy = u0.split(True)

    # Plot the solution u_x
    fig, ax1 = plt.subplots()
    plotting = plot(ux)
    fig.colorbar(plotting,ax=ax1)
    data_filename = os.path.join(TrialTypeDir, 'sol_ux_%s.pdf'%(level))
    fig.savefig(data_filename, format='pdf', transparent=True)
    plt.close()

    # Plot the solution u_y
    fig, ax1 = plt.subplots()
    plotting = plot(uy)
    fig.colorbar(plotting,ax=ax1)
    data_filename = os.path.join(TrialTypeDir, 'sol_uy_%s.pdf'%(level))
    fig.savefig(data_filename, format='pdf', transparent=True)
    plt.close()
    
    # Plot the solution xx stress
    fig, ax1 = plt.subplots()
    plotting = plot(q11[0])
    fig.colorbar(plotting,ax=ax1)
    data_filename = os.path.join(TrialTypeDir, 'sol_sig_xx_%s.pdf'%(level))
    fig.savefig(data_filename, format='pdf', transparent=True)
    plt.close()
    
    # Plot the solution xy stress
    fig, ax1 = plt.subplots()
    plotting = plot(q11[1])
    fig.colorbar(plotting,ax=ax1)
    data_filename = os.path.join(TrialTypeDir, 'sol_tau_xy_%s.pdf'%(level))
    fig.savefig(data_filename, format='pdf', transparent=True)
    plt.close()

    # Plot the solution yy stress
    fig, ax1 = plt.subplots()
    plotting = plot(q22[1])
    fig.colorbar(plotting,ax=ax1)
    data_filename = os.path.join(TrialTypeDir, 'sol_2D_sig_yy_RT_%s.pdf'%(level))
    fig.savefig(data_filename, format='pdf', transparent=True)
    plt.close()
    
    # Plot the solution xy stress
    fig, ax1 = plt.subplots()
    plotting = plot(q22[0])
    fig.colorbar(plotting,ax=ax1)
    data_filename = os.path.join(TrialTypeDir, 'sol_2D_tau_yx_RT_%s.pdf'%(level))
    fig.savefig(data_filename, format='pdf', transparent=True)
    plt.close()






    # Plot the mesh
    fig = plt.figure()
    plot(M)
    data_filename = os.path.join(TrialTypeDir, 'mesh_%s.pdf'%(level))
    fig.savefig(data_filename, format='pdf', transparent=True)
    plt.close()
    
    # Plot the local error indicator
    fig, ax1 = plt.subplots()
    plotting = plot(gplot)
    fig.colorbar(plotting,ax=ax1)
    #plotting.set_clim(0., 2.)
    data_filename = os.path.join(TrialTypeDir, 'localerrfunc_%s.pdf'%(level))
    fig.savefig(data_filename, format='pdf', transparent=True)
    plt.close()
    


    # Mark cells for refinement
    cell_markers = MeshFunction("bool", M, M.topology().dim())
    if REFINEMENT == 1:
        rg = np.sort(g)
        rg = rg[::-1]
        rgind = np.argsort(g)
        rgind = rgind[::-1]
        junk = 0.
        g0 = RATIO**2*E**2
        Ntot = M.num_cells()
        for cellm in cells(M):
            if cellm.index() == rgind[0]:
                break

        cell_markers[cellm] = True


        for nj in range(1,Ntot):
            junk += g[rgind[nj]]
            for cellm in cells(M):
                if cellm.index() == rgind[nj]:
                    break
            cell_markers[cellm] = junk < g0
            if junk > g0:
                break


    # Refine and update meshfunctions
    if REFINEMENT ==2:
        M =refine(M)
        markers = MeshFunction('size_t', M, 2, M.domains())
        boundaries = MeshFunction('size_t', M, 1, M.domains())
        interfaces =  MeshFunction('size_t', M, 1, M.domains())
        dx = Measure('dx', domain=M, subdomain_data=markers)
        ds = Measure('ds', domain=M, subdomain_data=boundaries)
        dS = Measure('dS', domain=M, subdomain_data=interfaces)

    if REFINEMENT ==1:
        M  = refine(M,cell_markers)
        markers = MeshFunction('size_t', M, 2, M.domains())
        boundaries = MeshFunction('size_t', M, 1, M.domains())
        interfaces =  MeshFunction('size_t', M, 1, M.domains())
        dx = Measure('dx', domain=M, subdomain_data=markers)
        ds = Measure('ds', domain=M, subdomain_data=boundaries)
        dS = Measure('dS', domain=M, subdomain_data=interfaces)

    

    print("Energy error ", Evect[level])






fig, ax1  = plt.subplots()
ax1.set_ylabel('Error Norm')
ax1.set_xlabel('dofs')
plt.loglog(Dofsvect[:MAX_REFS],Evect[:MAX_REFS],linewidth=2.0)
ax1.legend(['Energy norm'], loc='best')
data_filename = os.path.join(TrialTypeDir, 'Err_norm.pdf')
fig.savefig(data_filename, format='pdf', transparent=True)
plt.close()


# Plot the solution u
fig, ax1 = plt.subplots()
plotting = plot(u0)
fig.colorbar(plotting,ax=ax1)
data_filename = os.path.join(TrialTypeDir, 'U_sol_final.pdf')
fig.savefig(data_filename, format='pdf', transparent=True)
plt.close()

ux,uy = u0.split(True)

# Plot the solution u_x
fig, ax1 = plt.subplots()
plotting = plot(ux)
fig.colorbar(plotting,ax=ax1)
data_filename = os.path.join(TrialTypeDir, 'sol_ux_final.pdf')
fig.savefig(data_filename, format='pdf', transparent=True)
plt.close()

# Plot the solution u_y
fig, ax1 = plt.subplots()
plotting = plot(uy)
fig.colorbar(plotting,ax=ax1)
#plotting.set_clim(0., 2.)
data_filename = os.path.join(TrialTypeDir, 'sol_uy_final.pdf')
fig.savefig(data_filename, format='pdf', transparent=True)
plt.close()

# Plot the solution xx stress
fig, ax1 = plt.subplots()
plotting = plot(q11[0])
fig.colorbar(plotting,ax=ax1)
data_filename = os.path.join(TrialTypeDir, 'sol_sig_xx_final.pdf')
fig.savefig(data_filename, format='pdf', transparent=True)
plt.close()

# Plot the solution xy stress
fig, ax1 = plt.subplots()
plotting = plot(q11[1])
fig.colorbar(plotting,ax=ax1)
data_filename = os.path.join(TrialTypeDir, 'sol_tau_xy_final.pdf')
fig.savefig(data_filename, format='pdf', transparent=True)
plt.close()

# Plot the solution yy stress
fig, ax1 = plt.subplots()
plotting = plot(q22[1])
fig.colorbar(plotting,ax=ax1)
data_filename = os.path.join(TrialTypeDir, 'sol_sig_yy_final.pdf')
fig.savefig(data_filename, format='pdf', transparent=True)
plt.close()

# Plot the solution xy stress
fig, ax1 = plt.subplots()
plotting = plot(q22[0]) # if RT
fig.colorbar(plotting,ax=ax1)
data_filename = os.path.join(TrialTypeDir, 'sol_tau_yx_final.pdf')
fig.savefig(data_filename, format='pdf', transparent=True)
plt.close()
