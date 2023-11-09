from firedrake import *
from firedrake_adjoint import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from nudging.model import *
import numpy as np

class KS_SD(base_model):
    def __init__(self, n, nsteps, dt = 0.01, seed=12353):

        self.n = n
        self.nsteps = nsteps
        self.dt = dt
        self.seed = seed

    def setup(self, comm = MPI.COMM_WORLD):

        #define the mesh and function space with Hermite elements of degree 3
        self.mesh = PeriodicIntervalMesh(self.n, 40.0)
        self.V = FunctionSpace(self.mesh, "HER", 3)
        #dt = 0.1

        #u at time n-1
        #for the finite difference approximation of the time derivative
        self.u0 = Function(self.V)
        self.x = SpatialCoordinate(self.mesh)

        #initial condition
        u0.project(0.2*2/(exp(self.x-403./15.) + exp(-self.x+403./15.))
                       + 0.5*2/(exp(self.x-203./15.)+exp(-self.x+203./15.)))

        #test function for the variational form
        self.phi = TestFunction(self.V)

        #u at time n
        self.u1 = Function(self.V)
        self.u1.assign(self.u0)

        #set the space-time noise
        self.V_ = FunctionSpace(self.mesh, "DG", 0)
        self.U = Function(self.V_)

        #use backward Euler scheme for the variational form with space-time noise
        L = ((self.u1-self.u0)/self.dt * self.phi + (self.u1.dx(0)).dx(0)*(self.phi.dx(0)).dx(0) - self.u1.dx(0)* self.phi.dx(0) -0.5 * self.u1*self.u1*self.phi.dx(0) + (self.dt**(1/2)*self.U*self.phi)) * dx

        #define a problem and solver over which we will iterate in a loop
        uprob = NonlinearVariationalProblem(L, self.u1)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=
           {'mat_type': 'aij',
            'ksp_type': 'preonly',
            'pc_type': 'lu'})

        # state for controls
        self.X = self.allocate()

    def noise():
        # PCG64 random number generator
        pcg = PCG64(seed=123456789)
        rg = RandomGenerator(pcg)
        #normal distribution
        amplitude = Constant(0.05)
        fx = rg.normal(V, 0.0, amplitude)
        #divide coeffs by area of each cell to get w
        w = fx * 1000/40
        #we will approximate dW with w*dx
        #now calculate Matern field by solving the PDE with variational form
        #a(u, v) = nu * <v, dW>
        #where a is the variational form of the operator M[u] = u + k^-2 * u_xx
        k = Constant(1.0)
        nu = Constant(1.0)
        self.v = TestFunction(self.V_)
        L_ = (self.U * self.v + k**(-2) * self.U.dx(0) * self.v.dx(0) - nu * self.v * w) * dx
        #solve problem and store it on u
        noiseprob = NonlinearVariationalProblem(L_, self.U)
        noisesolver = NonlinearVariationalSolver(noiseprob, solver_parameters=
           {'mat_type': 'aij',
            'ksp_type': 'preonly',
            'pc_type': 'lu'})
        noisesolver.solve()
        #project the noise value into Lagrange elements of degree 3
        #Hermite elements contained as a subspace
        #LagrangeSpace = FunctionSpace(mesh, "DG", 2)
        #U_projected = project(U, LagrangeSpace)
        #U.assign(U_projected)

    def run(self, X0, X1):
        for i in range(len(X0)):
            self.X[i].assign(X0[i])
        #solve the SPDE
        while (t < T - 0.5 * dt):
            t += dt
            #calculate noise U
            noise()
            #solve the SPDE at time t
            self.usolver.solve()
            self.u0.assign(self.u1)

        self.u0.assign(self.X[0])

        X1[0].assign(self.u0) # save sol at the nstep th time

    def controls(self):
        controls_list = []
        for i in range(len(self.X)):
            controls_list.append(Control(self.X[i]))
        return controls_list


    def obs(self):
        self.u1.assign(self.u0)
        self.usolver.solve()
        return self.u1


#    def allocate(self):
#        particle = [Function(self.V)]
#        for i in range(self.nsteps):
#            dW_star = Function(self.V_)
#            particle.append(dW_star)
#s        return particle



    def randomize(self, X):
        rg = self.rg
        count = 0
        for i in range(self.nsteps):
               self.dXi.assign(rg.normal(self.V, 0., 1.0))
               self.dW_solver.solve()
               count += 1
               X[count].assign(c1*X[count] + c2*self.dW_n)
               if g:
                    X[count] += gscale*g[count]
