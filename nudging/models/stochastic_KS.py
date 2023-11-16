from firedrake import *
from firedrake_adjoint import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from nudging.model import *
import numpy as np
from operator import mul
from functools import reduce

class KS(base_model):
    def __init__(self, n, nsteps, xpoints, lambdas=False, dt = 0.025, seed=12353):
        self.n = n
        self.nsteps = nsteps
        self.dt = dt
        self.seed = seed
        self.xpoints = xpoints
        self.lambdas = lambdas # include lambdas in allocate

    def setup(self, comm = MPI.COMM_WORLD):
        self.mesh = PeriodicIntervalMesh(self.n, 40.0, comm = comm)
        x, = SpatialCoordinate(self.mesh)
        V = FunctionSpace(self.mesh, "HER", 3)
        #w at time n-1
        #for the finite difference approximation of the time derivative
        self.w0 = Function(self.V)
        self.x = SpatialCoordinate(self.mesh)

        #initial condition
        w0.project(0.2*2/(exp(self.x-403./15.) + exp(-self.x+403./15.))
                       + 0.5*2/(exp(self.x-203./15.)+exp(-self.x+203./15.)))

        #test function for the variational form
        self.phi = TestFunction(self.V)

        #w at time n
        self.w1 = Function(self.V)
        self.w1.assign(self.w0)

        #set the space-time noise
        self.V_ = FunctionSpace(self.mesh, "DG", 0)
        self.U = Function(self.V_)
        self.dW = self.dt**(1/2)*self.U

        #use backward Euler scheme for the variational form with space-time noise
        L = ((self.w1-self.w0)/self.dt * self.phi + (self.w1.dx(0)).dx(0)*(self.phi.dx(0)).dx(0) - self.w1.dx(0)* self.phi.dx(0) -0.5 * self.w1*self.w1*self.phi.dx(0) + (self.W*self.phi)) * dx

        #define a problem and solver over which we will iterate in a loop
        uprob = NonlinearVariationalProblem(L, self.w1)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=
           {'mat_type': 'aij',
            'ksp_type': 'preonly',
            'pc_type': 'lu'})

        # state for controls
        self.X = self.allocate()

        # vertex only mesh for observations
        #x_obs =np.linspace(0, 40,num=self.xpoints, endpoint=False) # This is better choice
        x_obs = np.arange(0.5,self.xpoints)
        x_obs_list = []
        for i in x_obs:
            x_obs_list.append([i])
        self.VOM = VertexOnlyMesh(self.mesh, x_obs_list)
        self.VVOM = FunctionSpace(self.VOM, "DG", 0)

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

    def run(self, X0, X1, operation = None):
        # copy input into model variables for taping
        for i in range(len(X0)):
            self.X[i].assign(X0[i])

        # copy initial condition into model variable
        self.w0.assign(self.X[0])

        # do the timestepping
        for step in range(self.nsteps):
            # get noise (noise variables and lambdas TBD)
            self.dW.assign(self.X[step+1])
            if self.lambdas:
                self.dW += self.X[step+1+self.nsteps]*(self.dt)**0.5
            # advance in time
            self.usolver.solve()
            # copy output to input
            self.w0.assign(self.w1)

        # exact callbacks
        if operation:
            operation(self.w0)

        # return outputs
        X1[0].assign(self.w0) # save sol at the nstep th time

    def controls(self):
        controls_list = []
        for i in range(len(self.X)):
            controls_list.append(Control(self.X[i]))
        return controls_list

    def obs(self):
        m, u = self.w0.split()
        Y = Function(self.VVOM)
        Y.interpolate(u)
        return Y

    def allocate(self):
        particle = [Function(self.W)]
        for i in range(self.nsteps):
            dW = self.rg.normal(self.noise_space, 0., 1.0)
            particle.append(dW)
        return particle

    def randomize(self, X, c1=0, c2=1, gscale=None, g=None):
        rg = self.rg
        count = 0
        for i in range(self.nsteps):
            count += 1
            X[count].assign(c1*X[count] + c2*rg.normal(
                self.noise_space, 0., 1.0))
            if g:
                X[count] += gscale*g[count]


#    def lambda_functional(self):
#        nsteps = self.nsteps
#        dt = self.dt
#        for step in range(nsteps):
#            lambda_step = self.X[nsteps + 1 + step]
#            dW_step = self.X[1 + step]
#            for cpt in range(self.n_noise_cpts):
#                dlfunc = assemble(
#                    lambda_step.sub(cpt)**2*dt/2*dx
#                    - lambda_step.sub(cpt)*dW_step.sub(cpt)*dt**0.5*dx
#                )
#                dlfunc /= self.Area
#                if step == 0 and cpt == 0:
#                    lfunc = dlfunc
#                else:
#                    lfunc += dlfunc
#        return lfunc
