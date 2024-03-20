"""This code gives a numerical approximation of the stochastic PDE given by 
the Kuramoto–Sivashinsky equation: u_t + u_xx + u_xxxx + 1/2 * u_x^2 = 0,
with space-time white noise. 
We use Firedrake with Hermite elements of degree 3.
"""

from firedrake import *
import matplotlib.pyplot as plt
import os

#from firedrake_adjoint import *
#import numpy as np

n = 1000
dt = 0.1
nu = 0.5


mesh = PeriodicIntervalMesh(n, 40.0)
x, = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "HER", 3)

V_interpolate = FunctionSpace(mesh, 'CG', 3)
u_obs = Function(V_interpolate)

#w at time n-1
#for the finite difference approximation of the time derivative
w0 = Function(V)

#initial condition
w0.project(0.2*2/(exp(x-403./15.) + exp(-x+403./15.))
               + 0.5*2/(exp(x-203./15.)+exp(-x+203./15.)))

#test function for the variational form
phi = TestFunction(V)

#w at time n
w1 = Function(V)
w1.assign(w0)

#set the space-time noise
V_ = FunctionSpace(mesh, "DG", 0)
U = Function(V_)
dW = dt**(1/2)*U

#use backward Euler scheme for the variational form with space-time noise
L = ((w1-w0) * phi + dt*(w1.dx(0)).dx(0)* \
    (phi.dx(0)).dx(0) - dt*w1.dx(0)* phi.dx(0) \
    -dt*0.5 * w1*w1*phi.dx(0) + (dW*phi)) * dx

#define a problem and solver over which we will iterate in a loop
uprob = NonlinearVariationalProblem(L, w1)
usolver = NonlinearVariationalSolver(uprob, solver_parameters=
   {'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu'})


# PCG64 random number generator
pcg = PCG64(seed=123456789)
rg = RandomGenerator(pcg)
#normal distribution
amplitude = Constant(0.05)
fx = Function(V_)
#divide coeffs by area of each cell to get w
w = fx / (CellVolume(mesh)**0.5)
#we will approximate dW with w*dx
#now calculate Matern field by solving the PDE with variational form
#a(u, v) = nu * <v, dW>
#where a is the variational form of the operator M[u] = u + k^-2 * u_xx
k = Constant(1.0)

v = TestFunction(V_)
L_ = (U * v + k**(-2) * U.dx(0) * v.dx(0) - nu * v * w) * dx
#solve problem and store it on u
noiseprob = NonlinearVariationalProblem(L_, U)
noisesolver = NonlinearVariationalSolver(noiseprob, solver_parameters=
   {'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu'})

# Define parameters
T = 50.0
t = 0.0

# Create output directory 
output_directory = "output_pvd"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

output_file = File(os.path.join(output_directory, "output.pvd"))

# Solve the SPDE
while t < T - 0.5 * dt:
    t += dt

    # Solve the SPDE at time t
    usolver.solve()
    w0.assign(w1)

    # Write the solution to each output file
    output_file.write(w1, time=t)



