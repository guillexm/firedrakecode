"""This code gives a numerical approximation of the stochastic PDE given by 
the Kuramoto–Sivashinsky equation: u_t + u_xx + u_xxxx + 1/2 * u_x^2 = 0,
with space-time white noise. 
We use Firedrake with Hermite elements of degree 3.
"""

from firedrake import *
import matplotlib.pyplot as plt
import os

#define the mesh and function space with Hermite elements of degree 3
n = 1000
mesh = PeriodicIntervalMesh(n, 40.0)
V = FunctionSpace(mesh, "HER", 3)
dt = 0.1

#u at time n-1 
#for the finite difference approximation of the time derivative
u0 = Function(V)
x, = SpatialCoordinate(mesh)

#initial condition
u0.project(0.2*2/(exp(x-403./15.) + exp(-x+403./15.))
               + 0.5*2/(exp(x-203./15.)+exp(-x+203./15.)))

#test function for the variational form
phi = TestFunction(V)

#u at time n
u1 = Function(V)
u1.assign(u0)

#set the space-time noise
V_ = FunctionSpace(mesh, "DG", 0)
k = 1
nu = 1
U = Function(V_)
v = TestFunction(V_)
amplitude = 0.05

def noise():
    # PCG64 random number generator
    pcg = PCG64(seed=123456789)
    rg = RandomGenerator(pcg)
    #normal distribution
    fx = rg.normal(V, 0.0, amplitude)
    #divide coeffs by area of each cell to get w
    w = fx * 1000/40
    #we will approximate dW with w*dx
    #now calculate Matern field by solving the PDE with variational form
    #a(u, v) = nu * <v, dW>
    #where a is the variational form of the operator M[u] = u_x + k^-2 * u_xx
    L_ = (U * v + k**(-2) * U.dx(0) * v.dx(0) - nu * v * w) * dx
    #solve problem and store it on u
    noiseprob = NonlinearVariationalProblem(L_, U)
    noisesolver = NonlinearVariationalSolver(noiseprob, solver_parameters=
       {'mat_type': 'aij',
        'ksp_type': 'preonly',
        'pc_type': 'lu'})
    noisesolver.solve()

 

#use backward Euler scheme for the variational form with space-time noise
L = ((u1-u0)/dt * phi + (u1.dx(0)).dx(0)*(phi.dx(0)).dx(0) - u1.dx(0)* phi.dx(0) -0.5 * u1*u1*phi.dx(0) + (dt**(1/2)*U*phi)) * dx

#define a problem and solver over which we will iterate in a loop
uprob = NonlinearVariationalProblem(L, u1)
usolver = NonlinearVariationalSolver(uprob, solver_parameters=
   {'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu'})

#writing the output
T = 300.0
t = 0.0
#ufile = File('u.pvd')
#ufile.write(u1, time=t)

image_directory = "output_images"
if not os.path.exists(image_directory):
    os.makedirs(image_directory)

#solve the SPDE
frame_count = 0
iteration = 0
while (t < T - 0.5 * dt):
    t += dt

    #calculate noise U
    noise()

    #solve the SPDE at time t
    usolver.solve()
    u0.assign(u1)
    
    if iteration % 5 == 0:
        #plot and save the figure as an image
        fig, axes = plt.subplots()
        x_values = mesh.coordinates.dat.data_ro
        u_values = u1.dat.data_ro
        axes.plot(x_values, u_values)
        plt.xlabel('x')
        plt.ylabel('u')
        plt.title('Solution at t={}'.format(t))
        
        #save the figure as an image
        plt.savefig(os.path.join(image_directory, f"frame_{frame_count:04d}.png"))
        plt.close(fig)
        frame_count += 1
    iteration += 1

#use FFmpeg to convert the images to an OGV video
video_name = "output005.ogv"
os.system(f"ffmpeg -r 10 -i {image_directory}/frame_%04d.png -c:v libtheora -vf scale=640:480 {video_name}")

#remove the image directory
import shutil
shutil.rmtree(image_directory)



