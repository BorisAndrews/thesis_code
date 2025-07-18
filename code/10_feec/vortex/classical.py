'''
Imports
'''
from firedrake import *
import avfet_modules.cheb_fet as cheb_fet
import gc



'''
Parameters
'''
# Vortex setup
layers = 10  # Layers in sum for Weierstrass approximation
vortex = (0.381966, 0.763932)  # Vortex location

# Spatial discretisation
nx = 2**5  # Mesh number
k = 3  # Spatial degree (Must be >=3, I think)
sigma = Constant(2**5)  # IP parameter

# Temporal discretisation
s = 1  # Stages (Must be >=1)
timestep = Constant(2**(-10))
duration = 2**4

# Setting
# Re_arr = [2**i for i in range(0, 1)]
Re_arr = [2**24]

# Other
# save_int = round(2**-8 / float(timestep))  # How regularly to save output
save_int = 16 * 4  # How regularly to save output



'''
General purpose functions
'''
# Parallelised "print"
print_ = print
def print(x):
    if mesh.comm.rank == 0:
        print_(x, flush = True)



'''
Vortex definition
'''
# Layer summands
def layer_summands(x, y, summand_partial, layers=10):
    summand = lambda x, y, m, n : summand_partial(x, y, m, n) - summand_partial(0, 0, m, n)  # Summand with normalisation
    out = summand_partial(x, y, 0, 0)
    for m in range(-layers, layers+1):
        for n in range(-layers, layers+1):
            if m!=0 or n!=0:
                out += summand(x, y, m, n)
    return out

# Real and imaginary parts
denominator = lambda x, y, m, n : ((x - 2*m)**2 + (y - 2*n)**2)**2  # General purpose denominator
real_part = lambda x, y : layer_summands(
    x, y,
    lambda x, y, m, n : ((x - 2*m)**2 - (y - 2*n)**2) / denominator(x, y, m, n),
    layers
)
imag_part = lambda x, y : layer_summands(
    x, y,
    lambda x, y, m, n : - 2 * (x - 2*m) * (y - 2*n) / denominator(x, y, m, n),
    layers
)

# Stream function
ln_func = lambda x, y, X, Y : 0.5 * ln(  # Taking ln of absolute value
    (real_part(x, y) - real_part(X, Y))**2
  + (imag_part(x, y) - imag_part(X, Y))**2
)
stream_func = lambda x, y, X, Y : ln_func(x, y, X, Y) - ln_func(x, y, X, -Y)  # Removing complex conjugate

# Tidy up BCs using reflections
def stream_func_tidy(x, y, X, Y):
    return conditional(
        le(x, 0.5),
        conditional(
            le(y, 0.5),
            stream_func(  x,   y,   X,   Y),
            stream_func(  x, 1-y,   X, 1-Y)
        ),
        conditional(
            le(y, 0.5),
            stream_func(1-x,   y, 1-X,   Y),
            stream_func(1-x, 1-y, 1-X, 1-Y)
        )
    )



'''
Mesh (and properties)
'''
# Create mesh
mesh = UnitSquareMesh(nx, nx)

# Get properties
(x, y) = SpatialCoordinate(mesh)  # Cartesian coordinates
n = FacetNormal(mesh)  # Facet normal
h = CellDiameter(mesh)  # Cell diameter



'''
Function spaces
'''
# Individual (spatial) spaces
S_ = FunctionSpace(mesh, "CG", k)

# Individual (space-time) spaces
S = cheb_fet.FETFunctionSpace(mesh, "CG", k, s-1)

# Print number of degrees of freedom
print(RED % f"Degrees of freedom: {S.dim()} {[S__.dim() for S__ in S]}")




'''
Functions
'''
# Trial functions (Space-time)
s_ufl = Function(S)
(s_t,) = cheb_fet.FETsplit(s_ufl)

# Test functions (Space-time)
v_ufl = TestFunction(S)
(v,) = cheb_fet.FETsplit(v_ufl)

# Persistent value tracker (Spatial only)
s_ = Function(S_)

# Integated trial functions (Space-time)
s       = cheb_fet.integrate(s_t, s_, timestep)
s_tilde = cheb_fet.project(s)



'''
Set up ICs
'''
# Functions
s_ref = Function(S_)
v_ref = TestFunction(S_)

# Residual
ip = sigma * k**2 / avg(h)  # Interior penalty constant
F = (
    (s_ref - stream_func_tidy(x, y, vortex[0], vortex[1])) * v_ref * dx
  + ip * avg(inner(grad(s_ref), n)) * avg(inner(grad(v_ref), n)) * dS
)

# Solve
solve(F==0, s_ref, bcs=DirichletBC(S_, 0, "on_boundary"))

# Normalise
energy = assemble(1/2 * inner(grad(s_ref), grad(s_ref))*dx)
s_ref.assign(s_ref / sqrt(energy))



'''
Residual definition
'''
# Default Reynolds no.
Re = Constant(1)

# Broken inner product
Hdiv0 = lambda a, b : (
    div(a) * div(b) * dx
  - avg(inner(a, n)) * avg(div(b)) * dS
  - avg(div(a)) * avg(inner(b, n)) * dS
  + ip * avg(inner(a, n)) * avg(inner(b, n)) * dS
)

# Cross product
cross = lambda a, b : a[0] * b[1] - a[1] * b[0]

# Initialise residual
F = 0

# Stream function equation
F = cheb_fet.residual(
    F,
    lambda a, b : inner(grad(a), grad(b)) * dx,
    (s_t, v)
)
F = cheb_fet.residual(
    F,
    lambda a, b, c : (
      - div(grad(a)) * cross(grad(b), grad(c)) * dx
      + 2*avg(inner(grad(a), n)) * cross(avg(grad(b)), avg(grad(c))) * dS
    ),
    (s_tilde, s_tilde, v)
)
# F = cheb_fet.residual(
#     F,
#     lambda a, b : 1/Re * Hdiv0(grad(a), grad(b)),
#     (s_tilde, v)
# )




'''
Solver parameters
'''
sp = {
    # # Outer (nonlinear) solver
    # "snes_atol": 1.0e-11,
    # "snes_rtol": 1.0e-11,

    # "snes_converged_reason"     : None,
    # "snes_linesearch_monitor"   : None,
    # "snes_monitor"              : None,

    # # Inner (linear) solver
    # "ksp_type"                  : "preonly",  # Krylov subspace = GMRes
    # "pc_type"                   : "lu",
    # "pc_factor_mat_solver_type" : "mumps",
    # # "ksp_atol"                  : 1e-8,
    # # "ksp_rtol"                  : 1e-8,
    # # "ksp_max_it"                : 100,

    # # "ksp_monitor" : None,
    # # "ksp_converged_reason" : None,
    # "ksp_monitor_true_residual" : None,
}



'''
Write text outputs
'''
# Print and write QoIs
def print_write_qoi(qoi_name, qoi_file, qoi_operator, write_type):
    qoi = qoi_operator(s_)
    print(GREEN % f"{qoi_name}: {qoi}")
    if mesh.comm.rank == 0:
        open("output/10_feec/classical/" + qoi_file + ".txt", write_type).write(str(qoi) + "\n")

qois = [
    {"Name": "Energy",              "File": "energy",              "Operator": lambda s_ref : assemble(1/2 * inner(grad(s_), grad(s_)) * dx)},
    {"Name": "Broken enstrophy",    "File": "broken_enstrophy",    "Operator": lambda s_ref : assemble(1/2 * Hdiv0(grad(s_), grad(s_)))},
    {"Name": "Internal enstrophy",  "File": "internal_enstrophy",  "Operator": lambda s_ref : assemble(1/2 * inner(div(grad(s_)), div(grad(s_))) * dx)}
]

def print_write(write_type):
    for qoi in qois:
        print_write_qoi(qoi["Name"], qoi["File"], qoi["Operator"], write_type)



'''
Solve loop
'''
for (i, Re_) in enumerate(Re_arr):
    '''
    Initialisation
    '''
    # Print Reynolds no.
    print(RED % f"Solving for Re = {Re_}:")

    # Reset
    Re.assign(Re_)
    s_.assign(s_ref)



    '''
    Output setup
    '''
    # Set up Paraview
    pvd = VTKFile("output/10_feec/classical/solution.pvd")
    s_.rename("Stream function")
    pvd.write(s_)

    # Print and write initial QoIs
    print_write("w")



    '''
    Solve
    '''
    time = 0.0
    i = 0
    while (time < duration - float(timestep)/2):
        # Print timestep
        print(RED % f"Solving for t = {float(time) + float(timestep)}:")

        # Solve
        solve(F==0, s_ufl, bcs=[DirichletBC(S__, 0, "on_boundary") for S__ in S], solver_parameters=sp)

        # Collect garbage
        gc.collect()

        # Update stream function
        s_.assign(cheb_fet.FETeval((s_, None), (s_ufl, None), timestep, timestep))

        # Write to Paraview (at intervals to save on memory!)
        i += 1
        if i % save_int == 0:
            pvd.write(s_)

        # Write text outputs
        print_write("a")

        # Increment time
        time += float(timestep)
