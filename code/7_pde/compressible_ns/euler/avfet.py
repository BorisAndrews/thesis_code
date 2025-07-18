'''
Imports
'''
from firedrake import *
import avfet_modules.cheb_fet as cheb_fet
import gc
import numpy as np



'''
General
'''
# Parallelise "print"
_print = print  # Save old serial "print"
def print(x):  # Create new parallel "print"
    if mesh.comm.rank == 0:
        _print(x, flush = True)

# Increase spatial quadrature degree
dx = dx(degree=100)



'''
Parameters
'''
# Model
#   Setting
# M = Constant(1 / np.sqrt(2))  # Mach number
# Re = Constant(32)  # Reynolds number

#  Constitutive relations (Set to standard for ideal fluid)
CV = Constant(2.50)  # Specific heat capacity (Material-dependent, ~2.50 standard for air)

theta = lambda rho, eps : eps / rho / CV  # Non-AV theta (temperature)
p     = lambda rho, eps : 1/CV * eps  # Non-AV p (pressure)
s     = lambda rho, eps : rho * (ln(theta(rho, eps)**CV / rho))  # Non-AV s (entropy density)

rho_tilde = lambda g_tilde, beta_tilde : exp(- (g_tilde + CV + 1)) / beta_tilde**CV  # AV rho (density)
eps_tilde = lambda g_tilde, beta_tilde : CV * rho_tilde(g_tilde, beta_tilde) / beta_tilde  # AV epsilon (energy density)



# Discretisation
#   Space
nx = 32  # Mesh number (Should be greater than ~Re & ~Re*Pr)
k = 1  # Spatial degree (Must be >=1)

#   Time
timestep = Constant(1 / 4 / nx)  # Timestep (Should be smaller than ~1/M/Re)
duration = Constant(16)  # Duration
S = 1  # Temporal degree (must be >=1)



'''
Useful dependent state variables
'''
g = lambda rho, eps : (s(rho, eps) - (eps + p(rho, eps)) / theta(rho, eps)) / rho
p_tilde = lambda g_tilde, beta_tilde : p(rho_tilde(g_tilde, beta_tilde), eps_tilde(g_tilde, beta_tilde))



'''
Cheeky ideal variables to speed things along
'''
s_fast = lambda sigma, ln_eps : sigma**2 * (CV*ln_eps - 2*(CV+1)*ln(sigma) - CV*ln(CV))
g_fast = lambda sigma, ln_eps : CV*ln_eps - 2*(CV+1)*ln(sigma) - (CV*ln(CV) + (CV+1))
p_tilde_fast = lambda g_tilde, beta_tilde : exp(- (g_tilde + CV + 1)) / beta_tilde**(CV+1)



'''
Mesh
'''
# Create mesh and coordinates
mesh = PeriodicUnitSquareMesh(nx, nx, quadrilateral = False)
(x, y) = SpatialCoordinate(mesh)



'''
Function spaces
'''
# Spatial spaces
Vec_ = VectorFunctionSpace(mesh, "P", k)
Sca_ = FunctionSpace(mesh, "P", k)
SME_ = MixedFunctionSpace([Sca_, Vec_, Sca_])

# Space-time spaces
Vec = cheb_fet.FETVectorFunctionSpace(mesh, "P", k, S-1)
Sca = cheb_fet.FETFunctionSpace(mesh, "P", k, S-1)
SMEGUB = cheb_fet.FETMixedFunctionSpace([Sca, Vec, Sca, Sca, Vec, Sca])



'''
Functions
'''
# IC/persistent value trackers
sme_ = project(as_vector([
    exp(1/2 * sin(2*np.pi*x) * sin(2*np.pi*y)),
    0,
    0,
    (1+1/CV) * sin(2*np.pi*x) * sin(2*np.pi*y)
]), SME_)
(sigma_, mu_, ln_eps_) = split(sme_)

# Trial functions (sigma_t, mu_t, eps_t, g_tilde, u_tilde, beta_tilde)
# smegub = Function(SMEGUB)
# HOTFIX: Trial functions, but with non-zero initial guesses for g_tilde, beta_tilde
smegub = project(as_vector(
    [0 for _ in range(4*S)]
  + [g(sigma_**2, exp(ln_eps_))]
  + [0 for _ in range(3*S - 1)]
  + [1 / theta(sigma_**2, exp(ln_eps_))]
  + [0 for _ in range(S - 1)]
), SMEGUB)
(sigma_t, mu_t, ln_eps_t, g_tilde, u_tilde, beta_tilde) = cheb_fet.FETsplit(smegub)

# Test functions (v_rho, v_m, v_eps, v_g, v_eps, v_beta)
v_rmegub = TestFunction(SMEGUB)
(v_rho, v_m, v_eps, v_g, v_u, v_beta) = cheb_fet.FETsplit(v_rmegub)

# Integrated trial functions
sigma  = cheb_fet.integrate(sigma_t, sigma_, timestep)
mu     = cheb_fet.integrate(mu_t, mu_, timestep)
ln_eps = cheb_fet.integrate(ln_eps_t, ln_eps_, timestep)



'''
Residual
'''
# Symmetric, trace-free gradient
tau = lambda u : sym(grad(u)) - 1/3 * div(u) * Identity(2)



# Initialise
F = 0



# Mass
F = cheb_fet.residual(
    F,
    lambda a, b, c : 2 * inner(a * b, c) * dx,
    (sigma, sigma_t, v_rho)
)

F = cheb_fet.residual(
    F,
    lambda a, b, c, d : - inner(rho_tilde(a, b) * c, grad(d)) * dx,
    (g_tilde, beta_tilde, u_tilde, v_rho),
    poly=False
)



# Momentum
F = cheb_fet.residual(
    F,
    lambda a, b, c : inner(a * b, c) * dx,
    (sigma, mu_t, v_m)
)

F = cheb_fet.residual(
    F,
    lambda a, b, c, d : - 0.5 * (
        inner(rho_tilde(a, b) * outer(c, c), grad(d))
      - inner(rho_tilde(a, b) * dot(grad(c), c), d)
    ) * dx,
    (g_tilde, beta_tilde, u_tilde, v_m),
    poly=False
)
# F = cheb_fet.residual(
#     F,
#     lambda a, b, c : inner(grad(p_tilde(a, b)), c) * dx,
#     (g_tilde, beta_tilde, v_m),
#     poly=False
# )
F = cheb_fet.residual(
    F,
    lambda a, b, c : inner(grad(p_tilde_fast(a, b)), c) * dx,
    (g_tilde, beta_tilde, v_m),
    poly=False
)



# Energy
F = cheb_fet.residual(
    F,
    lambda a, b, c : inner(exp(a) * b, c) * dx,
    (ln_eps, ln_eps_t, v_eps),
    poly=False
)

F = cheb_fet.residual(
    F,
    lambda a, b, c, d : - inner(eps_tilde(a, b) * c, grad(d)) * dx,
    (g_tilde, beta_tilde, u_tilde, v_eps),
    poly=False
)
# F = cheb_fet.residual(
#     F,
#     lambda a, b, c, d : - (
#         inner(dot(grad(p_tilde(a, b)), c), d)
#       + inner(p_tilde(a, b) * c, grad(d))
#     ) * dx,
#     (g_tilde, beta_tilde, u_tilde, v_eps),
#     poly=False
# )
F = cheb_fet.residual(
    F,
    lambda a, b, c, d : - (
        inner(dot(grad(p_tilde_fast(a, b)), c), d)
      + inner(p_tilde_fast(a, b) * c, grad(d))
    ) * dx,
    (g_tilde, beta_tilde, u_tilde, v_eps),
    poly=False
)



# g_tilde
F = cheb_fet.residual(
    F,
    lambda a, b, c : 2 * inner(a * b, c) * dx,
    (sigma, g_tilde, v_g)
)

# F = cheb_fet.residual(
#     F,
#     lambda a, b, c : - 2 * inner(a * g(a**2, exp(b)), c) * dx,
#     (sigma, ln_eps, v_g),
#     poly=False
# )
F = cheb_fet.residual(
    F,
    lambda a, b, c : - 2 * inner(a * g_fast(a, b), c) * dx,
    (sigma, ln_eps, v_g),
    poly=False
)



# u_tilde
F = cheb_fet.residual(
    F,
    lambda a, b, c : inner(a * b, c) * dx,
    (sigma, u_tilde, v_u)
)

F = cheb_fet.residual(
    F,
    lambda a, b : - inner(a, b) * dx,
    (mu, v_u)
)



# beta_tilde
F = cheb_fet.residual(
    F,
    lambda a, b, c : inner(exp(a) * b, c) * dx,
    (ln_eps, beta_tilde, v_beta),
    poly=False
)

F = cheb_fet.residual(
    F,
    lambda a, b, c : - inner(exp(a) / theta(b**2, exp(a)), c) * dx,
    (ln_eps, sigma, v_beta),
    poly=False
)



'''
Solver parameters
'''
sp = {
    # Outer (nonlinear) solver
    "snes_atol": 1e-15,
    "snes_rtol": 1e-15,

    "snes_converged_reason"     : None,
    "snes_linesearch_monitor"   : None,
    "snes_monitor"              : None,

    # Inner (linear) solver
    "ksp_type"                  : "preonly",  # Krylov subspace = GMRes
    "pc_type"                   : "lu",
    "pc_factor_mat_solver_type" : "mumps",
    #"ksp_atol"                  : 1e-8,
    #"ksp_rtol"                  : 1e-8,
    #"ksp_max_it"                : 100,

    "ksp_monitor_true_residual" : None,
}



'''
Solve setup
'''
# Create ParaView file
pvd = VTKFile("output/7_pde/compressible_ns/euler/avfet/solution.pvd")

# Write to Paraview file
(sigma_sub, mu_sub, ln_eps_sub) = sme_.subfunctions
sigma_sub.rename("Root density")
mu_sub.rename("Root density * velocity")
ln_eps_sub.rename("Log internal energy")

pvd.write(sigma_sub, mu_sub, ln_eps_sub)



# Create text files
mass_txt            = "output/7_pde/compressible_ns/euler/avfet/mass.txt"
momentum_txt        = "output/7_pde/compressible_ns/euler/avfet/momentum.txt"
kinetic_energy_txt  = "output/7_pde/compressible_ns/euler/avfet/kinetic_energy.txt"
internal_energy_txt = "output/7_pde/compressible_ns/euler/avfet/internal_energy.txt"
energy_txt          = "output/7_pde/compressible_ns/euler/avfet/energy.txt"
entropy_txt         = "output/7_pde/compressible_ns/euler/avfet/entropy.txt"

# Write to text files
mass = assemble(sigma_**2 * dx)
print(GREEN % f"Mass: {mass}")
if mesh.comm.rank == 0:
    open(mass_txt, "w").write(str(mass) + "\n")

momentum = [float(assemble(sigma_ * mu_[i] * dx)) for i in range(2)]
print(GREEN % f"Momentum: {momentum}")
if mesh.comm.rank == 0:
    open(momentum_txt, "w").write(str(momentum) + "\n")

kinetic_energy = assemble(1/2 * inner(mu_, mu_) * dx)
internal_energy = assemble(exp(ln_eps_) * dx)
energy = kinetic_energy + internal_energy
print(GREEN % f"Energy: {energy}")
if mesh.comm.rank == 0:
    open(kinetic_energy_txt, "w").write(str(kinetic_energy) + "\n")
if mesh.comm.rank == 0:
    open(internal_energy_txt, "w").write(str(internal_energy) + "\n")
if mesh.comm.rank == 0:
    open(energy_txt, "w").write(str(energy) + "\n")

# entropy = assemble(s(sigma_**2, exp(ln_eps_)) * dx)
entropy = assemble(s_fast(sigma_, ln_eps_) * dx)
print(GREEN % f"Entropy: {entropy}")
if mesh.comm.rank == 0:
    open(entropy_txt, "w").write(str(entropy) + "\n")



'''
Solve
'''
time = Constant(0.0)
while (float(time) < float(duration) - float(timestep)/2):
    # Print timestep
    print(RED % f"Solving for t = {float(time) + float(timestep)}:")

    # Solve
    solve(F == 0, smegub, solver_parameters = sp)

    # Collect garbage
    gc.collect()

    # Update variables
    sigma_sub.assign(cheb_fet.FETeval((sme_, 0), (smegub, 0), timestep, timestep))
    mu_sub.assign(cheb_fet.FETeval((sme_, 1), (smegub, 1), timestep, timestep))
    ln_eps_sub.assign(cheb_fet.FETeval((sme_, 2), (smegub, 2), timestep, timestep))

    # Write to Paraview
    pvd.write(sigma_sub, mu_sub, ln_eps_sub)

    # Write to text files
    mass = assemble(sigma_**2 * dx)
    print(GREEN % f"Mass: {mass}")
    if mesh.comm.rank == 0:
        open(mass_txt, "a").write(str(mass) + "\n")

    momentum = [float(assemble(sigma_ * mu_[i] * dx)) for i in range(2)]
    print(GREEN % f"Momentum: {momentum}")
    if mesh.comm.rank == 0:
        open(momentum_txt, "a").write(str(momentum) + "\n")

    kinetic_energy = assemble(1/2 * inner(mu_, mu_) * dx)
    internal_energy = assemble(exp(ln_eps_) * dx)
    energy = kinetic_energy + internal_energy
    print(GREEN % f"Energy: {energy}")
    if mesh.comm.rank == 0:
        open(kinetic_energy_txt, "a").write(str(kinetic_energy) + "\n")
    if mesh.comm.rank == 0:
        open(internal_energy_txt, "a").write(str(internal_energy) + "\n")
    if mesh.comm.rank == 0:
        open(energy_txt, "a").write(str(energy) + "\n")

    # entropy = assemble(s(sigma_**2, exp(ln_eps_)) * dx)
    entropy = assemble(s_fast(sigma_, ln_eps_) * dx)
    print(GREEN % f"Entropy: {entropy}")
    if mesh.comm.rank == 0:
        open(entropy_txt, "a").write(str(entropy) + "\n")

    # Increment time
    time.assign(float(time) + float(timestep))
