'''
Imports
'''
from firedrake import *
import avfet_modules.cheb_fet as cheb_fet
import gc



'''
General
'''
# Parallelise "print"
_print = print  # Save old serial "print"
def print(x):  # Create new parallel "print"
    if mesh.comm.rank == 0:
        _print(x, flush = True)

# Increase spatial quadrature degree
dx = dx(degree=20)



'''
Parameters
'''
# Model
#   Setting
M = Constant(2**0)  # Mach number
Re = Constant(2**0)  # Reynolds number

#   Material
Pr = Constant(0.71)  # Prandtl number (Material-dependent, ~0.71 standard for air)

#   Constitutive relations (Set to standard for ideal fluid)  
CV = Constant(2.50)  # Specific heat capacity (Material-dependent, ~2.50 standard for air)

p     = lambda rho, eps : 1/CV * eps  # Non-AV p (pressure)
theta = lambda rho, eps : p(rho, eps) / rho  # Non-AV theta (temperature)
s     = lambda rho, eps : CV * ln(theta(rho, eps)) - ln(rho)  # Non-AV s (specific entropy)

rho_tilde = lambda g_tilde, beta_tilde : beta_tilde**(-CV) * exp(- (g_tilde + CV + 1))  # AV rho (density)
eps_tilde = lambda g_tilde, beta_tilde : CV * rho_tilde(g_tilde, beta_tilde) / beta_tilde  # AV epsilon (energy density)



# Discretisation
#   Space
nx = round(2 * (2**5))  # Mesh number (Should be greater than ~Re & ~Re*Pr)
k = 1  # Spatial degree (Must be >=1)

#   Time
timestep = Constant(1 / 2 / float(M) / (2**7))  # Timestep (Should be smaller than ~1/M/Re)
duration = Constant(1 / 4)  # Duration
S = 1  # Temporal degree (must be >=1)



'''
Useful dependent state variables
'''
# Cell
g = lambda rho, eps : s(rho, eps) - (eps + p(rho, eps)) / theta(rho, eps) / rho
p_tilde = lambda g_tilde, beta_tilde : p(rho_tilde(g_tilde, beta_tilde), eps_tilde(g_tilde, beta_tilde))

# Facet
rho_facet = lambda g_tilde, beta_tilde : conditional(
    le(abs(g_tilde('+') - g_tilde('-')), 1e-10),
    rho_tilde(avg(g_tilde), avg(beta_tilde)),
  - avg(beta_tilde)
  * (p_tilde(g_tilde('+'), avg(beta_tilde)) - p_tilde(g_tilde('-'), avg(beta_tilde)))
  / (g_tilde('+') - g_tilde('-'))
)



'''
Cheeky variables to speed things along in the ideal case
'''
# Cell
s_fast = lambda sigma, ln_eps : CV*ln_eps - 2*(CV+1)*ln(sigma) - CV*ln(CV)
g_fast = lambda sigma, ln_eps : s_fast(sigma, ln_eps) - (CV + 1)
p_tilde_fast = lambda g_tilde, beta_tilde : rho_tilde(g_tilde, beta_tilde) / beta_tilde

# Facet
rho_facet_fast = lambda g_tilde, beta_tilde : conditional(
    le(abs(g_tilde('+') - g_tilde('-')), 1e-10),
    rho_tilde(avg(g_tilde), avg(beta_tilde)),
  - exp(- (CV+1))
  * avg(beta_tilde)**(- (CV+1))
  * (exp(g_tilde('+')) - exp(g_tilde('-'))) / (g_tilde('+') - g_tilde('-'))
)



'''
Mesh
'''
# Create mesh and coordinates
mesh = PeriodicUnitSquareMesh(nx, nx, quadrilateral = False)
(x, y) = SpatialCoordinate(mesh)
n = FacetNormal(mesh)



'''
Function spaces
'''
# Spatial spaces
Vec_CG_ = VectorFunctionSpace(mesh, "CG", k)
Vec_DG_ = VectorFunctionSpace(mesh, "DG", k)
Sca_CG_ = FunctionSpace(mesh, "CG", k)
Sca_DG_ = FunctionSpace(mesh, "DG", k)
SME_ = MixedFunctionSpace([Sca_DG_, Vec_DG_, Sca_DG_])

# Space-time spaces
Vec_CG = cheb_fet.FETVectorFunctionSpace(mesh, "CG", k, S-1)
Vec_DG = cheb_fet.FETVectorFunctionSpace(mesh, "DG", k, S-1)
Sca_CG = cheb_fet.FETFunctionSpace(mesh, "CG", k, S-1)
Sca_DG = cheb_fet.FETFunctionSpace(mesh, "DG", k, S-1)
SMEGUB = cheb_fet.FETMixedFunctionSpace([Sca_DG, Vec_DG, Sca_DG, Sca_DG, Vec_CG, Sca_CG])



'''
Functions
'''
# IC/persistent value trackers
sme_ = project(as_vector([
    1,
    M * exp(1*(cos(2*pi*(x-0)) - 1)) * exp(1*(cos(2*pi*(y-1/2)) - 1)),
    0,
    0
]), SME_)
(sigma_, mu_, ln_eps_) = split(sme_)

# Trial functions (sigma_t, mu_t, eps_t, g_tilde, u_tilde, beta_tilde)
# smegub = Function(SMEGUB)
smegub = project(as_vector(
    [0 for _ in range(4*S)]
  + [g(sigma_**2, exp(ln_eps_))]
  + [0 for _ in range(3*S - 1)]
  + [1 / theta(sigma_**2, exp(ln_eps_))]
  + [0 for _ in range(S - 1)]
), SMEGUB)  # HOTFIX: Trial functions, but with non-zero initial guesses for g_tilde, beta_tilde
(sigma_t, mu_t, ln_eps_t, g_tilde, u_tilde, beta_tilde) = cheb_fet.FETsplit(smegub)

# Test functions (v_rho, v_m, v_eps, v_g, v_eps, v_beta)
v_rmegub = TestFunction(SMEGUB)
(v_g, v_u, v_beta, v_rho, v_m, v_eps) = cheb_fet.FETsplit(v_rmegub)

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
F = cheb_fet.residual(
    F,
    lambda a, b, c, d : inner(rho_facet_fast(a, b) * avg(c), 2*avg(d * n)) * dS,
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
F = cheb_fet.residual(
    F,
    lambda a, b, c, d : 2/Re * a * b * inner(sym(grad(c)), sym(grad(d))) * dx,
    (sigma, sigma, u_tilde, v_m)
)
F = cheb_fet.residual(
    F,
    lambda a, b, c, d : - 2/Re * 1/3 * a * b * inner(div(c), div(d)) * dx,
    (sigma, sigma, u_tilde, v_m)
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
F = cheb_fet.residual(
    F,
    lambda a, b, c, d, e : - 2/Re * a * b * inner(inner(sym(grad(c)), sym(grad(d))), e) * dx,
    (sigma, sigma, u_tilde, u_tilde, v_eps)
)
F = cheb_fet.residual(
    F,
    lambda a, b, c, d, e : 2/Re * 1/3 * a * b * inner(inner(div(c), div(d)), e) * dx,
    (sigma, sigma, u_tilde, u_tilde, v_eps)
)
F = cheb_fet.residual(
    F,
    lambda a, b, c, d : - 1/Re/Pr * a**2 * theta(a**2, exp(b))**2 * inner(grad(c), grad(d)) * dx,
    (sigma, ln_eps, beta_tilde, v_eps),
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
pvd = VTKFile("output/7_pde/compressible_ns/shockwave/avfet_dg/solution.pvd")

# Write to Paraview file
(sigma_sub, mu_sub, ln_eps_sub) = sme_.subfunctions
sigma_sub.rename("Root density")
mu_sub.rename("Root density * velocity")
ln_eps_sub.rename("Log internal energy")

pvd.write(sigma_sub, mu_sub, ln_eps_sub)



# Create text files
mass_txt            = "output/7_pde/compressible_ns/shockwave/avfet_dg/mass.txt"
momentum_txt        = "output/7_pde/compressible_ns/shockwave/avfet_dg/momentum.txt"
kinetic_energy_txt  = "output/7_pde/compressible_ns/shockwave/avfet_dg/kinetic_energy.txt"
internal_energy_txt = "output/7_pde/compressible_ns/shockwave/avfet_dg/internal_energy.txt"
energy_txt          = "output/7_pde/compressible_ns/shockwave/avfet_dg/energy.txt"
entropy_txt         = "output/7_pde/compressible_ns/shockwave/avfet_dg/entropy.txt"

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

# entropy = assemble(sigma_**2 * s(sigma_**2, exp(ln_eps_)) * dx)
entropy = assemble(sigma_**2 * s_fast(sigma_, ln_eps_) * dx)
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
    
    # Record dissipation
    # visc_diss = cheb_fet.FETassemble(
    #     lambda a, b, c, d, e : 2/Re * a*b * c * inner(tau(d), tau(e)) * dx,
    #     (sigma, sigma, beta_tilde, u_tilde, u_tilde),
    #     timestep
    # )
    visc_diss = cheb_fet.FETassemble(
        lambda a, b, c, d, e : 2/Re * a*b * c * inner(sym(grad(d)), sym(grad(e))) * dx,
        (sigma, sigma, beta_tilde, u_tilde, u_tilde),
        timestep
    ) + cheb_fet.FETassemble(
        lambda a, b, c, d, e : - 2/Re * 1/3 * a*b * c * inner(div(d), div(e)) * dx,
        (sigma, sigma, beta_tilde, u_tilde, u_tilde),
        timestep
    )
    ther_diss = cheb_fet.FETassemble(
        lambda a, b, c : 1/Re/Pr * a**2 * theta(a**2, exp(b))**2 * inner(grad(c), grad(c)) * dx,
        (sigma, ln_eps, beta_tilde),
        timestep,
        poly=False
    )
    diss = visc_diss + ther_diss
    print(BLUE % f"Dissipation: {diss}")

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

    # entropy = assemble(sigma_**2 * s(sigma_**2, exp(ln_eps_)) * dx)
    entropy = assemble(sigma_**2 * s_fast(sigma_, ln_eps_) * dx)
    print(GREEN % f"Entropy: {entropy}")
    if mesh.comm.rank == 0:
        open(entropy_txt, "a").write(str(entropy) + "\n")

    # Increment time
    time.assign(float(time) + float(timestep))
