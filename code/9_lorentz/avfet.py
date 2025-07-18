from firedrake import *
import numpy as np
import avfet_modules.terminal_options as terminal_options



'''
Parameters
'''
# Discretisation
stages = terminal_options.get("stages", type=int, default=1)
mu_x_tol = 1e-6  # Tolerance below which we flatten out ||mu_x|| (so the solver doesn't break)

# Timestepping
dt = terminal_options.get("dt", type=float, default=2**-4)
dur = terminal_options.get("dur", type=float, default=2**6)

# Output
folder = terminal_options.get("folder", type=str, default="9_lorentz/avfet")

# Field
rho = terminal_options.get("rho", type=Constant, default=2**-5)

r = 2**2  # Current loop radius
L = 2**4  # Length between loops
B_ = lambda x : 1/2 * (r**2 + (L/2)**2)**(3/2) * (
    1/(r**2 + (x[2]+L/2)**2)**(3/2) * as_vector([
        3*x[0]/2 * (x[2]+L/2) / (r**2 + (x[2]+L/2)**2),
        3*x[1]/2 * (x[2]+L/2) / (r**2 + (x[2]+L/2)**2),
        1
    ])
  + 1/(r**2 + (x[2]-L/2)**2)**(3/2) * as_vector([
        3*x[0]/2 * (x[2]-L/2) / (r**2 + (x[2]-L/2)**2),
        3*x[1]/2 * (x[2]-L/2) / (r**2 + (x[2]-L/2)**2),
        1
    ])
)

# Initial conditions
x_ic = np.array([0, rho, 0])
v_ic = np.array([1, 0,   2.1])  # 2.1 - 2.142... (Should hypothetically mirror at 2.142468... = (1/2 * (1 + (L/2/r)**2)**(3/2) - 1)**(1/2))



'''
Meshes and functions spaces
'''
# Mesh
mesh = IntervalMesh(1, float(dt))
(t,) = SpatialCoordinate(mesh)

# Function spaces
V = VectorFunctionSpace(mesh, "CG", stages,   dim=3)
W = VectorFunctionSpace(mesh, "DG", stages-1, dim=3)
Z = MixedFunctionSpace([V, V, W, W, W])

S = FunctionSpace(mesh, "CG", 1)  # For interpolation at end points

# Functions
z = Function(Z, name="Solution")
(x, v, v_til, mu_x_til, mu_v_til) = split(z)
(y, w, w_til, lm_x_til, lm_v_til) = split(TestFunction(Z))



'''
Utility functions
'''
# Energy
energy = 1/2 * dot(v, v)

# Utility functions
cross = lambda a, b : as_vector([
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0]
])
triple = lambda a, b, c : dot(cross(a, b), c)

# Magnetic field
B = B_(x)  # field...
B_norm = dot(B, B)**(1/2)  # ...norm...
b_par = B / B_norm  # ...normalised

# Parallel/perpendicular velocity
v_par = dot(v, b_par)  # parallel velocity...
v_perp = dot(v - v_par * b_par, v - v_par * b_par)**(1/2)  # ...perpendicular velocity
b_perp = (v - v_par * b_par) / v_perp  # direction of velocity perpendicular to field...
b_star = cross(b_par, b_perp)  # ...completed as a basis by this

# B gradient
B_grad = as_vector([diff(B, z)[:, i] for i in range(3)]).T

# Magnetic moment
mu_0 = 1/2 * v_perp**2 / B_norm
mu_1 = 1/B_norm**3 * inner(
    1/4 * v_par**2*v_perp * outer(b_perp, b_star)
  + 1/4 * v_par**2*v_perp * outer(b_star, b_perp)
  + 1/2 * v_perp**3       * outer(b_par,  b_star)
  + 1   * v_par**2*v_perp * outer(b_star, b_par),
    B_grad
)

# Magnetic moment derivatives
mu_x = as_vector([diff(mu_0,            z)[i] for i in range(0, 3)])
mu_x_norm = dot(mu_x, mu_x)**(1/2)

mu_v = as_vector([diff(mu_0 + rho*mu_1, z)[i] for i in range(3, 6)])



'''
Residual
'''
# Quadrature rules
dx_h = dx(degree=2*stages-1)
dx_e = dx(degree=10)  # Increase quadrature degree

# Residual
F = (
    (  # Position
        inner(mu_x_norm * x.dx(0), y.dx(0)) * dx_e
      - inner(mu_x_norm * cross(mu_x_til, v_til), cross(mu_x_til, y.dx(0))) * dx_e
      + 1/rho * inner(triple(v_til, B, mu_v_til), dot(mu_x_til, y.dx(0))) * dx_e
    )
  + (  # Velocity
        inner(v.dx(0), w.dx(0)) * dx_e
      - 1/rho * inner(dot(mu_x_til, mu_x_til) * cross(v_til, B), w.dx(0)) * dx_e
    )
  + (  # Auxiliary velocity
        inner(v_til, w_til) * dx_e
      - inner(v,     w_til) * dx_e  # Not dx_h!
    )
  + (  # Moment gradient (x)
        inner(mu_x_norm * mu_x_til, lm_x_til) * dx_e
      - inner(mu_x,                 lm_x_til) * dx_e  # Not dx_h!
    )
  + (  # Moment gradient (v)
        inner(mu_v_til, lm_v_til) * dx_e
      - inner(mu_v,     lm_v_til) * dx_e  # Not dx_h!
    )
)



'''
Initial conditions
'''
x_ = Constant(x_ic)
v_ = Constant(v_ic)

ic = [
    DirichletBC(Z.sub(0), x_, 1),
    DirichletBC(Z.sub(1), v_, 1)
]



'''
Initial guesses
'''
z.subfunctions[0].interpolate(x_ + t*v_)
z.subfunctions[1].interpolate(v_)
z.subfunctions[2].interpolate(v)
z.subfunctions[3].interpolate(mu_x)
z.subfunctions[4].interpolate(mu_v)



'''
Solver parameters
'''
sp = {
    "snes_monitor": None,
    "snes_linesearch_type": "l2",
    #"snes_converged_reason": None,
    "snes_atol": 1.0e-14,
    "snes_rtol": 1.0e-14,
    "snes_max_it": 500,
    "mat_type": "dense",
    "pc_type": "lu"
}



'''
Solve loop
'''
# Make data arrays
x_arr = np.array([x_ic])
v_arr = np.array([v_ic])

# Run loop
print(GREEN % f"dt = {dt}, stages = {stages}:")
for i in range(round(dur/dt)):
    print(BLUE % f"Solving for time t = {(i+1)*dt}...")

    # Solve
    solve(F == 0, z, ic, solver_parameters=sp)
    
    # Get output
    x_out = z.subfunctions[0](dt)
    v_out = z.subfunctions[1](dt)
    energy_ = Function(S).interpolate(energy)
    mu_0_ = Function(S).interpolate(mu_0)
    mu_1_ = Function(S).interpolate(mu_1)

    # Make data arrays
    if i == 0:
        energy_arr = np.array([energy_(0)])
        mu_0_arr = np.array([mu_0_(0)])
        mu_1_arr = np.array([mu_1_(0)])

    # Record data
    x_arr = np.vstack([x_arr, x_out])
    v_arr = np.vstack([v_arr, v_out])
    energy_arr = np.vstack([energy_arr, energy_(dt)])
    mu_0_arr = np.vstack([mu_0_arr, mu_0_(dt)])
    mu_1_arr = np.vstack([mu_1_arr, mu_1_(dt)])

    # Update
    x_.assign(x_out)
    v_.assign(v_out)



'''
Save data
'''
if mesh.comm.rank == 0:
    np.savetxt("output/" + folder + "/x.txt", x_arr)
    np.savetxt("output/" + folder + "/energy.txt", energy_arr)
    np.savetxt("output/" + folder + "/mu.txt", mu_0_arr)
    np.savetxt("output/" + folder + "/mu_corrected.txt", mu_0_arr + rho*mu_1_arr)
