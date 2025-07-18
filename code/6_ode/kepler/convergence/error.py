from firedrake import *
import avfet_modules.terminal_options as terminal_options



'''
Parameters
'''
stages = terminal_options.get("stages", type=int, default=1)

dt  = terminal_options.get("dt", type=float, default=2**(-5)*2*pi)
dur = terminal_options.get("dur", type=float, default=2*pi)



'''
Meshes and functions spaces
'''
# Mesh
mesh = IntervalMesh(1, float(dt))

# Function spaces
V = VectorFunctionSpace(mesh, "CG", stages,   dim=4)
W = VectorFunctionSpace(mesh, "DG", stages-1, dim=4)
Z = MixedFunctionSpace([V, W, W, W])

S = FunctionSpace(mesh, "CG", 1)  # For interpolation at end points

# Functions
z = Function(Z, name="Solution")
(u, w0, w1, w2) = split(z)
(v, v0, v1, v2) = split(TestFunction(Z))



'''
Utility functions for residual
'''
# Quadrature rule (I_n)
dx_h = dx(degree=2*stages - 1)
dx_e = dx(degree=25)  # Increase quadrature degree

# Invariants
uv = variable(u)
p = as_vector([uv[k] for k in range(0, 2)])
q = as_vector([uv[k] for k in range(2, 4)])

H  = 0.5*(p[0]**2 + p[1]**2) - 1/sqrt(q[0]**2 + q[1]**2)
L  =  p[1]*q[0] - p[0]*q[1]
A1 =  L*p[1] - 1/sqrt(dot(q, q)) * q[0]
A2 = -L*p[0] - 1/sqrt(dot(q, q)) * q[1]
invariants = [H, L, A1, A2]
names = ["Hamiltonian     ",
         "Angular momentum",
         "Runge-Lenz 1    ",
         "Runge-Lenz 2    "]

dHdu = diff(H, uv)
dA1du = diff(A1, uv)
dA2du = diff(A2, uv)



'''
Residual
'''
to_arr = lambda ufl_obj : [ufl_obj[i] for i in range(4)]

F = (
      inner(u.dx(0), v.dx(0)) * dx_h
    + 0.5 * inner(1/L/H, det(as_matrix([to_arr(w0), to_arr(w1), to_arr(w2), to_arr(v.dx(0))]))) * dx_h
    # - 0.5/0.5/0.8 * det(as_matrix([to_arr(w0), to_arr(w1), to_arr(w2), to_arr(v.dx(0))])) * dx_h
    + inner(w0, v0) * dx_h
    - inner(dHdu, v0) * dx_e  # not dx_h!
    + inner(w1, v1) * dx_h
    - inner(dA1du, v1) * dx_e  # not dx_h!
    + inner(w2, v2) * dx_h
    - inner(dA2du, v2) * dx_e  # not dx_h!
    )



'''
Initial conditions
'''
ubc = Constant((0, 2, 0.4, 0))
bc = DirichletBC(Z.sub(0), ubc, 1)



'''
Initial guesses
'''
z.subfunctions[0].interpolate(ubc)
z.subfunctions[1].interpolate(dHdu)
z.subfunctions[2].interpolate(dA1du)
z.subfunctions[3].interpolate(dA2du)



'''
Solver parameters
'''
sp = {
    #"snes_monitor": None,
    "snes_linesearch_type": "l2",
    #"snes_converged_reason": None,
    "snes_atol": 1.0e-14,
    "snes_rtol": 1.0e-14,
    "mat_type": "dense",
    "pc_type": "lu"
}



'''
Solve loop
'''
print(f"dt = {dt}, stages = {stages}")

for i in range(round(dur/dt)):
    solve(F == 0, z, bc, solver_parameters=sp)
    u_dt = z.subfunctions[0](dt)
    #print(f"u({t}) = ", u_dt)
    ubc.assign(u_dt)

    # print("-"*19 + " invariants at time t = ", dt*(i+1))
    # for (inv, name) in zip(invariants, names):
    #     J_ = Function(S).interpolate(inv)
    #     J_prev = J_(0)
    #     J_next = J_(dt)
    #     print(f"  {name}: prev = {J_prev} next = {J_next} diff = {J_next - J_prev:e}")
    # print()



'''
Error evaluation
'''
error = np.linalg.norm(u_dt[2:4] - [0.4, 0])
print(BLUE % f"error = {error:e}")
with open("output/6_ode/kepler/convergence/stages_" + str(stages) + ".txt", "a") as file:
    file.write(str(dt) + " " + str(error) + "\n")
