'''
Need to tidy this whole file up:
- Don't import all of firedrake
- Add documentation
'''

from firedrake import *



'''
Project (With chosen inner product)
'''
def project_alt(target, U, inner_prod=lambda a,b : inner(a,b)*dx, **kwargs):
    # Create functions for projection
    u = Function(U)
    v = TestFunction(U)

    # Create residual
    F = inner_prod(u - target, v)

    # Solve
    if "solver_parameters" in kwargs:
        if "bcs" in kwargs:
            solve(F == 0, u, bcs=kwargs["bcs"], solver_parameters=kwargs["solver_parameters"])
        else:
            solve(F == 0, u, solver_parameters=kwargs["solver_parameters"])
    else:
        if "bcs" in kwargs:
            solve(F == 0, u, bcs=kwargs["bcs"])
        else:
            solve(F == 0, u)

    return u



'''
Project into operator-free subspace (With chosen inner product)
'''
def project_op_free(target, U, *op_tup, inner_prod=lambda a,b : inner(a,b)*dx, **kwargs):
    # Create Lagrange multiplier space
    UP = MixedFunctionSpace([U, *[op[1] for op in op_tup]])

    # Create functions for projection
    up = Function(UP)
    (u, *p_arr) = split(up)
    (v, *q_arr) = split(TestFunction(UP))

    # Create Lagrange multiplier residual
    F = (
        inner_prod(u - target, v)
      - sum([
            op[0](v, p) + op[0](u, q)
            for (op, p, q) in zip(op_tup, p_arr, q_arr)
        ])
    )

    # Solve
    if "solver_parameters" in kwargs:
        if "bcs" in kwargs:
            solve(F == 0, up, bcs=kwargs["bcs"], solver_parameters=kwargs["solver_parameters"])
        else:
            solve(F == 0, up, solver_parameters=kwargs["solver_parameters"])
    else:
        if "bcs" in kwargs:
            solve(F == 0, up, bcs=kwargs["bcs"])
        else:
            solve(F == 0, up)
        

    # Assign solution
    u_ = Function(U)
    u_.assign(up.sub(0))

    return u_
