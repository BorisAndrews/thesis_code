'''
Specialist integrators for Hamiltonian systems
'''
from .default_initial_guess import *
from .collocation_runge_kutta import *
import numpy as np



# The new fully conservative scheme of Andrews & Farrell (2024) (as a modification of Gauss--Legendre)
def andrews_farrell(x_start, dt, N_grad, alt, s, res=1, quad_pts_min=10, snes_tol=1e-15, initial_guess=default_initial_guess):
    # Test PETSc is imported
    try:
        PETSc.__spec__
    except NameError:
        try:
            import petsc4py
            import sys
            petsc4py.init(sys.argv)
            from petsc4py import PETSc
        except ImportError:
            raise ImportError("Failed to import PETSc, which is required to run implicit schemes. Either specify a destination from which to import PETSc through '-petsc' (e.g. '-petsc firedrake') or use an explicit scheme (e.g. '-scheme rk4').")

    # Retrieve problem size
    n = x_start.size

    # Normalise input
    x_start = x_start.astype(float)

    # Set collocation points as GL points
    (c_nodes, c_weights) = np.polynomial.legendre.leggauss(s)
    c_nodes = 0.5 * (c_nodes + 1)
    c_weights = 0.5 * c_weights

    # Get GL quadrature points
    (quad_nodes, quad_weights) = np.polynomial.legendre.leggauss(max(2*s, quad_pts_min))
    quad_nodes = 0.5 * (quad_nodes + 1)
    quad_weights = 0.5 * quad_weights

    # Get coeffs
    coll_rk_coeffs = get_coll_rk_coeffs(c_nodes)
    lagrange_eval_coeffs_grid = get_lagrange_eval_coeffs_grid(c_nodes, res)

    # Define evaluation function
    def x(t, x_mat_arr_):
        lagrange_eval_coeffs = get_lagrange_eval_coeffs(c_nodes, t)
        x = lagrange_eval_coeffs[0] * x_start + sum([
            lagrange_eval_coeff * x_mat_arr_[i*n : (i+1)*n]
            for (i, lagrange_eval_coeff) in enumerate(lagrange_eval_coeffs[1:])
        ])
        return x

    # Define residual
    def res_func(snes, x_mat_PETSc_, res_PETSc_):
        x_mat_arr_ = x_mat_PETSc_.getArray(readonly=True)
        res_arr = res_PETSc_.getArray()
        # At each collocation point...
        for (i, c_node) in enumerate(c_nodes):
            c_nodes_red = np.delete(c_nodes, i)  # c with c_i removed
            l_i = lambda t: np.prod([t - c_nodes_red])/np.prod([c_node - c_nodes_red])  # l_i(t)

            N_grad_arr = 1/c_weights[i] * sum([  # \tilde{N_grad} at c_i
                quad_weight  # quad_weight
              * l_i(quad_node)  # l_i(quad_node)
              * N_grad(x(quad_node, x_mat_arr_))  # N_grad(quad_node)
                for (quad_node, quad_weight) in zip(quad_nodes, quad_weights)
            ])

            for j in range(n):
                res_arr[i*n + j] = dt * alt(x_mat_arr_[i*n : (i+1)*n], *N_grad_arr, np.eye(1, n, j).flatten()) - (
                    coll_rk_coeffs[0, i+1] * x_start[j]
                  + np.sum([coll_rk_coeffs[k+1, i+1] * x_mat_arr_[k*n+j] for k in range(s)])
                )

    # Create PETSc vector for dx/dt-like term
    x_mat_PETSc = PETSc.Vec().create()
    x_mat_PETSc.setSizes(n*s)
    x_mat_PETSc.setFromOptions()
    dx_initial_guess = x_start - x_start
    x_mat_PETSc.setArray(np.array([x_start + c_nodes[i] * dx_initial_guess for i in range(s)]))
            
    # Create SNES
    solver = PETSc.SNES().create()
    x_mat_PETSc_dup = x_mat_PETSc.duplicate()
    solver.setFunction(res_func, x_mat_PETSc_dup)
    solver.setTolerances(atol=snes_tol, rtol=snes_tol)
    solver.setFromOptions()
    
    # Run solve
    solver.solve(None, x_mat_PETSc)
    x_mat_arr = x_mat_PETSc.getArray().reshape(s, n)

    # Destroy PETSc objects
    x_mat_PETSc.destroy()
    x_mat_PETSc_dup.destroy()
    solver.destroy()

    # Find output
    x_out = np.outer(
        lagrange_eval_coeffs_grid[0, :],
        x_start
    ) + np.sum([np.outer(
        lagrange_eval_coeffs_grid[i+1, :],
        x_mat_arr[i, :]
    ) for i in range(s)], axis=0)

    return x_out