The code used to generate the numerical results in the thesis
"Geometric numerical integration via auxiliary variables" by
Boris D. Andrews.

Packages (located in "avfet_modules/") can be installed by running
    >> pip install .
in the root directory. They can be subsequently uninstalled by running
    >> pip uninstall avfet_modules
in the root directory.



----------



The following gives instructions to reproduce each of the figures in the
manuscript. Code marked with a ">>" must be executed in a shell in the
root directory; unless otherwise stated, this must be done in a
Firedrake virtual environment. Relevant targets are available in the
included Makefile for convenience. All outputs will be saved to a
created "output/" directory.



- To generate the plots and data in Figs. 3.1 & 3.2, in the
Q_2-preserving case:
    >> mkdir -p output/3_framework/helicity/
    >> python code/3_framework/helicity.py
and in the non-Q_2-preserving case:
    >> mkdir -p output/3_framework/no_helicity/
    >> python code/3_framework/no_helicity.py

Data and Paraview files will be saved in either case to
"output/3_framework/".



- To generate the plots and data in Figs. 6.1 & 6.2 for each scheme:
    >> python code/6_ode/kepler/comparison.py --scheme implicit_midpoint
    >> python code/6_ode/kepler/comparison.py --scheme cohen_hairer
    >> python code/6_ode/kepler/comparison.py --scheme labudde_greenspan
    >> python code/6_ode/kepler/comparison.py --scheme andrews_farrell

These do not require a Firedrake virtual environment, however it does
require a PETSc installation which can be fetched through Firedrake; a
Firedrake virtual environment is therefore sufficient.



- To generate the data in Fig. 6.3:
    >> mkdir -p output/6_ode/kepler/convergence/
    >> python code/6_ode/kepler/convergence/looper.py

Outputs will be saved as text files in "output/kepler_convergence/".



- To generate the plots and data in Figs. 6.4 & 6.5:
    >> mkdir -p output/6_ode/kovalevskaya/im/ output/6_ode/kovalevskaya/avfet/
    >> matlab -batch "run('code/6_ode/kovalevskaya.mlx')"

Images will be saved in "output/6_pde/kovalevskaya/". Naturally, this does
not require a Firedrake virtual environment.



- To generate the plots and data in Figs. 7.1, 7.2 & 7.3, for the
energy-conserving integrator:
    >> mkdir -p output/7_pde/bbm/avfet/
    >> python code/7_pde/bbm/avfet.py
and for the Gauss method:
    >> mkdir -p output/7_pde/bbm/gauss/
    >> python code/7_pde/bbm/gauss.py

Data will saved in either case to "output/7_pde/bbm/".

To produce an animation of the data, for the energy-conserving
integrator:
    >> python code/7_pde/bbm/animation.py --dir output/7_pde/bbm/avfet/
and for the Gauss method:
    >> python code/7_pde/bbm/animation.py --dir output/7_pde/bbm/gauss/
In either case, the animation can be viewed through a frame of
reference moving at the exact speed of the continuous soliton by appending:
    >> * --cam_speed 1.618034
e.g.
    >> python code/7_pde/bbm/animation.py --dir output/7_pde/bbm/avfet/ --cam_speed 1.618034



- To generate the plots and data in Figs. 7.4 & 7.5, for the
structure-preserving integrator:
    >> python code/7_pde/compressible_ns/shockwave/avfet.py
and for the implicit midpoint method:
    >> python code/7_pde/compressible_ns/shockwave/im.py

Data and Paraview files will saved in either case to
"output/7_pde/compressible_ns/shockwave/".



- To generate the data in Fig. 7.6, for the structure-
preserving integrator:
    >> python code/7_pde/compressible_ns/euler/avfet.py
and for the implicit midpoint method:
    >> python code/7_pde/compressible_ns/euler/im.py

Data and Paraview files will saved in either case to
"output/7_pde/compressible_ns/euler/".



- To generate the data in Figs. 9.2 & 9.3, for the structure-
preserving integrator:
    >> mkdir -p output/9_lorentz/avfet/
    >> python code/9_lorentz/avfet.py
and for the implicit midpoint method:
    >> mkdir -p output/9_lorentz/im/
    >> python code/9_lorentz/im.py

Data will saved in either case to "output/9_lorentz/".



- To generate the diagram in Fig. 10.1, for the coarser, upper
diagrams:
    >> python code/10_feec/weierstrass/large.py
and for the finer, lower diagrams:
    >> python code/10_feec/weierstrass/large.py

This does not require a Firedrake virtual environment.



- To generate the plots and data in Figs. 10.2 & 10.3, for the
structure-preserving integrator:
    >> python code/10_feec/vortex/avfet.py
for the MEEVC scheme:
    >> python code/10_feec/vortex/meevc.py
for the classical integrator:
    >> python code/10_feec/vortex/classical.py

Data and Paraview files will saved in each case to
"output/10_feec/".



----------



Thank you for your interest!
