# Code for "Geometric numerical integration via auxiliary variables"

This repository contains the code used to generate the numerical results in the thesis  _**Geometric numerical integration via auxiliary variables**_ by **Boris Andrews**.



## Installation

The custom Python packages (in `avfet_modules/`) can be installed from the root directory:
```
>> pip install .
```
To uninstall:
```
>> pip uninstall avfet_modules
```



## Reproducing figures

The following instructions describe how to reproduce the figures in the thesis.
Commands marked with `>>` should be run in a shell from the root directory;
unless otherwise stated, this must be done in a Firedrake virtual environment.

The code run can be located in the `code/` directory, while output will be saved to a
created `output/` directory.



### Chapter 3 – *The general framework*

**Figs. 3.1 & 3.2**

Helicity-stable case:
```
>> mkdir -p output/3_framework/helicity/
>> python code/3_framework/helicity.py
```
Non-helicity-stable case:
```
>> mkdir -p output/3_framework/no_helicity/
>> python code/3_framework/no_helicity.py
```
Data and Paraview files will be saved in `output/3_framework/`.



### Chapter 6 – *ODEs*

**Figs. 6.1 & 6.2**

To generate the plots and data for each scheme:
```
>> python code/6_ode/kepler/comparison.py --scheme implicit_midpoint
>> python code/6_ode/kepler/comparison.py --scheme cohen_hairer
>> python code/6_ode/kepler/comparison.py --scheme labudde_greenspan
>> python code/6_ode/kepler/comparison.py --scheme andrews_farrell
```
These do not require a Firedrake virtual environment, however it does require a PETSc installation which can be fetched through Firedrake.
A Firedrake virtual environment is therefore sufficient.



**Fig. 6.3**

```
>> mkdir -p output/6_ode/kepler/convergence/
>> python code/6_ode/kepler/convergence/looper.py
```

Data outputs will be saved in `output/6_ode/kepler/`.



**Figs. 6.4 & 6.5**

```
>> mkdir -p output/6_ode/kovalevskaya/im/ output/6_ode/kovalevskaya/avfet/
>> matlab -batch "run('code/6_ode/kovalevskaya.mlx')"
```
Images will be saved in `output/6_pde/kovalevskaya/`.
Naturally, this does
not require a Firedrake virtual environment.



### Chapter 7 – *PDEs*

**Figs. 7.1, 7.2 & 7.3**

For the energy-conserving integrator:
```
>> mkdir -p output/7_pde/bbm/avfet/
>> python code/7_pde/bbm/avfet.py
```
For the Gauss method:
```
>> mkdir -p output/7_pde/bbm/gauss/
>> python code/7_pde/bbm/gauss.py
```
Plots and data will be saved in `output/7_pde/bbm/`.

Files are available to produce an animation of the data.
For the energy-conserving integrator:
```
>> python code/7_pde/bbm/animation.py --dir output/7_pde/bbm/avfet/
```
For the Gauss method:
```
>> python code/7_pde/bbm/animation.py --dir output/7_pde/bbm/gauss/
```
The animation can be viewed through a frame of reference moving at the exact speed of the continuous soliton by appending:
```
>> * --cam_speed 1.618034
```
For example:
```
>> python code/7_pde/bbm/animation.py --dir output/7_pde/bbm/avfet/ --cam_speed 1.618034
```


**Figs. 7.4 & 7.5**

For the structure-preserving integrator:
```
>> python code/7_pde/compressible_ns/shockwave/avfet.py
```
For the implicit midpoint method:
```
>> python code/7_pde/compressible_ns/shockwave/im.py
```
Data and Paraview files will saved to `output/7_pde/compressible_ns/shockwave/`.



**Fig. 7.6**

For the structure-preserving integrator:
```
>> python code/7_pde/compressible_ns/euler/avfet.py
```
For the implicit midpoint method:
```
>> python code/7_pde/compressible_ns/euler/im.py
```
Data and Paraview files will saved to `output/7_pde/compressible_ns/euler/`.



### Chapter 9 – *The Lorentz problem*

**Figs. 9.2 & 9.3**

For the structure-preserving integrator:
```
>> mkdir -p output/9_lorentz/avfet/
>> python code/9_lorentz/avfet.py
```
For the implicit midpoint method:
```
>> mkdir -p output/9_lorentz/im/
>> python code/9_lorentz/im.py
```
Data will saved to `output/9_lorentz/`.



### Chapter 10 – *Simplification of discretisations through FEEC*

**Fig. 10.1**

For the coarser, upper diagrams:
```
>> python code/10_feec/weierstrass/large.py
```
For the finer, lower diagrams:
```
>> python code/10_feec/weierstrass/large.py
```
This does not require a Firedrake virtual environment.



**Fig. 10.2 & 10.3**

```
>> python code/10_feec/vortex/avfet.py
>> python code/10_feec/vortex/meevc.py
>> python code/10_feec/vortex/classical.py
```
Data and Paraview files will saved to `output/10_feec/`.



---



*Thank you for your interest!*