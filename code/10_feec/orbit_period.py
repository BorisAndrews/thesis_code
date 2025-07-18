import numpy as np
import avfet_modules.timestepping as timestepping
import matplotlib.pyplot as plt




'''
Parameters
'''
N = 2**7  # Number of poles in Weierstrass approximation
x_0 = (0.381966, 0.763932)  # Initial location of vortex
dt = 0.0001  # Timestep
dur = 0.342  # Duration (0.340 -- 0.341)




'''
Functions
'''
# Weierstrass approximation
def weierstrass_wp(z):
    z = np.mod(z.real, 2) + 1j * np.mod(z.imag, 2)  # Fold into the unit cell
    z = np.where(np.abs(z) < 1e-12, 1e-12, z)  # Avoid division by zero
    result = 1 / z**2
    for m in range(- N, N + 2):
        for n in range(- N, N + 2):
            if m == 0 and n == 0:
                continue
            omega = 2 * m + 1j * 2 * n
            result += (1 / (z - omega)**2) - (1 / omega**2)
    return result

# Weierstrass derivative
def weierstrass_wp_prime(z):
    z = np.mod(z.real, 2) + 1j * np.mod(z.imag, 2)  # Fold into the unit cell
    z = np.where(np.abs(z) < 1e-12, 1e-12, z)  # Avoid division by zero
    result = - 2 / z**3
    for m in range(- N, N + 2):
        for n in range(- N, N + 2):
            if m == 0 and n == 0:
                continue
            omega = 2 * m + 1j * 2 * n
            result += - 2 / (z - omega)**3
    return result

# Weierstrass second derivative
def weierstrass_wp_prime_prime(z):
    z = np.mod(z.real, 2) + 1j * np.mod(z.imag, 2)  # Fold into the unit cell
    z = np.where(np.abs(z) < 1e-12, 1e-12, z)  # Avoid division by zero
    result = 6 / z**4
    for m in range(- N, N + 2):
        for n in range(- N, N + 2):
            if m == 0 and n == 0:
                continue
            omega = 2 * m + 1j * 2 * n
            result += 6 / (z - omega)**4
    return result



# Vortex velocity
def velocity(x):
    z = x[0] + 1j * x[1]
    wp_z = weierstrass_wp(z)
    wp_z_prime = weierstrass_wp_prime(z)
    wp_z_prime_prime = weierstrass_wp_prime_prime(z)
    velocity =  - (wp_z_prime_prime / wp_z_prime + 1j * wp_z_prime / wp_z.imag) / 2
    return (velocity.imag, velocity.real)



# Timestepper
def rk4_timestepper(x_start):
    return timestepping.schemes.rk4(
        x_start,
        dt,
        velocity
    )



'''
Solve loop
'''
# print(velocity(x_0))
x = timestepping.solve_loop(
    np.array(x_0),
    rk4_timestepper,
    round(dur/dt)
)
print(x)
