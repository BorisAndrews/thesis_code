import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colours



'''
Set-up
'''
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.latex.preamble"] = r"\usepackage{palatino}"
plt.rcParams["font.size"] = 22




'''
Parameters
'''
N = 2**5  # Number of poles in Weierstrass approximation
res = 2**10  # Image resolution
alpha = 0.2  # Checker colouring
checker_size = 0.5  # Checker size
stripe_width = 0.1  # Stripe width
threshold = 2  # Threshold for colouring roots/poles
c = 0.6 + 1j * 0.7  # Location of vortex



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



# Complex plotter
def complex_plot(W, contour=False, box_color=False, title=False, poles=(-1, 1)):
    if contour:
        # Stripe pattern based on real part
        stripes = ((np.floor(np.real(W) / stripe_width)) % 2) == 0
        RGB = 0.7 * np.ones((*stripes.shape, 3))  # White background
        RGB[stripes] = 0  # Black stripes

        # Apply red where |Re(z)| > threshold
        RGB[np.real(W) < poles[0] * threshold] = [0.7686274509803922, 0.3058823529411765, 0.321568627450980]
        RGB[np.real(W) > poles[1] * threshold] = [0.3921568627450980, 0.7098039215686275, 0.803921568627451]
    else:
        # Domain colouring
        arg = np.angle(W)
        mod = np.abs(W)

        # HSV: hue from arg, saturation = 0.95, value from mod
        hue = (arg + np.pi) / (2 * np.pi)
        sat = 0.95 * np.ones_like(hue)
        val = 1 / (1 + mod/5)
        RGB = colours.hsv_to_rgb(np.stack((hue, sat, val), axis=-1))

        # Checkerboard overlay
        ReW = np.real(W); ImW = np.imag(W)
        checker = ((np.floor(ReW / checker_size) + np.floor(ImW / checker_size)) % 2) == 0
        overlay = np.ones_like(RGB)
        overlay[checker] = 0
        RGB = (1 - alpha) * RGB + alpha * overlay

        # Apply white where |z| < exp(- threshold) and black where |z| > exp(threshold)
        RGB[np.abs(W) < np.exp(poles[0] * threshold)] = [0.9, 0.9, 0.9]
        RGB[np.abs(W) > np.exp(poles[1] * threshold)] = [0.0, 0.0, 0.0]

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(RGB, extent=(x[0], x[-1], y[-1], y[0]))
    if box_color:
        plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=box_color, linewidth=2, linestyle='--')
    if title:
        plt.title(title)
    # plt.xlabel("Re(z)"); plt.ylabel("Im(z)")
    plt.grid(False)
    plt.show()



'''
Setting up domain
'''
x = np.linspace(-0.0, 1.0, res); y = np.linspace(1.0, -0.0, res)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y



'''
Plotting
'''
# Regular Weierstrass function
W = weierstrass_wp(Z)
complex_plot(W, box_color='white', poles=(-1.8, 2.3))

# Shifted Weierstrass function
wp_c = weierstrass_wp(c)
complex_plot(W - wp_c, box_color='white', poles=(-0.6, 2.3))

# Log Weierstrass function
wp_c = weierstrass_wp(c)
complex_plot(np.log(W - wp_c), contour=True, box_color='white', poles=(-0.6, 2.3))
complex_plot(- np.log(W - np.conj(wp_c)), contour=True, box_color='white', poles=(-2.3, 0.6))

# Log Weierstrass function with complement
complex_plot(np.log(W - wp_c) - np.log(W - np.conj(wp_c)), contour=True, box_color='white', poles=(-0.7, 0.7))
