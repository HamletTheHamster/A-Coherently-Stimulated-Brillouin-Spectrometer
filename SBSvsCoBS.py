# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sci

# Constants and parameters
n = 1.48  # Refractive index (UHNA3: behunin2016long-lived)
G_B = 0.6  # Brillouin gain coefficient (UHNA3)
Omega_B = 2 * np.pi * 9.18e9  # Brillouin frequency (rad/s)
lambda_P = 1.549e-6  # Pump wavelength (m)
k_B = 1.38e-23  # Boltzmann constant (J/K)
T = 295  # Temperature (K)
Gamma_B = 2 * np.pi * 80e6  # Acoustic damping rate (rad/s)
P_P = 1  # Pump power (W)
P_S = 1  # Stokes power (W)
P_Pr = 1  # Probe power (W)
delta_lambda = 0.02e-9  # Wavelength detuning (m)

# Length scales to explore (10 nm to 1 km)
L_values = np.logspace(-8, 3, 100)  # From 10 nm (1e-8 m) to 1 km (1e3 m)

# Pre-calculations
sincArg = 2 * np.pi * n * delta_lambda * L_values / (lambda_P) ** 2
sincSquared = np.sinc(sincArg / np.pi) ** 2  # Using np.sinc for normalized sinc

omega_P = 2 * np.pi * sci.c / (n * lambda_P)  # Optical frequency

# Calculating SBS scattered power (Boyd/Kahrel 2016)
SBSPower = P_P * omega_P * G_B * k_B * T * L_values * Gamma_B / (4 * Omega_B)

# Calculating CoBS scattered power
CoBSPower = 0.25 * (G_B * L_values) ** 2 * P_P * P_S * P_Pr * sincSquared

# Plotting results
plt.figure(figsize=(10, 6))
plt.loglog(L_values, SBSPower, label="SBS Scattered Power", linewidth=2)
plt.loglog(L_values, CoBSPower, label="CoBS Scattered Power", linewidth=2, linestyle="--")

# Adding labels and legend
plt.xlabel("Length (m)", fontsize=12)
plt.ylabel("Scattered Power (W)", fontsize=12)
plt.title("Comparison of SBS and CoBS Scattered Power vs Length", fontsize=14)
plt.legend()
plt.grid(which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
