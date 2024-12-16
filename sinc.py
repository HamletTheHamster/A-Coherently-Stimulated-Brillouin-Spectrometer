# import numpy as np
# import matplotlib.pyplot as plt
#
# # Constants
# c = 3e8  # Speed of light in vacuum (m/s)
# lambda_pump = 1.55e-6  # Pump wavelength (m)
# L = 4  # Fiber length (m)
# n_eff = 1.4682  # Effective refractive index for SMF28 at 1.55 µm
# v_a = 5960  # Acoustic velocity in silica (m/s)
#
# # Calculate the Brillouin frequency shift
# nu_B = 2 * n_eff * v_a / lambda_pump  # Brillouin frequency (Hz)
#
# # Define detuning frequency range around the Brillouin frequency shift
# delta_nu = np.linspace(-250e6, 250e6, 1000)  # Detuning range (Hz)
#
# # Calculate phase mismatch Δk as a function of detuning
# delta_k = (4 * np.pi * n_eff / c) * delta_nu  # Phase mismatch (1/m)
#
# # Calculate the argument of the sinc function
# sinc_arg = delta_k * L / 2
#
# # Calculate the sinc^2 function (normalized)
# sinc_squared = (np.sinc(sinc_arg / np.pi))**2
#
# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(delta_nu / 1e6, sinc_squared, label='sinc$^2$ function')
# plt.xlabel('Pump-Probe Detuning Frequency (MHz)')
# plt.ylabel('sinc$^2$($\\Delta k L / 2$)')
# plt.title('sinc$^2$ Function vs. Pump-Probe Detuning Frequency for 4m SMF28 @1.55$\mu$m')
# plt.grid(True)
#
# # Find and mark the peaks of the lobes
# from scipy.signal import find_peaks
#
# peaks, _ = find_peaks(sinc_squared)
# peak_frequencies = delta_nu[peaks] / 1e6  # Convert to MHz
# peak_values = sinc_squared[peaks]
#
# # Mark the peaks on the plot
# plt.plot(peak_frequencies, peak_values, 'ro', label='Lobe Peaks')
# for freq, val in zip(peak_frequencies, peak_values):
#     if val != max(peak_values):
#       plt.annotate(f'{freq:.1f} MHz', xy=(freq, val), xytext=(freq, val + 0.05),
#                  textcoords='data', ha='center')
#
# plt.legend()
# plt.show()

################################################################################

# import numpy as np
# import matplotlib.pyplot as plt
#
# # Constants
# lambda_pump = 1.55e-6  # Pump wavelength (m)
# L = 4  # Fiber length (m)
# n_eff = 1.4682  # Effective refractive index for SMF28 at 1.55 µm
#
# # Define detuning wavelength range around the Brillouin frequency shift
# delta_wavelength = np.linspace(-40e-12, 40e-12, 1000)  # Detuning range (m)
#
# # Calculate the argument for the sinc function
# x = (2 * n_eff * delta_wavelength * L) / (lambda_pump**2)
#
# # Calculate sinc^2 function
# sinc_squared = np.sinc(x)**2
#
# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(delta_wavelength / 1e-12, sinc_squared, label='sinc$^2$ function')
# plt.xlabel('Pump-Probe Detuning Wavelength (pm)')
# plt.ylabel('sinc$^2$($\\Delta k L / 2$)')
# plt.title('sinc$^2$ Function vs. Pump-Probe Detuning Wavelength for 4m SMF28 @1.55$\\mu$m')
# plt.grid(True)
#
# # # Find and mark the peaks of the lobes
# # from scipy.signal import find_peaks
# #
# # peaks, _ = find_peaks(sinc_squared)
# # peak_wavelengths = delta_wavelength[peaks] / 1e-12  # Convert to pm
# # peak_values = sinc_squared[peaks]
# #
# # # Mark the peaks on the plot
# # plt.plot(peak_wavelengths, peak_values, 'ro', label='Lobe Peaks')
# # for wl, val in zip(peak_wavelengths, peak_values):
# #     if val != max(peak_values):
# #         plt.annotate(f'{wl:.1f} pm', xy=(wl, val), xytext=(wl, val + 0.05),
# #                      textcoords='data', ha='center')
#
# plt.legend()
# plt.show()
#
# ################################################################################
#
# # Constants
# lambda_pump = 1.55e-6  # Pump wavelength (m)
# L = .01  # Fiber length (m)
# n_eff = 1.4682  # Effective refractive index for SMF28 at 1.55 µm
#
# # Define detuning wavelength range around the Brillouin frequency shift
# delta_wavelength = np.linspace(-40e-12, 40e-12, 1000)  # Detuning range (m)
#
# # Calculate the argument for the sinc function
# x = (2 * n_eff * delta_wavelength * L) / (lambda_pump**2)
#
# # Calculate sinc^2 function
# sinc_squared = np.sinc(x)**2
#
# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(delta_wavelength / 1e-12, sinc_squared, label='sinc$^2$ function')
# plt.xlabel('Pump-Probe Detuning Wavelength (pm)')
# plt.ylabel('sinc$^2$($\\Delta k L / 2$)')
# plt.title('sinc$^2$ Function vs. Pump-Probe Detuning Wavelength for 1cm waveguide @1.55$\\mu$m')
# plt.grid(True)
#
# # # Find and mark the peaks of the lobes
# # from scipy.signal import find_peaks
# #
# # peaks, _ = find_peaks(sinc_squared)
# # peak_wavelengths = delta_wavelength[peaks] / 1e-12  # Convert to pm
# # peak_values = sinc_squared[peaks]
# #
# # # Mark the peaks on the plot
# # plt.plot(peak_wavelengths, peak_values, 'ro', label='Lobe Peaks')
# # for wl, val in zip(peak_wavelengths, peak_values):
# #     if val != max(peak_values):
# #         plt.annotate(f'{wl:.1f} pm', xy=(wl, val), xytext=(wl, val + 0.05),
# #                      textcoords='data', ha='center')
#
# plt.legend()
# plt.show()

#############################

import numpy as np
import matplotlib.pyplot as plt

# Constants
lambda_pump = 1.55e-6  # Pump wavelength (m)
n_eff = 1.4682  # Effective refractive index for SMF28 at 1.55 µm

# Define detuning wavelength range around the Brillouin frequency shift
delta_wavelength = np.linspace(40e-12, 80e-12, 1000)  # Detuning range (m)

# Calculate sinc^2 for L = 4m
L1 = 4  # Fiber length (m)
x1 = (2 * n_eff * delta_wavelength * L1) / (lambda_pump**2)
sinc_squared_1 = np.sinc(x1)**2

# Calculate sinc^2 for L = 0.01m (1cm)
L2 = 0.01  # Fiber length (m)
x2 = (2 * n_eff * delta_wavelength * L2) / (lambda_pump**2)
sinc_squared_2 = np.sinc(x2)**2

# Plotting both on the same plot
plt.figure(figsize=(10, 6))
plt.plot(delta_wavelength / 1e-12, 200000*sinc_squared_1, label='4m SMF28')
plt.plot(delta_wavelength / 1e-12, sinc_squared_2, label='1cm waveguide')

# Labels and title
plt.xlabel('Pump-Probe Detuning Wavelength (pm)')
plt.ylabel('sinc$^2$($\\Delta k L / 2$)')
plt.title('sinc$^2$ Function vs. Pump-Probe Detuning Wavelength')
plt.grid(True)
plt.legend()
plt.show()

######################

# import numpy as np
# import matplotlib.pyplot as plt
#
# # Constants
# c = 3e8  # Speed of light in vacuum (m/s)
# n_eff = 1.4682  # Effective refractive index for silica at 1.55 µm
# L1 = 4      # Fiber length for SMF28 (m)
# L2 = 0.01   # Waveguide length (m)
#
# # Frequency detuning range from 5 GHz to 10 GHz
# delta_frequency = np.linspace(5e9, 10e9, 1000)  # Frequency detuning range (Hz)
#
# # Calculate x1 and x2 in radians
# x1 = (2 * np.pi * n_eff * delta_frequency * L1) / (2 * c)
# x2 = (2 * np.pi * n_eff * delta_frequency * L2) / (2 * c)
#
# # Avoid division by zero at x=0
# x1_nonzero = np.where(x1 == 0, 1e-20, x1)
# x2_nonzero = np.where(x2 == 0, 1e-20, x2)
#
# # Calculate sinc^2 for both lengths
# sinc_squared_1 = (np.sin(x1_nonzero) / x1_nonzero)**2
# sinc_squared_2 = (np.sin(x2_nonzero) / x2_nonzero)**2
#
# # Convert frequency to GHz for plotting
# delta_frequency_GHz = delta_frequency / 1e9
#
# # Plotting both on the same plot
# plt.figure(figsize=(10, 6))
# plt.plot(delta_frequency_GHz, 200000*sinc_squared_1, label='4 m SMF28')
# plt.plot(delta_frequency_GHz, sinc_squared_2, label='1 cm Waveguide')
#
# # Labels and title
# plt.xlabel('Pump-Probe Frequency Detuning (GHz)')
# plt.ylabel('sinc$^2$($\\Delta k L / 2$)')
# plt.title('sinc$^2$ Function vs. Pump-Probe Frequency Detuning')
# plt.grid(True)
# plt.legend()
# plt.show()

######################
