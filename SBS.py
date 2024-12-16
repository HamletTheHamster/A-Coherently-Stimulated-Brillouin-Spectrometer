import numpy as np
import scipy.constants as sci
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Parameters (as given)
n = 1.48
G_B = 0.6
Omega_B = 2*np.pi*9.18e9
lambda_P = 1.549e-6
k_B = 1.38e-23
T = 295
L_values = np.logspace(-8, 4.5, 10000000)
Gamma_B = 2*np.pi*80e6
P_P = 1
P_S = 1
P_Pr = 1
delta_lambda = 0.02e-9

sincArg = 2*np.pi*n*delta_lambda*L_values/(lambda_P)**2
sincSquared = (np.sin(sincArg)**2)/(sincArg**2)

omega_P = 2*np.pi*sci.c/(n*lambda_P)

# SBS and CoBS calculations
SBSPower = P_P*omega_P*G_B*k_B*T*L_values*Gamma_B/(4*Omega_B)
CoBSPower = 0.25*(G_B*L_values)**2 * P_P * P_S * P_Pr * sincSquared

# Compute local average for CoBS envelope
window_fraction = 0.01
sinc_avg_local = np.zeros_like(L_values)
for i, L in enumerate(L_values):
    delta_L = window_fraction * L
    mask = (L_values >= L - delta_L) & (L_values <= L + delta_L)
    sinc_avg_local[i] = np.mean(sincSquared[mask])

CoBS_Envelope = 0.25 * (G_B * L_values)**2 * P_P * P_S * P_Pr * sinc_avg_local

# Plot
plt.figure(figsize=(10, 6), dpi=150)

# Use loglog to set both scales to log
plt.loglog(L_values, CoBSPower, label="CoBS Scattered Power", linewidth=2, color="darkorange")
plt.loglog(L_values, SBSPower, label="SBS Scattered Power", linewidth=2, linestyle="--", color="blue")
plt.loglog(L_values, CoBS_Envelope, label="CoBS Envelope", linewidth=2, linestyle=":", color="red")

L_coherence = 9.55e3
plt.axvline(L_coherence, color='gray', linewidth=1, label="10 kHz Laser Coherence Length")

plt.xlabel("Length (m)", fontsize=12)
plt.ylabel("Scattered Power (W)", fontsize=12)
plt.title("Comparison of SBS and CoBS Scattered Power vs Length", fontsize=14)
plt.legend()


# Set up the major and minor tick locators.
# For major ticks, this will place ticks at powers of 10.
# For minor ticks, the 'subs' parameter puts ticks at specified intervals between powers of 10.
ax = plt.gca()

ax.set_xscale('log')
ax.set_yscale('log')

# Major ticks every power of 10
ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=50))
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=50))

# Minor ticks at 0.2, 0.4, 0.6, 0.8 between each major tick
ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=50))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=50))

# Remove minor tick labels
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.yaxis.set_minor_formatter(ticker.NullFormatter())

# Custom formatter for major ticks: only label every other power of ten
def every_other_label(value, pos):
    # value is the tick value in data coordinates
    # On a log scale, major ticks are at powers of 10: 10^..., so:
    if value <= 0:
        return ''  # No negative or zero ticks in log scale, just a safeguard

    exponent = np.log10(value)
    # Check if exponent is nearly an integer
    if np.isclose(exponent, np.round(exponent)):
        exponent = int(np.round(exponent))
        # Label only if exponent is even
        if exponent % 2 == 0:
            return f'$10^{{{exponent}}}$'  # Format as 10^{exponent}
        else:
            return ''  # No label for odd powers of 10
    else:
        return ''  # If not a neat power of 10, skip labeling

# Apply the custom formatter to major ticks
ax.xaxis.set_major_formatter(ticker.FuncFormatter(every_other_label))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(every_other_label))


# Add both major and minor grid
plt.grid(which='major', linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("plot.png", dpi=600, bbox_inches="tight")
plt.savefig("plot.svg")
plt.show()
