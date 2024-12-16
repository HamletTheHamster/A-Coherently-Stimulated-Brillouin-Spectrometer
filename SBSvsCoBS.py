import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# Parameters
n = 1.48
G_B = 0.6
Omega_B = 2 * np.pi * 9.18e9
lambda_P = 1.549e-6
k_B = 1.38e-23
T = 295
Gamma_B = 2 * np.pi * 80e6
P_P = 1
P_S = 1
P_Pr = 1
delta_lambda = 0.02e-9

# Split L range into two sections
L_split = 1e-2  # Split point where oscillations begin (you can tune this)

# Define sparse sampling for the first half
L_first = np.logspace(-8, np.log10(L_split), 1000)  # 100 points in the first half

# Define dense sampling for the second half
L_second = np.logspace(np.log10(L_split), 4.5, 500000)  # 5 million points in the second half

# Combine the two ranges into one array
L_values = np.concatenate([L_first, L_second])

sincArg = 2 * np.pi * n * delta_lambda * L_values / (lambda_P) ** 2
sincSquared = np.sinc(sincArg / np.pi) ** 2

omega_P = 2 * np.pi * 3e8 / (n * lambda_P)

# SBS and CoBS calculations
SBSPower = P_P * omega_P * G_B * k_B * T * L_values * Gamma_B / (4 * Omega_B)
CoBSPower = 0.25 * (G_B * L_values) ** 2 * P_P * P_S * P_Pr * sincSquared

# Compute local average for CoBS envelope
window_fraction = 0.01

# Function to compute averages for a batch
def compute_local_avg_for_batch(L_batch, sincSquared, L_values, window_fraction, update_step=1000):
    results = []
    with tqdm(
        total=len(L_batch),
        desc="Processing L batch",
        ncols=80,
        leave=False,
        bar_format="{l_bar}{bar} {elapsed} Remaining: {remaining}"
    ) as inner_pbar:
        for i, L in enumerate(L_batch):
            delta_L = window_fraction * L
            mask = (L_values >= L - delta_L) & (L_values <= L + delta_L)
            results.append(np.mean(sincSquared[mask]))
            if i % update_step == 0:  # Update progress bar every `update_step` iterations
                inner_pbar.update(update_step)
        inner_pbar.update(len(L_batch) % update_step)  # Final update for leftover iterations
    return results

# Create outer progress bar
def parallel_with_progress(L_batches, sincSquared, L_values, window_fraction):
    results = []
    with Parallel(n_jobs=-1, backend="loky") as parallel:  # Explicit cleanup
        with tqdm(
            total=len(L_batches),
            desc="Processing Batches",
            ncols=80,
            leave=False,
            bar_format="{l_bar}{bar}"
        ) as outer_pbar:
            for result in Parallel(n_jobs=-1, backend="loky")(
                delayed(compute_local_avg_for_batch)(
                    L_batch, sincSquared, L_values, window_fraction
                )
                for L_batch in L_batches
            ):
                results.extend(result)
                outer_pbar.update(1)
    return results

# Split L_values into chunks
num_batches = 8
L_batches = np.array_split(L_values, num_batches)

# Run computation with progress bar
sinc_avg_local = parallel_with_progress(L_batches, sincSquared, L_values, window_fraction)

CoBS_Envelope = 0.25 * (G_B * L_values) ** 2 * P_P * P_S * P_Pr * sinc_avg_local

# Plot
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure(figsize=(10, 6), dpi=150)

plt.loglog(
    L_values,
    CoBSPower,
    label="CoBS Scattered Power",
    linewidth=2,
    color="darkorange",
)
plt.loglog(
    L_values,
    SBSPower,
    label="SBS Scattered Power",
    linewidth=2,
    linestyle="--",
    color="blue",
)
plt.loglog(
    L_values,
    CoBS_Envelope,
    label="CoBS Envelope",
    linewidth=2,
    linestyle=":",
    color="red",
)

L_coherence = 9.55e3
plt.axvline(
    L_coherence,
    color="gray",
    linewidth=2,
    label="10 kHz Laser Coherence Length"
)

plt.xlabel("Length (m)", fontsize=12)
plt.ylabel("Scattered Power (W)", fontsize=12)
plt.title("Comparison of SBS and CoBS Scattered Power vs Length", fontsize=14)
plt.legend()

ax = plt.gca()

ax.set_xscale("log")
ax.set_yscale("log")

ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=50))
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=50))

ax.xaxis.set_minor_locator(
    ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=50)
)
ax.yaxis.set_minor_locator(
    ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=50)
)

ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.yaxis.set_minor_formatter(ticker.NullFormatter())

def every_other_label(value, pos):
    if value <= 0:
        return ""

    exponent = np.log10(value)
    if np.isclose(exponent, np.round(exponent)):
        exponent = int(np.round(exponent))
        if exponent % 2 == 0:
            return f"$10^{{{exponent}}}$"
        else:
            return ""
    else:
        return ""

ax.xaxis.set_major_formatter(ticker.FuncFormatter(every_other_label))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(every_other_label))

plt.grid(which="major", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("plot.png", dpi=600, bbox_inches="tight")
plt.savefig("plot.svg")
