import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Physical constants
# ------------------------------------------------------------
kB   = 1.380649e-23      # Boltzmann [J/K]
hbar = 1.054571817e-34   # reduced Planck [J s]
pi   = np.pi

# ------------------------------------------------------------
# Physical parameters
# ------------------------------------------------------------
omega_a = 2 * pi * 2e9      # absorber frequency (rad/s), 1 GHz
lam_hz  = 50e3                # nominal cross-Kerr in Hz
lam_eff_hz = 10e3             # effective cross-Kerr used in plot (adjustable)
lam = 2 * pi * lam_eff_hz     # effective cross-Kerr (rad/s)

tau  = 20e-6                  # interaction duration for each tooth [s]
tau_c = 20e-6                 # correlation time of Lorentzian noise [s]

# Temperature range [K]
T_min, T_max, nT = 0.01, 0.05, 500
# Delay range [s]
Delta_min, Delta_max, nD = 0.0, 80e-6, 300

T = np.linspace(T_min, T_max, nT)               # K
Delta = np.linspace(Delta_min, Delta_max, nD)   # s
TT, DD = np.meshgrid(T, Delta, indexing="ij")

# ------------------------------------------------------------
# Thermal occupation and variance of absorber mode
#   nbar(T) = 1 / (exp(hbar ω / kB T) - 1)
#   Var[n]  = nbar (1 + nbar)
# ------------------------------------------------------------
x = hbar * omega_a / (kB * TT)
nbar = 1.0 / (np.exp(x) - 1.0)
Var_n = nbar * (1.0 + nbar)

# ------------------------------------------------------------
# Lorentzian noise → exponential memory kernel in time domain:
#   S_nn(ω;T) ∝ Var_n(T) τ_c / [1 + (ω τ_c)^2]
#   K(Δ;T)    = Var_n(T) exp(-|Δ|/τ_c)
# ------------------------------------------------------------
K = Var_n * np.exp(-DD / tau_c)

# ------------------------------------------------------------
# Dephasing rates
#   Γ_φ1(T)          = λ² τ² Var_n(T)
#   Γ_φ2(T,Δ)        = λ² [ 2 τ² Var_n(T) + 2 τ² K(Δ;T) ]
# ------------------------------------------------------------
lam2 = lam**2
Gamma1 = lam2 * tau**2 * Var_n
Gamma2 = lam2 * (2.0 * tau**2 * Var_n + 2.0 * tau**2 * K)

# ------------------------------------------------------------
# Coherences and temperature derivatives
#   C = exp(-Γ)
#   F(T) = (∂_T C)^2 / (1 - C^2)
# ------------------------------------------------------------
C1 = np.exp(-Gamma1)   # shape (nT, nD) by broadcasting
C2 = np.exp(-Gamma2)

# derivatives along T axis
dC1_dT = np.gradient(C1, T, axis=0)
dC2_dT = np.gradient(C2, T, axis=0)

eps = 1e-14
C1_clip = np.clip(C1, -1 + eps, 1 - eps)
C2_clip = np.clip(C2, -1 + eps, 1 - eps)

F1 = (dC1_dT**2) / (1.0 - C1_clip**2 + eps)
F2 = (dC2_dT**2) / (1.0 - C2_clip**2 + eps)

# total single-tooth QFI for two identical interactions
F1_total = 2.0 * F1

# Memory advantage
A = F2 / (F1_total + eps)

# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
plt.rcParams["figure.figsize"] = (3.4, 5.0)
plt.rcParams["font.size"] = 9
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["xtick.major.width"] = 0.6
plt.rcParams["ytick.major.width"] = 0.6

fig, axes = plt.subplots(3, 1, figsize=(6, 8), constrained_layout=True)
ax_a, ax_b, ax_c = axes

# Convenience: convert Δ-axis to microseconds for plotting
Delta_us = Delta * 1e6
extent_TD = [T_min, T_max, Delta_min * 1e6, Delta_max * 1e6]

# (a) Two-tooth QFI F2(T, Δ)
im0 = ax_a.imshow(
    F2.T,
    origin="lower",
    extent=extent_TD,
    aspect="auto",
    cmap='plasma',
)
ax_a.set_ylabel(r'$\Delta~(\mu\mathrm{s})$')
ax_a.text(0.03, 0.90, r'(a)', transform=ax_a.transAxes, color='w')
# cbar0 = fig.colorbar(im0, ax=ax_a, pad=0.01)
# cbar0.set_label(r'$\mathcal{F}_2(T,\Delta)$')

# (b) Memory advantage A(T, Δ)
A_plot = np.clip(A, 0.0, 3.0)
im1 = ax_b.imshow(
    A_plot.T,
    origin="lower",
    extent=extent_TD,
    aspect="auto",
    cmap='plasma',
)
ax_b.set_ylabel(r'$\Delta~(\mu\mathrm{s})$')
ax_b.text(0.03, 0.90, r'(b)', transform=ax_b.transAxes, color='w')
# cbar1 = fig.colorbar(im1, ax=ax_b, pad=0.01)
# cbar1.set_label(r'$\mathcal{A}(T,\Delta)$')

# Contour where A = 1 (boundary of memory advantage)
ax_b.contour(
    T,
    Delta_us,
    A.T,
    levels=[1.0],
    colors='w',
    linestyles='--',
    linewidths=1.0,
)

# (c) Linecuts A(Δ) at a few temperatures
T_lines = [0.01, 0.025, 0.04]
labels = [
    r'$T=0.01~\mathrm{K}$ (slow absorber)',
    r'$T=0.02~\mathrm{K}$ (intermediate)',
    r'$T=0.04~\mathrm{K}$ (fast absorber)',
]

for T0, lab in zip(T_lines, labels):
    idx = np.argmin(np.abs(T - T0))
    ax_c.plot(Delta_us, A[idx, :], label=lab, lw=2.0)

ax_c.axhline(1.0, linestyle='--', linewidth=0.8, color='gray')
ax_c.set_xlim(0.0, 60.0)
ax_c.set_xlabel(r'$\Delta~(\mu\mathrm{s})$')
ax_c.set_ylabel(r'$\mathcal{A}(\Delta)$')
ax_c.text(0.03, 0.90, r'(c)', transform=ax_c.transAxes)
ax_c.legend(frameon=False, fontsize=7, loc='upper right')

plt.show()
