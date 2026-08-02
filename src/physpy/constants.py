"""
physpy.constants
=================
Physical constants for use throughout PhysPy, in SI units.

We wrap ``scipy.constants`` rather than redefining values by hand. scipy's
constants are sourced from CODATA (the internationally recognized reference
for physical constants) and are kept up to date with each scipy release.
Wrapping them here does two things:

1. Gives physics-friendly names, so a user writes ``physpy.constants.hbar``
   instead of remembering scipy's naming conventions.
2. Gives PhysPy a single place to add *derived* constants later (Bohr
   radius, Rydberg energy, Compton wavelength, ...) without every module
   needing to import ``scipy.constants`` directly.
"""

try:
    from scipy import constants as _c  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "physpy.constants requires scipy. Install scipy to use this module."
    ) from exc

# --- Fundamental constants (SI units) ---
c_light = _c.c              # speed of light in vacuum, m/s
G_grav = _c.G                # Newtonian gravitational constant, m^3 kg^-1 s^-2
h_planck = _c.h              # Planck constant, J s
hbar = _c.hbar                # reduced Planck constant, J s
k_boltzmann = _c.k           # Boltzmann constant, J/K
e_charge = _c.e              # elementary charge, C
epsilon_0 = _c.epsilon_0      # vacuum permittivity, F/m
mu_0 = _c.mu_0                # vacuum permeability, N/A^2
N_avogadro = _c.N_A          # Avogadro's number, 1/mol

# --- Particle rest masses (kg) ---
m_electron = _c.m_e
m_proton = _c.m_p
m_neutron = _c.m_n

# --- Convenience / Earth-surface constants ---
g_earth = 9.80665  # standard gravity, m/s^2 (exact by international definition)
