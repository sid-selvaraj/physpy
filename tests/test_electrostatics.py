"""Tests for physpy.electromagnetism.statics.

Validates electric field calculations against Coulomb's law and
dipole field theory.
"""

import pytest
import numpy as np

from physpy.electromagnetism.statics import PointCharge, ElectricDipole, ElectricField
from physpy.constants import epsilon_0


def test_point_charge_field_magnitude():
    """Field magnitude obeys Coulomb's law: E = k*q/r^2."""
    q = 1e-6  # 1 microcoulomb
    charge = PointCharge(q=q, x=0, y=0)

    # Test at r=1 meter along x-axis
    Ex, Ey = charge.electric_field(1.0, 0.0)
    k = 1 / (4 * np.pi * epsilon_0)
    expected_E = k * q / 1.0**2
    assert np.isclose(Ex, expected_E, rtol=1e-6)
    assert np.isclose(Ey, 0.0, atol=1e-12)


def test_point_charge_field_direction():
    """Field points radially outward from positive charge."""
    charge = PointCharge(q=1e-6, x=0, y=0)

    # Test at (1, 1) — should point diagonally outward
    Ex, Ey = charge.electric_field(1.0, 1.0)
    assert Ex > 0 and Ey > 0
    assert np.isclose(Ex, Ey, rtol=1e-6)


def test_negative_charge_field_direction():
    """Field points radially inward toward negative charge."""
    charge = PointCharge(q=-1e-6, x=0, y=0)

    Ex, Ey = charge.electric_field(1.0, 0.0)
    assert Ex < 0
    assert Ey == 0


def test_point_charge_potential():
    """Potential V = k*q/r."""
    q = 1e-6
    charge = PointCharge(q=q, x=0, y=0)

    k = 1 / (4 * np.pi * epsilon_0)
    V = charge.potential(1.0, 0.0)
    expected_V = k * q / 1.0
    assert np.isclose(V, expected_V, rtol=1e-6)


def test_dipole_moment():
    """Dipole moment p = q * d where d is separation."""
    q = 1e-6
    separation = 0.1
    dipole = ElectricDipole(
        q=q,
        x_pos=0.05, y_pos=0,
        x_neg=-0.05, y_neg=0
    )
    expected_p = q * separation
    assert np.isclose(dipole.dipole_moment, expected_p, rtol=1e-6)


def test_dipole_field_decay():
    """Dipole field decays faster than monopole field at large distances."""
    charge = 1e-6
    dipole = ElectricDipole(
        q=charge,
        x_pos=0.05, y_pos=0,
        x_neg=-0.05, y_neg=0
    )

    # Single charge (monopole) for comparison
    q_mono = PointCharge(q=charge, x=0.05, y=0)

    # At large distance along the dipole axis, dipole field ~ 1/r^3, monopole ~ 1/r^2
    # So dipole should fall off faster
    r_test = 0.5  # Large distance compared to dipole separation

    E_mono_mag, _ = q_mono.electric_field(r_test, 0)
    E_dip_x, E_dip_y = dipole.electric_field(r_test, 0)
    E_dip_mag = np.sqrt(E_dip_x**2 + E_dip_y**2)

    # Dipole field should be much smaller than monopole at large distance
    assert E_dip_mag < abs(E_mono_mag)


def test_electric_field_superposition():
    """Multiple charges: field is sum of individual fields."""
    q1 = PointCharge(q=1e-6, x=-0.05, y=0)
    q2 = PointCharge(q=-1e-6, x=0.05, y=0)

    # Individual fields
    E1x, E1y = q1.electric_field(0, 0.1)
    E2x, E2y = q2.electric_field(0, 0.1)

    # Superposition field
    field = ElectricField(charges=[q1, q2])
    Etx, Ety = field.electric_field(0, 0.1)

    assert np.isclose(Etx, E1x + E2x, rtol=1e-6)
    assert np.isclose(Ety, E1y + E2y, rtol=1e-6)


def test_electric_field_potential_superposition():
    """Multiple charges: potential is sum of individual potentials."""
    q1 = PointCharge(q=1e-6, x=-0.05, y=0)
    q2 = PointCharge(q=-1e-6, x=0.05, y=0)

    V1 = q1.potential(0, 0.1)
    V2 = q2.potential(0, 0.1)

    field = ElectricField(charges=[q1, q2])
    Vt = field.potential(0, 0.1)

    assert np.isclose(Vt, V1 + V2, rtol=1e-6)


def test_field_grid_shape():
    """Field grid has correct shape."""
    charge = PointCharge(q=1e-6, x=0, y=0)
    field = ElectricField(charges=[charge])

    X, Y, Ex, Ey = field.field_grid(resolution=20)
    assert X.shape == (20, 20)
    assert Y.shape == (20, 20)
    assert Ex.shape == (20, 20)
    assert Ey.shape == (20, 20)


def test_potential_grid_shape():
    """Potential grid has correct shape."""
    charge = PointCharge(q=1e-6, x=0, y=0)
    field = ElectricField(charges=[charge])

    X, Y, V = field.potential_grid(resolution=25)
    assert X.shape == (25, 25)
    assert Y.shape == (25, 25)
    assert V.shape == (25, 25)
