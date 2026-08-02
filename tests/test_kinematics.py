"""Tests for physpy.mechanics.kinematics.

Each test validates the code against something independently verifiable:
a hand-derivable value, a cross-check between two formulas, agreement
between the numerical and analytic solvers, or a physical law that must
hold regardless of the exact numbers involved.
"""

import pytest

from physpy.mechanics.kinematics import Projectile


def test_range_analytic_45_degrees():
    """At 45 degrees (no drag), range simplifies to v0^2 / g."""
    p = Projectile(v0=20, angle_deg=45)
    expected_range = 20 ** 2 / p.gravity
    assert p.range_analytic() == pytest.approx(expected_range, rel=1e-6)


def test_max_height_matches_kinematics():
    """v_y^2 = v_y0^2 - 2 g h  =>  h_max = v_y0^2 / (2g)."""
    p = Projectile(v0=15, angle_deg=30)
    expected_h = p.vy0 ** 2 / (2 * p.gravity)
    assert p.max_height_analytic() == pytest.approx(expected_h, rel=1e-6)


def test_numerical_matches_analytic_when_no_drag():
    """With drag_coefficient=0, the ODE solver must reproduce the
    textbook analytic range — this is the core correctness check."""
    p = Projectile(v0=25, angle_deg=40, drag_coefficient=0.0)
    sol = p.solve()
    numerical_range = sol["x"][-1]
    assert numerical_range == pytest.approx(p.range_analytic(), rel=1e-2)


def test_drag_reduces_range():
    """Physical law, not a specific number: adding air resistance must
    always reduce range relative to the no-drag case, for identical
    launch conditions."""
    no_drag = Projectile(v0=30, angle_deg=45, drag_coefficient=0.0)
    with_drag = Projectile(
        v0=30, angle_deg=45,
        drag_coefficient=0.47, cross_section_area=0.01, mass=0.145,  # baseball-ish
    )

    no_drag.solve()
    with_drag.solve()

    assert with_drag._solution["x"][-1] < no_drag._solution["x"][-1]


def test_zero_angle_launch():
    """A launch angle of 0 degrees should have zero max height."""
    p = Projectile(v0=10, angle_deg=0)
    assert p.max_height_analytic() == pytest.approx(0.0, abs=1e-9)
