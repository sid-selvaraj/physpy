"""
Example: compare a baseball's real trajectory (with air resistance)
against the idealized no-drag parabola from intro physics.

Run with:
    python examples/projectile_demo.py
"""

from physpy.mechanics.kinematics import Projectile

# A baseball: ~145 g, ~7.3 cm diameter, C_d ~0.47 for a smooth sphere.
baseball = Projectile(
    v0=30,               # ~67 mph
    angle_deg=45,
    mass=0.145,
    drag_coefficient=0.47,
    cross_section_area=0.0042,  # pi * (0.037 m)^2
)

sol = baseball.solve()
print(f"Range with drag:    {sol['x'][-1]:.2f} m")
print(f"Range without drag: {baseball.range_analytic():.2f} m")
print(f"Drag cost the hitter {baseball.range_analytic() - sol['x'][-1]:.2f} m of distance")

baseball.plot()
