"""
Demo: Electric field visualization with point charges and dipoles.

Shows how to:
1. Create point charges
2. Visualize electric field and equipotential lines
3. Calculate field and potential at specific points
"""

from physpy.electromagnetism.statics import PointCharge, ElectricDipole, ElectricField

print("=" * 60)
print("ELECTRIC FIELD VISUALIZATION DEMO")
print("=" * 60)

# ------- Example 1: Two opposite point charges (dipole-like) -------
print("\n1. Two-charge configuration (dipole field)")
print("-" * 60)

q1 = PointCharge(q=1e-6, x=-0.05, y=0)  # 1 µC at x = -5 cm
q2 = PointCharge(q=-1e-6, x=0.05, y=0)  # -1 µC at x = +5 cm

field = ElectricField(charges=[q1, q2])

# Evaluate field at origin
Ex, Ey = field.electric_field(0, 0)
print(f"Field at origin (0, 0): Ex = {Ex:.2e} N/C, Ey = {Ey:.2e} N/C")

# Evaluate potential at origin
V = field.potential(0, 0)
print(f"Potential at origin (0, 0): V = {V:.2e} V")

# Evaluate field at (0, 0.1)
Ex, Ey = field.electric_field(0, 0.1)
print(f"Field at (0, 0.1 m): Ex = {Ex:.2e} N/C, Ey = {Ey:.2e} N/C")

print("\nGenerating field plot...")
field.plot(show=False)
print("✓ Field plot saved")

# ------- Example 2: Single point charge -------
print("\n2. Single point charge")
print("-" * 60)

positive_charge = PointCharge(q=2e-6, x=0, y=0)
field2 = ElectricField(charges=[positive_charge])

# Calculate field at different distances
distances = [0.01, 0.05, 0.1, 0.2]
print("Field magnitude at various distances from the charge:")
for r in distances:
    Ex, Ey = field2.electric_field(r, 0)
    magnitude = (Ex**2 + Ey**2)**0.5
    print(f"  r = {r:5.2f} m: E = {magnitude:.2e} N/C")

# ------- Example 3: Electric dipole -------
print("\n3. Dipole configuration")
print("-" * 60)

dipole = ElectricDipole(
    q=1e-6,
    x_pos=0.05, y_pos=0,    # positive charge at +5 cm
    x_neg=-0.05, y_neg=0,   # negative charge at -5 cm
)

print(f"Dipole moment: p = {dipole.dipole_moment:.2e} C·m")

field3 = ElectricField(dipoles=[dipole])
Ex, Ey = field3.electric_field(0, 0.1)
print(f"Field at (0, 0.1 m): Ex = {Ex:.2e} N/C, Ey = {Ey:.2e} N/C")

# ------- Example 4: Superposition with mixed charges -------
print("\n4. Superposition: mixed charges and dipoles")
print("-" * 60)

q_center = PointCharge(q=2e-6, x=0, y=0)
dipole2 = ElectricDipole(
    q=1e-6,
    x_pos=0.0, y_pos=0.1,
    x_neg=0.0, y_neg=-0.1,
)

field4 = ElectricField(charges=[q_center], dipoles=[dipole2])
print("Created field with 1 point charge and 1 dipole")

V = field4.potential(0.1, 0.1)
print(f"Potential at (0.1, 0.1) m: V = {V:.2e} V")

print("\n" + "=" * 60)
print("Demo complete!")
print("=" * 60)
