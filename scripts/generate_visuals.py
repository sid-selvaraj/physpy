"""
Generate visualization images for README documentation.
Produces PNG images of projectile trajectories and electric field plots.
"""

import matplotlib.pyplot as plt
import numpy as np
from physpy.mechanics.kinematics import Projectile
from physpy.electromagnetism.statics import PointCharge, ElectricField

# Set style for better visuals
plt.style.use('default')

# 1. Projectile Motion: Effect of Air Drag
fig, ax = plt.subplots(figsize=(10, 6))

# Projectile with drag (baseball)
p_drag = Projectile(
    v0=30,
    angle_deg=45,
    mass=0.145,
    drag_coefficient=0.47,
    cross_section_area=0.0042,
)
sol_drag = p_drag.solve()

# Projectile without drag
p_no_drag = Projectile(v0=30, angle_deg=45, drag_coefficient=0.0)
sol_no_drag = p_no_drag.solve()

ax.plot(sol_no_drag['x'], sol_no_drag['y'], '--', linewidth=2.5,
        label='No Air Resistance (Analytic)', color='#2E86AB', alpha=0.8)
ax.plot(sol_drag['x'], sol_drag['y'], linewidth=2.5,
        label='With Air Drag (Realistic)', color='#A23B72')
ax.fill_between(sol_drag['x'], 0, sol_drag['y'], alpha=0.1, color='#A23B72')

ax.set_xlabel('Horizontal Distance (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Height (m)', fontsize=12, fontweight='bold')
ax.set_title('Projectile Motion: Impact of Air Resistance', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(alpha=0.3, linestyle='--')
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)

# Add annotation
ax.annotate(f'Range loss: {sol_no_drag["x"][-1] - sol_drag["x"][-1]:.1f} m',
            xy=(sol_drag['x'][-1], 0), xytext=(sol_drag['x'][-1]-20, 8),
            arrowprops=dict(arrowstyle='->', color='#A23B72', lw=1.5),
            fontsize=10, color='#A23B72', fontweight='bold')

plt.tight_layout()
plt.savefig('docs/projectile_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: docs/projectile_comparison.png")
plt.close()

# 2. Electric Field Visualization: Dipole Configuration
fig, ax = plt.subplots(figsize=(9, 8))

q1 = PointCharge(q=1e-6, x=-0.05, y=0)
q2 = PointCharge(q=-1e-6, x=0.05, y=0)
field = ElectricField(charges=[q1, q2])

# Plot equipotential lines
X, Y, V = field.potential_grid(x_range=(-0.12, 0.12), y_range=(-0.12, 0.12), resolution=100)
levels = np.linspace(-8e4, 8e4, 20)
cs = ax.contour(X, Y, V, levels=levels, colors='gray', alpha=0.4, linewidths=0.8)
ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0e')

# Plot field vectors
X_vec, Y_vec, Ex, Ey = field.field_grid(x_range=(-0.12, 0.12), y_range=(-0.12, 0.12), resolution=16)
magnitude = np.sqrt(Ex**2 + Ey**2)
magnitude[magnitude == 0] = 1
ax.quiver(X_vec, Y_vec, Ex/magnitude, Ey/magnitude, magnitude,
          cmap='hot', scale=25, width=0.004)

# Plot charges
ax.plot(q1.x, q1.y, 'o', color='#1f77b4', markersize=12, label='−1 μC', zorder=5)
ax.plot(q2.x, q2.y, 'o', color='#ff7f0e', markersize=12, label='+1 μC', zorder=5)

ax.set_xlabel('x (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('y (m)', fontsize=12, fontweight='bold')
ax.set_title('Electric Dipole Field & Equipotential Lines', fontsize=14, fontweight='bold')
ax.set_aspect('equal')
ax.legend(fontsize=11, loc='upper right')
ax.grid(alpha=0.2, linestyle=':')

plt.tight_layout()
plt.savefig('docs/electric_dipole_field.png', dpi=150, bbox_inches='tight')
print("✓ Saved: docs/electric_dipole_field.png")
plt.close()

# 3. Circuit Analysis: Voltage Divider with Load
from physpy.electromagnetism.circuits import VoltageSource, Resistor, Circuit

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Unloaded vs Loaded voltage divider
circuit_unloaded = Circuit()
circuit_unloaded.add_node(0, gnd=True)
circuit_unloaded.add_node(1)
circuit_unloaded.add_node(2)
circuit_unloaded.add_connection(0, 1, VoltageSource('V1', voltage=10.0))
circuit_unloaded.add_connection(1, 2, Resistor('R1', resistance=1000.0))
circuit_unloaded.add_connection(2, 0, Resistor('R2', resistance=1000.0))
sol_unloaded = circuit_unloaded.solve_dc()

circuit_loaded = Circuit()
circuit_loaded.add_node(0, gnd=True)
circuit_loaded.add_node(1)
circuit_loaded.add_node(2)
circuit_loaded.add_node(3)
circuit_loaded.add_connection(0, 1, VoltageSource('V1', voltage=10.0))
circuit_loaded.add_connection(1, 2, Resistor('R1', resistance=1000.0))
circuit_loaded.add_connection(2, 0, Resistor('R2', resistance=1000.0))
circuit_loaded.add_connection(2, 3, Resistor('R_load', resistance=1000.0))
circuit_loaded.add_connection(3, 0, Resistor('R3', resistance=1000.0))
sol_loaded = circuit_loaded.solve_dc()

# Plot 1: Voltage divider output
load_values = [float('inf'), 10000, 1000, 100, 10]
v_outputs = [5.0]
for R_load in [10000, 1000, 100, 10]:
    circuit = Circuit()
    circuit.add_node(0, gnd=True)
    circuit.add_node(1)
    circuit.add_node(2)
    circuit.add_connection(0, 1, VoltageSource('V1', voltage=10.0))
    circuit.add_connection(1, 2, Resistor('R1', resistance=1000.0))
    circuit.add_connection(2, 0, Resistor('R2', resistance=1000.0))
    circuit.add_connection(2, 0, Resistor('R_load', resistance=float(R_load)))
    sol = circuit.solve_dc()
    v_outputs.append(sol['node_voltages'][2])

load_labels = ['No Load', '10 kΩ', '1 kΩ', '100 Ω', '10 Ω']
ax1.plot(load_labels, v_outputs, 'o-', linewidth=2.5, markersize=8, color='#2E86AB')
ax1.axhline(y=5.0, color='green', linestyle='--', alpha=0.5, label='Ideal (no load)')
ax1.set_ylabel('Output Voltage (V)', fontsize=11, fontweight='bold')
ax1.set_title('Voltage Divider: Load Effect', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)
ax1.set_ylim([0, 5.5])

# Plot 2: Current through elements
labels = ['No Load', 'R_load=1kΩ']
I_R1 = [sol_unloaded['branch_currents']['R1']*1000, sol_loaded['branch_currents']['R1']*1000]
I_R2 = [sol_unloaded['branch_currents']['R2']*1000, sol_loaded['branch_currents']['R2']*1000]
I_load = [0, sol_loaded['branch_currents'].get('R_load', 0)*1000]

x = np.arange(len(labels))
width = 0.25
ax2.bar(x - width, I_R1, width, label='I(R1)', color='#A23B72')
ax2.bar(x, I_R2, width, label='I(R2)', color='#F18F01')
ax2.bar(x + width, I_load, width, label='I(Load)', color='#2E86AB')
ax2.set_ylabel('Current (mA)', fontsize=11, fontweight='bold')
ax2.set_title('Circuit Currents: Kirchhoff Validation', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend()
ax2.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('docs/circuit_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: docs/circuit_analysis.png")
plt.close()

print("\n✅ All visualizations generated successfully!")
