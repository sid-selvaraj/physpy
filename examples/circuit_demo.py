"""
Demo: DC circuit solver using Kirchhoff's laws.

Shows how to:
1. Build a circuit with resistors and voltage sources
2. Solve for node voltages and branch currents
3. Validate against known circuit laws
"""

from physpy.electromagnetism.circuits import (
    Resistor, VoltageSource, CurrentSource, Circuit
)

print("=" * 70)
print("DC CIRCUIT SOLVER DEMO")
print("=" * 70)

# ------- Example 1: Simple series circuit (Ohm's Law) -------
print("\n1. Series Circuit: V = 10V, R = 1 kΩ")
print("-" * 70)

circuit1 = Circuit()
circuit1.add_node(0, gnd=True)
circuit1.add_node(1)
circuit1.add_connection(0, 1, VoltageSource('V1', voltage=10.0))
circuit1.add_connection(1, 0, Resistor('R1', resistance=1000.0))

print(circuit1.summary())

sol1 = circuit1.solve_dc()
print("\nSolution:")
print(f"  V(node 1) = {sol1['node_voltages'][1]:.4f} V")
print(f"  I(R1)     = {sol1['branch_currents']['R1']:.6f} A = {sol1['branch_currents']['R1']*1000:.2f} mA")
print(f"  Expected: V = 10 V, I = 10 mA (Ohm's Law: I = V/R = 10/1000)")

# ------- Example 2: Voltage Divider -------
print("\n2. Voltage Divider: V_in = 10V, R1 = 1kΩ, R2 = 1kΩ")
print("-" * 70)

circuit2 = Circuit()
circuit2.add_node(0, gnd=True)
circuit2.add_node(1)
circuit2.add_node(2)
circuit2.add_connection(0, 1, VoltageSource('V1', voltage=10.0))
circuit2.add_connection(1, 2, Resistor('R1', resistance=1000.0))
circuit2.add_connection(2, 0, Resistor('R2', resistance=1000.0))

print(circuit2.summary())

sol2 = circuit2.solve_dc()
print("\nSolution:")
print(f"  V(node 1) = {sol2['node_voltages'][1]:.4f} V")
print(f"  V(node 2) = {sol2['node_voltages'][2]:.4f} V")
print(f"  I(R1)     = {sol2['branch_currents']['R1']:.6f} A = {sol2['branch_currents']['R1']*1000:.2f} mA")
print(f"  I(R2)     = {sol2['branch_currents']['R2']:.6f} A = {sol2['branch_currents']['R2']*1000:.2f} mA")
print(f"  Expected: V(node 2) = 5 V (with equal resistors)")

# ------- Example 3: Parallel Resistors -------
print("\n3. Parallel Resistors: V = 10V, R1 = 1kΩ, R2 = 1kΩ")
print("-" * 70)

circuit3 = Circuit()
circuit3.add_node(0, gnd=True)
circuit3.add_node(1)
circuit3.add_connection(0, 1, VoltageSource('V1', voltage=10.0))
circuit3.add_connection(1, 0, Resistor('R1', resistance=1000.0))
circuit3.add_connection(1, 0, Resistor('R2', resistance=1000.0))

print(circuit3.summary())

sol3 = circuit3.solve_dc()
print("\nSolution:")
print(f"  V(node 1) = {sol3['node_voltages'][1]:.4f} V")
print(f"  I(R1)     = {sol3['branch_currents']['R1']:.6f} A = {sol3['branch_currents']['R1']*1000:.2f} mA")
print(f"  I(R2)     = {sol3['branch_currents']['R2']:.6f} A = {sol3['branch_currents']['R2']*1000:.2f} mA")
print(f"  I(total)  = {sol3['branch_currents']['R1'] + sol3['branch_currents']['R2']:.6f} A")
print(f"  Expected: I(total) = 20 mA (each resistor: 10 mA)")

# ------- Example 4: Current Source -------
print("\n4. Current Source: I = 5 mA, R = 2kΩ")
print("-" * 70)

circuit4 = Circuit()
circuit4.add_node(0, gnd=True)
circuit4.add_node(1)
circuit4.add_connection(0, 1, CurrentSource('I1', current=0.005))
circuit4.add_connection(1, 0, Resistor('R1', resistance=2000.0))

print(circuit4.summary())

sol4 = circuit4.solve_dc()
print("\nSolution:")
print(f"  V(node 1) = {sol4['node_voltages'][1]:.4f} V")
print(f"  I(source) = {sol4['branch_currents']['I1']:.6f} A = {sol4['branch_currents']['I1']*1000:.2f} mA")
print(f"  Expected: V = 10 V (Ohm's Law: V = I × R = 0.005 × 2000)")

# ------- Example 5: Complex Multi-Node Circuit -------
print("\n5. Complex Circuit: 3 nodes, mixed components")
print("-" * 70)

circuit5 = Circuit()
circuit5.add_node(0, gnd=True)
circuit5.add_node(1)
circuit5.add_node(2)
circuit5.add_connection(0, 1, VoltageSource('V1', voltage=15.0))
circuit5.add_connection(1, 2, Resistor('R1', resistance=1000.0))
circuit5.add_connection(2, 0, Resistor('R2', resistance=1000.0))
circuit5.add_connection(1, 0, Resistor('R3', resistance=2000.0))

print(circuit5.summary())

sol5 = circuit5.solve_dc()
print("\nSolution:")
for node_id in sorted([n for n in sol5['node_voltages'].keys()]):
    print(f"  V(node {node_id}) = {sol5['node_voltages'][node_id]:.4f} V")

print("\nBranch Currents:")
for elem_name in sorted(sol5['branch_currents'].keys()):
    I = sol5['branch_currents'][elem_name]
    print(f"  I({elem_name}) = {I:.6f} A = {I*1000:.3f} mA")

# Verify Kirchhoff's current law
print("\nKirchhoff's Current Law Check:")
I_R1 = sol5['branch_currents']['R1']
I_R2 = sol5['branch_currents']['R2']
print(f"  I(R1) = {I_R1*1000:.3f} mA (node 1 → node 2)")
print(f"  I(R2) = {I_R2*1000:.3f} mA (node 2 → node 0)")
print(f"  At node 2: I(R1) should equal I(R2)")
print(f"  Match: {abs(I_R1 - I_R2) < 1e-8}")

print("\n" + "=" * 70)
print("Demo complete!")
print("=" * 70)
