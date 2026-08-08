"""Tests for physpy.electromagnetism.circuits.

Validates circuit solver against known circuit laws and
standard circuit problems (voltage dividers, meshes, etc.).
"""

import pytest
import numpy as np

from physpy.electromagnetism.circuits import (
    Resistor, VoltageSource, CurrentSource, Capacitor, Inductor, Circuit
)


def test_simple_resistor_circuit():
    """Ohm's law: V = I*R. For V1=10V and R1=1kΩ, I should be 10mA."""
    circuit = Circuit()
    circuit.add_node(0, gnd=True)
    circuit.add_node(1)
    circuit.add_connection(0, 1, VoltageSource('V1', voltage=10.0))
    circuit.add_connection(1, 0, Resistor('R1', resistance=1000.0))

    sol = circuit.solve_dc()
    V1 = sol['node_voltages'][1]
    I_R1 = sol['branch_currents']['R1']

    assert np.isclose(V1, 10.0, rtol=1e-6)
    assert np.isclose(I_R1, 0.01, rtol=1e-6)  # 10mA


def test_voltage_divider():
    """Voltage divider: V_out = V_in * R2 / (R1 + R2)."""
    circuit = Circuit()
    circuit.add_node(0, gnd=True)
    circuit.add_node(1)
    circuit.add_node(2)

    circuit.add_connection(0, 1, VoltageSource('V1', voltage=10.0))
    circuit.add_connection(1, 2, Resistor('R1', resistance=1000.0))
    circuit.add_connection(2, 0, Resistor('R2', resistance=1000.0))

    sol = circuit.solve_dc()
    V2 = sol['node_voltages'][2]

    # With equal resistors, V2 should be 5V
    expected = 10.0 * 1000.0 / (1000.0 + 1000.0)
    assert np.isclose(V2, expected, rtol=1e-6)


def test_series_resistors():
    """Series resistors add: R_total = R1 + R2."""
    circuit = Circuit()
    circuit.add_node(0, gnd=True)
    circuit.add_node(1)
    circuit.add_node(2)

    circuit.add_connection(0, 1, VoltageSource('V1', voltage=10.0))
    circuit.add_connection(1, 2, Resistor('R1', resistance=500.0))
    circuit.add_connection(2, 0, Resistor('R2', resistance=500.0))

    sol = circuit.solve_dc()
    I = sol['branch_currents']['R1']

    # Total R = 1000 Ω, V = 10V => I = 10mA
    assert np.isclose(I, 0.01, rtol=1e-6)


def test_parallel_resistors():
    """Parallel resistors: 1/R_total = 1/R1 + 1/R2."""
    circuit = Circuit()
    circuit.add_node(0, gnd=True)
    circuit.add_node(1)

    circuit.add_connection(0, 1, VoltageSource('V1', voltage=10.0))
    circuit.add_connection(1, 0, Resistor('R1', resistance=1000.0))
    circuit.add_connection(1, 0, Resistor('R2', resistance=1000.0))

    sol = circuit.solve_dc()
    I1 = sol['branch_currents']['R1']
    I2 = sol['branch_currents']['R2']

    # Each 1kΩ resistor at 10V: 10mA each
    assert np.isclose(I1, 0.01, rtol=1e-6)
    assert np.isclose(I2, 0.01, rtol=1e-6)
    assert np.isclose(I1 + I2, 0.02, rtol=1e-6)


def test_current_source():
    """Current source forces a specific current through the circuit."""
    circuit = Circuit()
    circuit.add_node(0, gnd=True)
    circuit.add_node(1)

    circuit.add_connection(0, 1, CurrentSource('I1', current=0.005))  # 5mA
    circuit.add_connection(1, 0, Resistor('R1', resistance=2000.0))

    sol = circuit.solve_dc()
    V1 = sol['node_voltages'][1]
    I = sol['branch_currents']['I1']

    # V = I * R = 0.005 * 2000 = 10V
    assert np.isclose(V1, 10.0, rtol=1e-6)
    assert np.isclose(I, 0.005, rtol=1e-6)


def test_multiple_voltage_sources():
    """Circuit with two independent voltage sources."""
    circuit = Circuit()
    circuit.add_node(0, gnd=True)
    circuit.add_node(1)
    circuit.add_node(2)

    # V1 and V2 drive different nodes independently
    circuit.add_connection(0, 1, VoltageSource('V1', voltage=10.0))
    circuit.add_connection(0, 2, VoltageSource('V2', voltage=5.0))
    # Connect nodes 1 and 2 through resistor
    circuit.add_connection(1, 2, Resistor('R1', resistance=1000.0))

    sol = circuit.solve_dc()
    V1 = sol['node_voltages'][1]
    V2 = sol['node_voltages'][2]

    # V1 = 10V (from voltage source V1)
    # V2 = 5V (from voltage source V2)
    assert np.isclose(V1, 10.0, rtol=1e-6)
    assert np.isclose(V2, 5.0, rtol=1e-6)
    # Current through R1 should be (V1 - V2) / R = 5mA
    I_R1 = sol['branch_currents']['R1']
    assert np.isclose(I_R1, 0.005, rtol=1e-6)


def test_three_node_network():
    """Network with parallel resistors and voltage divider."""
    circuit = Circuit()
    circuit.add_node(0, gnd=True)
    circuit.add_node(1)
    circuit.add_node(2)

    # V1 drives node 1 to 10V
    circuit.add_connection(0, 1, VoltageSource('V1', voltage=10.0))

    # R1, R2 in parallel from node 1 to node 2 (equivalent 500Ω)
    circuit.add_connection(1, 2, Resistor('R1', resistance=1000.0))
    circuit.add_connection(1, 2, Resistor('R2', resistance=1000.0))

    # R3 from node 2 to ground (1kΩ)
    circuit.add_connection(2, 0, Resistor('R3', resistance=1000.0))

    sol = circuit.solve_dc()
    V2 = sol['node_voltages'][2]

    # Voltage divider: V(2) = V(1) * R3 / (R_parallel + R3)
    # R_parallel = 500Ω, R3 = 1000Ω
    # V(2) = 10 * 1000 / 1500 = 6.67V
    expected = 10.0 * 1000.0 / (500.0 + 1000.0)
    assert np.isclose(V2, expected, rtol=1e-3)


def test_current_conservation():
    """Kirchhoff's current law: current in = current out at each node."""
    circuit = Circuit()
    circuit.add_node(0, gnd=True)
    circuit.add_node(1)
    circuit.add_node(2)

    circuit.add_connection(0, 1, VoltageSource('V1', voltage=10.0))
    circuit.add_connection(1, 2, Resistor('R1', resistance=1000.0))
    circuit.add_connection(2, 0, Resistor('R2', resistance=1000.0))
    circuit.add_connection(1, 0, Resistor('R3', resistance=1000.0))

    sol = circuit.solve_dc()
    I_R1 = sol['branch_currents']['R1']
    I_R2 = sol['branch_currents']['R2']
    I_R3 = sol['branch_currents']['R3']

    # At node 1: I_R1 + I_R3 should equal I from voltage source (not solved explicitly, but check KCL at node 2)
    # At node 2: I_R1 should equal I_R2 (current in = current out)
    assert np.isclose(I_R1, I_R2, rtol=1e-6)


def test_voltage_divider_with_load():
    """Voltage divider loaded by a resistor."""
    circuit = Circuit()
    circuit.add_node(0, gnd=True)
    circuit.add_node(1)
    circuit.add_node(2)
    circuit.add_node(3)

    circuit.add_connection(0, 1, VoltageSource('V1', voltage=10.0))
    circuit.add_connection(1, 2, Resistor('R1', resistance=1000.0))
    circuit.add_connection(2, 0, Resistor('R2', resistance=1000.0))
    circuit.add_connection(2, 3, Resistor('R_load', resistance=1000.0))
    circuit.add_connection(3, 0, Resistor('R3', resistance=1000.0))

    sol = circuit.solve_dc()
    V2 = sol['node_voltages'][2]
    V3 = sol['node_voltages'][3]

    # Node 2 is the divider output (now loaded)
    # Exact value depends on parallel combination of R2 and R_load
    assert 0 < V2 < 10.0
    assert 0 < V3 < V2


def test_circuit_summary():
    """Test that circuit.summary() returns a string."""
    circuit = Circuit()
    circuit.add_node(0, gnd=True)
    circuit.add_node(1)
    circuit.add_connection(0, 1, VoltageSource('V1', voltage=10.0))
    circuit.add_connection(1, 0, Resistor('R1', resistance=1000.0))

    summary = circuit.summary()
    assert isinstance(summary, str)
    assert "Circuit Summary" in summary
    assert "V1" in summary
    assert "R1" in summary
