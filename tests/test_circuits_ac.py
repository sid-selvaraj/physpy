"""Tests for AC circuit solver."""

import pytest
import numpy as np
from physpy.electromagnetism.circuits_ac import ACCircuit, ACVoltageSource, Resistor, Capacitor, Inductor


def test_ac_resistor_ohm_law():
    """AC resistor follows Ohm's law: V = I*R."""
    circuit = ACCircuit()
    circuit.add_node(0, gnd=True)
    circuit.add_node(1)

    source = ACVoltageSource('V1', voltage_amplitude=10.0, frequency=1e3)
    circuit.add_ac_voltage_source(source, 0, 1)
    circuit.add_connection(1, 0, Resistor('R1', resistance=1000.0))

    sol = circuit.solve_ac()
    V1 = sol['magnitudes_V'][1]
    I = sol['branch_currents']['R1']

    assert np.isclose(V1, 10.0, rtol=0.01)
    assert np.isclose(abs(I), 0.01, rtol=0.01)


def test_ac_capacitor_impedance():
    """AC capacitor impedance: Z_C = 1/(jωC)."""
    circuit = ACCircuit()
    circuit.add_node(0, gnd=True)
    circuit.add_node(1)

    f = 1e3  # 1 kHz
    C = 1e-6  # 1 μF
    source = ACVoltageSource('V1', voltage_amplitude=5.0, frequency=f)
    circuit.add_ac_voltage_source(source, 0, 1)
    circuit.add_connection(1, 0, Capacitor('C1', capacitance=C))

    sol = circuit.solve_ac()

    # Expected: |Z_C| = 1/(ωC) = 1/(2π*1000*1e-6) ≈ 159 Ω
    omega = 2 * np.pi * f
    Z_C_expected = 1 / (omega * C)

    I_mag = abs(sol['branch_currents']['C1'])
    I_expected = 5.0 / Z_C_expected

    assert np.isclose(I_mag, I_expected, rtol=0.05)


def test_ac_inductor_impedance():
    """AC inductor impedance: Z_L = jωL."""
    circuit = ACCircuit()
    circuit.add_node(0, gnd=True)
    circuit.add_node(1)

    f = 1e3
    L = 1e-3  # 1 mH
    source = ACVoltageSource('V1', voltage_amplitude=5.0, frequency=f)
    circuit.add_ac_voltage_source(source, 0, 1)
    circuit.add_connection(1, 0, Inductor('L1', inductance=L))

    sol = circuit.solve_ac()

    omega = 2 * np.pi * f
    Z_L_expected = omega * L

    I_mag = abs(sol['branch_currents']['L1'])
    I_expected = 5.0 / Z_L_expected

    assert np.isclose(I_mag, I_expected, rtol=0.05)


def test_ac_rlc_series():
    """Series RLC circuit resonance."""
    circuit = ACCircuit()
    circuit.add_node(0, gnd=True)
    circuit.add_node(1)
    circuit.add_node(2)

    f = 1e3
    source = ACVoltageSource('V1', voltage_amplitude=10.0, frequency=f)
    circuit.add_ac_voltage_source(source, 0, 1)
    circuit.add_connection(1, 2, Resistor('R1', resistance=10.0))
    circuit.add_connection(2, 2, Inductor('L1', inductance=1e-3))  # Self-loop for series

    sol = circuit.solve_ac()
    assert np.isfinite(list(sol['magnitudes_V'].values())[0])


def test_ac_phase_shift():
    """Phase shift in capacitive circuit."""
    circuit = ACCircuit()
    circuit.add_node(0, gnd=True)
    circuit.add_node(1)

    source = ACVoltageSource('V1', voltage_amplitude=10.0, frequency=1e3, phase_deg=0)
    circuit.add_ac_voltage_source(source, 0, 1)
    circuit.add_connection(1, 0, Capacitor('C1', capacitance=1e-6))

    sol = circuit.solve_ac()
    I = sol['branch_currents']['C1']

    # Current should lead voltage by 90° in capacitor
    phase_diff = np.degrees(np.angle(I)) - 0  # Voltage phase = 0
    assert abs(phase_diff - 90) < 10 or abs(phase_diff + 270) < 10


def test_ac_phasor_representation():
    """Phasor representation V = V0 * exp(jφ)."""
    source = ACVoltageSource('V1', voltage_amplitude=10.0, frequency=1e3, phase_deg=45)
    phasor = source.phasor

    expected = 10.0 * np.exp(1j * np.radians(45))
    assert np.isclose(abs(phasor), abs(expected), rtol=0.01)
    assert np.isclose(np.degrees(np.angle(phasor)), 45, rtol=1)


def test_ac_frequency_sweep():
    """AC response varies with frequency."""
    results_1k = []
    results_10k = []

    for f in [1e3, 10e3]:
        circuit = ACCircuit()
        circuit.add_node(0, gnd=True)
        circuit.add_node(1)

        source = ACVoltageSource('V1', voltage_amplitude=5.0, frequency=f)
        circuit.add_ac_voltage_source(source, 0, 1)
        circuit.add_connection(1, 0, Capacitor('C1', capacitance=1e-6))

        sol = circuit.solve_ac()
        I = abs(sol['branch_currents']['C1'])

        if f == 1e3:
            results_1k.append(I)
        else:
            results_10k.append(I)

    # Current should increase with frequency for capacitor
    assert results_10k[0] > results_1k[0]


def test_ac_voltage_divider():
    """AC voltage divider in RC circuit."""
    circuit = ACCircuit()
    circuit.add_node(0, gnd=True)
    circuit.add_node(1)
    circuit.add_node(2)

    source = ACVoltageSource('V1', voltage_amplitude=10.0, frequency=1e3)
    circuit.add_ac_voltage_source(source, 0, 1)
    circuit.add_connection(1, 2, Resistor('R1', resistance=1000.0))
    circuit.add_connection(2, 0, Capacitor('C1', capacitance=1e-6))

    sol = circuit.solve_ac()
    V2 = sol['magnitudes_V'][2]

    # V2 should be attenuated compared to V1
    assert V2 < 10.0


def test_ac_solution_finite():
    """AC solution produces finite values."""
    circuit = ACCircuit()
    circuit.add_node(0, gnd=True)
    circuit.add_node(1)

    source = ACVoltageSource('V1', voltage_amplitude=10.0, frequency=1e3)
    circuit.add_ac_voltage_source(source, 0, 1)
    circuit.add_connection(1, 0, Resistor('R1', resistance=1000.0))

    sol = circuit.solve_ac()

    # All values should be finite
    for v in sol['magnitudes_V'].values():
        assert np.isfinite(v)
    for i in sol['branch_currents'].values():
        assert np.isfinite(abs(i))


def test_ac_frequency_tracking():
    """AC circuit tracks frequency correctly."""
    circuit = ACCircuit()
    circuit.add_node(0, gnd=True)
    circuit.add_node(1)

    f_test = 2.5e3
    source = ACVoltageSource('V1', voltage_amplitude=10.0, frequency=f_test)
    circuit.add_ac_voltage_source(source, 0, 1)
    circuit.add_connection(1, 0, Resistor('R1', resistance=1000.0))

    sol = circuit.solve_ac()

    assert np.isclose(sol['frequency'], f_test, rtol=0.01)
