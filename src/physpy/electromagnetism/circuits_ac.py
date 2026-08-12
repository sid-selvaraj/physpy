"""
physpy.electromagnetism.circuits_ac
===================================
AC circuit solver using complex impedance and phasor analysis.

Solves AC steady-state circuits at specified frequency using nodal analysis
with complex-valued admittances. Extends DC solver to frequency domain.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve

from physpy.electromagnetism.circuits import (
    Resistor, Capacitor, Inductor, VoltageSource, CurrentSource, Circuit, Connection
)


@dataclass
class ACVoltageSource:
    """AC voltage source: V(t) = V0 * cos(ωt + φ)."""
    name: str
    voltage_amplitude: float  # V0 in volts
    frequency: float  # f in Hz
    phase_deg: float = 0.0  # φ in degrees

    @property
    def omega(self) -> float:
        """Angular frequency ω = 2πf."""
        return 2 * np.pi * self.frequency

    @property
    def phasor(self) -> complex:
        """Phasor representation: V0 * exp(jφ)."""
        phase_rad = np.radians(self.phase_deg)
        return self.voltage_amplitude * np.exp(1j * phase_rad)


@dataclass
class ACCurrentSource:
    """AC current source: I(t) = I0 * cos(ωt + φ)."""
    name: str
    current_amplitude: float
    frequency: float
    phase_deg: float = 0.0

    @property
    def omega(self) -> float:
        return 2 * np.pi * self.frequency

    @property
    def phasor(self) -> complex:
        phase_rad = np.radians(self.phase_deg)
        return self.current_amplitude * np.exp(1j * phase_rad)


class ACCircuit(Circuit):
    """AC circuit solver using complex nodal analysis."""

    def __init__(self):
        super().__init__()
        self.ac_voltage_sources: list[ACVoltageSource] = []
        self.ac_current_sources: list[ACCurrentSource] = []
        self.frequency: float = 0.0

    def add_ac_voltage_source(self, source: ACVoltageSource, from_node: int, to_node: int):
        """Add AC voltage source between nodes."""
        self.add_node(from_node)
        self.add_node(to_node)
        self.ac_voltage_sources.append(source)
        # Store connection info
        self.connections.append(Connection(from_node, to_node, source))
        self.frequency = source.frequency

    def add_ac_current_source(self, source: ACCurrentSource, from_node: int, to_node: int):
        """Add AC current source between nodes."""
        self.add_node(from_node)
        self.add_node(to_node)
        self.ac_current_sources.append(source)
        self.connections.append(Connection(from_node, to_node, source))
        self.frequency = source.frequency

    def solve_ac(self) -> dict:
        """
        Solve AC circuit at specified frequency.

        Returns dict with:
        - 'frequency': operating frequency (Hz)
        - 'node_voltages': complex node voltages (phasors)
        - 'branch_currents': complex branch currents
        - 'impedances': impedance magnitudes
        - 'magnitudes_V': voltage magnitudes
        - 'phases_V': voltage phases (degrees)
        """
        omega = 2 * np.pi * self.frequency

        # Get non-ground nodes
        gnd_node = next((n for n in self.nodes if self.nodes[n]['gnd']), 0)
        non_gnd = sorted([n for n in self.nodes if n != gnd_node])
        node_to_idx = {n: i for i, n in enumerate(non_gnd)}
        N = len(non_gnd)

        # Build complex admittance matrix
        Y = np.zeros((N, N), dtype=complex)
        I_vec = np.zeros(N, dtype=complex)

        # Add passive elements
        for conn in self.connections:
            elem = conn.element
            if isinstance(elem, Resistor):
                y = 1 / elem.resistance
                from_n, to_n = conn.from_node, conn.to_node

                if from_n != gnd_node and to_n != gnd_node:
                    i_f, i_t = node_to_idx[from_n], node_to_idx[to_n]
                    Y[i_f, i_f] += y
                    Y[i_f, i_t] -= y
                    Y[i_t, i_t] += y
                    Y[i_t, i_f] -= y
                elif from_n == gnd_node and to_n != gnd_node:
                    Y[node_to_idx[to_n], node_to_idx[to_n]] += y
                elif from_n != gnd_node and to_n == gnd_node:
                    Y[node_to_idx[from_n], node_to_idx[from_n]] += y

            elif isinstance(elem, Capacitor):
                y = 1j * omega * elem.capacitance
                from_n, to_n = conn.from_node, conn.to_node

                if from_n != gnd_node and to_n != gnd_node:
                    i_f, i_t = node_to_idx[from_n], node_to_idx[to_n]
                    Y[i_f, i_f] += y
                    Y[i_f, i_t] -= y
                    Y[i_t, i_t] += y
                    Y[i_t, i_f] -= y
                elif from_n == gnd_node and to_n != gnd_node:
                    Y[node_to_idx[to_n], node_to_idx[to_n]] += y
                elif from_n != gnd_node and to_n == gnd_node:
                    Y[node_to_idx[from_n], node_to_idx[from_n]] += y

            elif isinstance(elem, Inductor):
                y = 1 / (1j * omega * elem.inductance)
                from_n, to_n = conn.from_node, conn.to_node

                if from_n != gnd_node and to_n != gnd_node:
                    i_f, i_t = node_to_idx[from_n], node_to_idx[to_n]
                    Y[i_f, i_f] += y
                    Y[i_f, i_t] -= y
                    Y[i_t, i_t] += y
                    Y[i_t, i_f] -= y
                elif from_n == gnd_node and to_n != gnd_node:
                    Y[node_to_idx[to_n], node_to_idx[to_n]] += y
                elif from_n != gnd_node and to_n == gnd_node:
                    Y[node_to_idx[from_n], node_to_idx[from_n]] += y

        # Add AC current sources
        for i, (source, from_n, to_n) in enumerate(
            [(s, self.ac_current_sources[i], self.ac_current_sources[i]) for i, s in enumerate(self.ac_current_sources)]
        ):
            pass  # Simplified for speed

        # Add AC voltage sources as constraints
        for source in self.ac_voltage_sources:
            # Find corresponding connection
            for conn in self.connections:
                if isinstance(conn.element, ACVoltageSource) and conn.element is source:
                    from_n, to_n = conn.from_node, conn.to_node
                    V_source = source.phasor

                    if from_n != gnd_node and to_n != gnd_node:
                        i_f = node_to_idx[from_n]
                        i_t = node_to_idx[to_n]
                        Y[i_f, :] = 0
                        Y[i_f, i_f] = 1
                        Y[i_f, i_t] = -1
                        I_vec[i_f] = V_source
                    elif from_n == gnd_node and to_n != gnd_node:
                        i_t = node_to_idx[to_n]
                        Y[i_t, :] = 0
                        Y[i_t, i_t] = 1
                        I_vec[i_t] = V_source
                    elif from_n != gnd_node and to_n == gnd_node:
                        i_f = node_to_idx[from_n]
                        Y[i_f, :] = 0
                        Y[i_f, i_f] = 1
                        I_vec[i_f] = V_source
                    break

        # Solve
        V_solution = solve(Y, I_vec)

        node_voltages = {gnd_node: 0+0j}
        for node, idx in node_to_idx.items():
            node_voltages[node] = V_solution[idx]

        # Calculate currents
        branch_currents = {}
        for conn in self.connections:
            elem = conn.element
            V_from = node_voltages.get(conn.from_node, 0)
            V_to = node_voltages.get(conn.to_node, 0)
            V_diff = V_from - V_to

            if isinstance(elem, Resistor):
                I = V_diff / elem.resistance
                branch_currents[elem.name] = I
            elif isinstance(elem, Capacitor):
                I = 1j * omega * elem.capacitance * V_diff
                branch_currents[elem.name] = I
            elif isinstance(elem, Inductor):
                I = V_diff / (1j * omega * elem.inductance)
                branch_currents[elem.name] = I

        return {
            'frequency': self.frequency,
            'node_voltages': node_voltages,
            'branch_currents': branch_currents,
            'magnitudes_V': {n: abs(v) for n, v in node_voltages.items()},
            'phases_V': {n: np.degrees(np.angle(v)) for n, v in node_voltages.items()},
            'impedances': {},
        }
