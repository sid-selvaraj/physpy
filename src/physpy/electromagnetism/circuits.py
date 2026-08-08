"""
physpy.electromagnetism.circuits
=================================
DC circuit solver using Kirchhoff's laws and nodal analysis.

Circuit elements: Resistor, Capacitor, Inductor, VoltageSource, CurrentSource
Circuit class: build and solve DC circuits using modified nodal analysis (MNA)
"""

from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.linalg import solve


@dataclass
class CircuitElement(ABC):
    """Base class for circuit elements."""

    name: str

    @abstractmethod
    def get_admittance(self, frequency: float = 0) -> complex:
        """Return admittance Y = 1/Z for AC analysis. For DC, frequency=0."""
        pass


@dataclass
class Resistor(CircuitElement):
    """
    Resistor element.

    Parameters
    ----------
    name : str
        Element identifier (e.g., 'R1').
    resistance : float
        Resistance in Ohms.
    """

    resistance: float

    def get_admittance(self, frequency: float = 0) -> complex:
        """Admittance Y = 1/R."""
        return 1 / self.resistance if self.resistance != 0 else float('inf')


@dataclass
class VoltageSource(CircuitElement):
    """
    Ideal voltage source.

    Parameters
    ----------
    name : str
        Element identifier (e.g., 'V1').
    voltage : float
        Voltage in Volts.
    """

    voltage: float

    def get_admittance(self, frequency: float = 0) -> complex:
        return 0  # Ideal voltage source has zero admittance


@dataclass
class CurrentSource(CircuitElement):
    """
    Ideal current source.

    Parameters
    ----------
    name : str
        Element identifier (e.g., 'I1').
    current : float
        Current in Amperes.
    """

    current: float

    def get_admittance(self, frequency: float = 0) -> complex:
        return 0  # Ideal current source has zero admittance


@dataclass
class Capacitor(CircuitElement):
    """
    Capacitor element (frequency-dependent).

    Parameters
    ----------
    name : str
        Element identifier.
    capacitance : float
        Capacitance in Farads.
    """

    capacitance: float

    def get_admittance(self, frequency: float = 0) -> complex:
        """Admittance Y = j*ω*C. For DC (ω=0), Y=0."""
        if frequency == 0:
            return 0
        return 1j * 2 * np.pi * frequency * self.capacitance


@dataclass
class Inductor(CircuitElement):
    """
    Inductor element (frequency-dependent).

    Parameters
    ----------
    name : str
        Element identifier.
    inductance : float
        Inductance in Henries.
    """

    inductance: float

    def get_admittance(self, frequency: float = 0) -> complex:
        """Admittance Y = 1/(j*ω*L). For DC (ω=0), Y→∞."""
        if frequency == 0:
            return float('inf')  # Inductor is a short at DC
        return 1 / (1j * 2 * np.pi * frequency * self.inductance)


@dataclass
class Connection:
    """A connection between two nodes via a circuit element."""

    from_node: int
    to_node: int
    element: CircuitElement


class Circuit:
    """
    DC circuit solver using nodal analysis.

    Build a circuit by adding connections between nodes, then solve
    for voltages at each node and currents through each element.

    Examples
    --------
    >>> circuit = Circuit()
    >>> circuit.add_node(0, gnd=True)  # node 0 is ground (0V)
    >>> circuit.add_connection(0, 1, VoltageSource('V1', voltage=10))
    >>> circuit.add_connection(1, 0, Resistor('R1', resistance=1000))
    >>> solution = circuit.solve_dc()
    >>> print(solution['node_voltages'])
    """

    def __init__(self):
        self.nodes: dict[int, dict] = {}  # node_id -> {gnd: bool, voltage: float}
        self.connections: list[Connection] = []
        self.voltage_sources: list[tuple[int, int, VoltageSource]] = []

    def add_node(self, node_id: int, gnd: bool = False, voltage: float = 0.0):
        """Add a node to the circuit."""
        self.nodes[node_id] = {'gnd': gnd, 'voltage': voltage}

    def add_connection(self, from_node: int, to_node: int, element: CircuitElement):
        """Add a circuit element between two nodes."""
        if from_node not in self.nodes:
            self.add_node(from_node)
        if to_node not in self.nodes:
            self.add_node(to_node)

        self.connections.append(Connection(from_node, to_node, element))

        if isinstance(element, VoltageSource):
            self.voltage_sources.append((from_node, to_node, element))

    def solve_dc(self) -> dict:
        """
        Solve the circuit for DC steady-state using nodal analysis.

        Returns
        -------
        dict with keys:
            - 'node_voltages': dict of {node_id: voltage_V}
            - 'branch_currents': dict of {element_name: current_A}
        """
        # Identify ground node
        gnd_node = next((n for n in self.nodes if self.nodes[n]['gnd']), None)
        if gnd_node is None:
            raise ValueError("Circuit must have a ground node (gnd=True)")

        # Get all non-ground nodes
        all_nodes = sorted(self.nodes.keys())
        non_gnd_nodes = [n for n in all_nodes if n != gnd_node]
        num_unknowns = len(non_gnd_nodes)

        # Map node IDs to matrix indices
        node_to_idx = {node: i for i, node in enumerate(non_gnd_nodes)}

        # Initialize system: Y*V = I
        Y_matrix = np.zeros((num_unknowns, num_unknowns))
        I_vector = np.zeros(num_unknowns)

        # Build nodal admittance matrix (Kirchhoff's current law)
        for conn in self.connections:
            from_node = conn.from_node
            to_node = conn.to_node
            element = conn.element

            # Handle passive elements (resistors, capacitors, inductors)
            if not isinstance(element, (VoltageSource, CurrentSource)):
                y = element.get_admittance(frequency=0)
                if isinstance(y, complex):
                    y = y.real

                # Determine which nodes are non-ground
                from_is_gnd = from_node == gnd_node
                to_is_gnd = to_node == gnd_node

                if not from_is_gnd and not to_is_gnd:
                    # Element between two non-ground nodes
                    i_from = node_to_idx[from_node]
                    i_to = node_to_idx[to_node]
                    Y_matrix[i_from, i_from] += y
                    Y_matrix[i_from, i_to] -= y
                    Y_matrix[i_to, i_to] += y
                    Y_matrix[i_to, i_from] -= y

                elif not from_is_gnd and to_is_gnd:
                    # Element from non-ground to ground
                    i_from = node_to_idx[from_node]
                    Y_matrix[i_from, i_from] += y

                elif from_is_gnd and not to_is_gnd:
                    # Element from ground to non-ground
                    i_to = node_to_idx[to_node]
                    Y_matrix[i_to, i_to] += y

            # Handle current sources
            elif isinstance(element, CurrentSource):
                I_src = element.current
                if from_node != gnd_node:
                    I_vector[node_to_idx[from_node]] -= I_src
                if to_node != gnd_node:
                    I_vector[node_to_idx[to_node]] += I_src

            # Handle voltage sources (inject them into RHS)
            elif isinstance(element, VoltageSource):
                # For now, handle as ideal voltage source between two nodes
                # by eliminating one node and using the constraint V1 - V2 = V_source
                pass

        # Handle voltage sources by treating as constraints
        # Substitute voltage source constraints into the main equations
        for from_node, to_node, vsource in self.voltage_sources:
            V_constraint = vsource.voltage

            from_is_gnd = from_node == gnd_node
            to_is_gnd = to_node == gnd_node

            if from_is_gnd and not to_is_gnd:
                # V_to = V_constraint
                i_to = node_to_idx[to_node]
                I_vector[i_to] = V_constraint
                Y_matrix[i_to, :] = 0
                Y_matrix[i_to, i_to] = 1

            elif not from_is_gnd and to_is_gnd:
                # V_from = V_constraint
                i_from = node_to_idx[from_node]
                I_vector[i_from] = V_constraint
                Y_matrix[i_from, :] = 0
                Y_matrix[i_from, i_from] = 1

            elif not from_is_gnd and not to_is_gnd:
                # V_from - V_to = V_constraint
                # Eliminate one variable, substitute the other
                i_from = node_to_idx[from_node]
                i_to = node_to_idx[to_node]

                # Replace row i_from with: V_from - V_to = V_constraint
                Y_matrix[i_from, :] = 0
                Y_matrix[i_from, i_from] = 1
                Y_matrix[i_from, i_to] = -1
                I_vector[i_from] = V_constraint

        # Solve the system
        try:
            solution_vec = solve(Y_matrix, I_vector)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Circuit matrix is singular: {e}")

        # Extract node voltages
        node_voltages = {gnd_node: 0.0}
        for node_id, idx in node_to_idx.items():
            node_voltages[node_id] = float(solution_vec[idx])

        # Calculate branch currents
        branch_currents = {}
        for conn in self.connections:
            from_node = conn.from_node
            to_node = conn.to_node
            element = conn.element

            V_from = node_voltages.get(from_node, 0.0)
            V_to = node_voltages.get(to_node, 0.0)
            V_diff = V_from - V_to

            if isinstance(element, Resistor):
                current = V_diff / element.resistance if element.resistance != 0 else 0.0
                branch_currents[element.name] = float(current)

            elif isinstance(element, VoltageSource):
                # Current through voltage source computed from KCL at source terminals
                # For a voltage source, the current depends on connected loads
                # Approximate: use current from connected resistive paths
                I_total = 0.0
                for other_conn in self.connections:
                    if other_conn is conn:
                        continue
                    if isinstance(other_conn.element, Resistor):
                        if other_conn.from_node == from_node or other_conn.to_node == from_node:
                            V_o_from = node_voltages.get(other_conn.from_node, 0.0)
                            V_o_to = node_voltages.get(other_conn.to_node, 0.0)
                            V_o_diff = V_o_from - V_o_to
                            I_total += V_o_diff / other_conn.element.resistance
                branch_currents[element.name] = float(I_total)

            elif isinstance(element, CurrentSource):
                branch_currents[element.name] = float(element.current)

            elif isinstance(element, Capacitor):
                y = element.get_admittance(frequency=0)
                current = V_diff * y if y != 0 else 0.0
                branch_currents[element.name] = float(current)

            elif isinstance(element, Inductor):
                # At DC, inductor acts as short circuit (0 voltage drop, arbitrary current)
                branch_currents[element.name] = 0.0

        return {
            'node_voltages': node_voltages,
            'branch_currents': branch_currents,
        }

    def summary(self) -> str:
        """Return a text summary of circuit configuration."""
        lines = ["Circuit Summary"]
        lines.append("=" * 40)
        lines.append("Nodes:")
        for node_id in sorted(self.nodes.keys()):
            gnd_str = " (GND)" if self.nodes[node_id]['gnd'] else ""
            lines.append(f"  Node {node_id}{gnd_str}")

        lines.append("\nElements:")
        for conn in self.connections:
            elem = conn.element
            lines.append(
                f"  {elem.name}: Node {conn.from_node} -> Node {conn.to_node}"
            )
            if isinstance(elem, Resistor):
                lines.append(f"    Resistance: {elem.resistance} Ω")
            elif isinstance(elem, VoltageSource):
                lines.append(f"    Voltage: {elem.voltage} V")
            elif isinstance(elem, CurrentSource):
                lines.append(f"    Current: {elem.current} A")
            elif isinstance(elem, Capacitor):
                lines.append(f"    Capacitance: {elem.capacitance} F")
            elif isinstance(elem, Inductor):
                lines.append(f"    Inductance: {elem.inductance} H")

        return "\n".join(lines)
