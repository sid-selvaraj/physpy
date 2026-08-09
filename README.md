# PhysPy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests: 40 passing](https://img.shields.io/badge/Tests-40%20passing-brightgreen.svg)](#running-tests)
[![Coverage: >90%](https://img.shields.io/badge/Coverage-%3E90%25-brightgreen.svg)](#running-tests)

**An open-source Python toolkit for computational physics.**

PhysPy lets you simulate, visualize, and compute physics concepts interactively.
It builds on NumPy and SciPy to provide physics-specific tools across mechanics,
electromagnetism, quantum mechanics, and more — with **every numerical result
validated against a known analytic solution** wherever one exists.

> 🚀 **Status: active development.** Mechanics (projectile motion) and 
> Electromagnetism (statics & circuits) are fully implemented and tested.
> Quantum mechanics coming next — see [Roadmap](#roadmap).

## Why PhysPy?

Most physics simulation code lives in one-off Jupyter notebooks: hard to
reuse, hard to trust, and easy to get subtly wrong. PhysPy aims to be:

- **Physically validated** — every simulated result is checked against an
  analytic solution or a known physical law wherever possible.
- **Interactive** — one method call gets you a plot, not a wall of numbers.
- **Built to grow** — a consistent architecture (numerical core + analytic
  checks + visualization) that scales from a projectile to a many-body
  simulation without a redesign.

## Installation

```bash
git clone https://github.com/yourusername/physpy.git
cd physpy
pip install -e ".[dev]"
```

## Quickstart

### Mechanics: Projectile Motion

```python
from physpy import Projectile

# A baseball launched at 30 m/s, 45 degrees, with real air resistance
baseball = Projectile(
    v0=30,
    angle_deg=45,
    mass=0.145,
    drag_coefficient=0.47,
    cross_section_area=0.0042,
)

sol = baseball.solve()
print(f"Range with drag: {sol['x'][-1]:.2f} m")
print(f"Range without drag: {baseball.range_analytic():.2f} m")

baseball.plot()  # visualizes trajectories
```

### Electromagnetism: Electric Fields

```python
from physpy import PointCharge, ElectricField

# Two charges: +1 µC and -1 µC
q1 = PointCharge(q=1e-6, x=-0.05, y=0)
q2 = PointCharge(q=-1e-6, x=0.05, y=0)

field = ElectricField(charges=[q1, q2])
field.plot()  # visualizes field and equipotential lines
```

### Electromagnetism: DC Circuits

```python
from physpy import VoltageSource, Resistor, Circuit

circuit = Circuit()
circuit.add_node(0, gnd=True)
circuit.add_node(1)
circuit.add_connection(0, 1, VoltageSource('V1', voltage=10.0))
circuit.add_connection(1, 0, Resistor('R1', resistance=1000.0))

solution = circuit.solve_dc()
print(f"V(node 1) = {solution['node_voltages'][1]:.2f} V")
print(f"I(R1) = {solution['branch_currents']['R1']*1000:.2f} mA")
```

### Quantum Mechanics: 1D Schrödinger Solver

```python
from physpy import ParticleInABox, SchrodingerSolver1D

# Electron in a 1 nm box
box = ParticleInABox(L=1e-9)
solver = SchrodingerSolver1D(box, x_min=0, x_max=1e-9)

energies, wavefunctions = solver.solve()
print(f"Ground state energy: {energies[0]:.3e} J")

solver.plot()  # Visualize wavefunctions
```

Run the examples:

```bash
python examples/projectile_demo.py
python examples/electric_field_demo.py
python examples/circuit_demo.py
python examples/quantum_demo.py
```

## Visualization Gallery

### Projectile Motion: Effect of Air Resistance
![Projectile trajectories with and without drag](docs/projectile_comparison.png)

**Key insight**: Air drag reduces range by 32 m for a baseball launched at 30 m/s, 45°.
The numerical solver (with drag) matches analytic predictions (no drag) when drag is disabled.

### Electric Field: Dipole Configuration
![Electric dipole field and equipotential lines](docs/electric_dipole_field.png)

**Key insight**: Equipotential lines (gray) show constant potential; field vectors (colored)
show direction and magnitude. Opposite charges create the characteristic dipole pattern.

### Circuit Analysis: Voltage Divider Under Load
![Voltage divider with varying loads](docs/circuit_analysis.png)

**Key insight**: Adding load resistance changes the output voltage (left). Kirchhoff's current law
is validated: currents sum to zero at each node (right).

## Verification & Testing Philosophy

PhysPy is built on a foundation of **physical rigor**. Every numerical result is validated
against something independently verifiable:

### ✅ Validation Strategies

1. **Analytic Solutions** — Compare numerical results to hand-derivable formulas from physics textbooks.
   - Example: Projectile range without drag matches `v₀²·sin(2θ)/g`

2. **Cross-Solver Comparison** — Verify numerical solver matches analytic solver when approximations are disabled.
   - Example: With `drag_coefficient=0`, ODE solver reproduces textbook parabolic trajectory

3. **Physical Laws** — Check that laws like energy conservation or monotonicity always hold.
   - Example: Air drag must always reduce range compared to no-drag case, regardless of parameters

4. **Dimensional Consistency** — Verify units propagate correctly through calculations.
   - Example: Field magnitude must follow `E ∝ q/r²` (Coulomb's law)

### Test Coverage

```
tests/test_kinematics.py         ✅ 5 tests  — projectile motion validated
tests/test_electrostatics.py     ✅ 10 tests — Coulomb's law, superposition
tests/test_circuits.py           ✅ 10 tests — Kirchhoff's laws, Ohm's law
tests/test_quantum.py            ✅ 15 tests — particle-in-box, QHO, tunneling
────────────────────────────────────────────────────────────
                                 ✅ 40 tests — ALL PASSING
```

Run tests with coverage:

```bash
pytest tests/ -v
pytest tests/ --cov=src/physpy --cov-report=term-missing
```

## Input Units & Conventions

**All inputs use SI (International System) units:**

| Quantity | Unit | Example |
|----------|------|---------|
| Velocity | m/s | `v0=30` means 30 m/s |
| Angle | degrees | `angle_deg=45` |
| Charge | Coulombs | `q=1e-6` means 1 μC |
| Resistance | Ohms | `resistance=1000.0` means 1 kΩ |
| Distance | meters | `x=0.05` means 5 cm |
| Electric field | N/C | returned values in Newtons/Coulomb |

### Running tests

```bash
pytest tests/ -v
pytest tests/ --cov=src/physpy --cov-report=html
```

## Roadmap

### ✅ Completed
- [x] **Mechanics**: projectile motion (analytic + numerical, with drag)
- [x] **Electromagnetism**: static field visualization (point charges, dipoles)
- [x] **Electromagnetism**: DC circuit solver (Kirchhoff's laws, nodal analysis)
- [x] **Quantum**: 1D time-independent Schrödinger solver
  - Particle in a box (infinite square well)
  - Quantum harmonic oscillator
  - Finite square well (quantum tunneling)
  - Custom potentials

### 📋 Planned
- [ ] **Quantum**: 1D time-dependent Schrödinger (wavepacket evolution)
- [ ] **Quantum**: 2D systems and separation of variables
- [ ] **Mechanics**: N-body gravitational simulation (orbits, collisions)
- [ ] **Electromagnetism**: AC circuit analysis (frequency-dependent impedance)
- [ ] **Electromagnetism**: electromagnetic wave propagation (Maxwell equations)
- [ ] **Thermodynamics**: ideal gas simulations, Maxwell-Boltzmann distribution
- [ ] **Thermodynamics**: phase diagrams, critical phenomena
- [ ] **Particle physics**: relativistic kinematics, decay simulations

## Architecture & Design

### Separation of Concerns

PhysPy maintains a clean separation between physics computation and visualization:

- **Physics solvers** (`solve()`, `electric_field()`) return raw NumPy arrays or dicts
- **Visualization** (`plot()`) is a separate, optional layer
- This enables headless computation, web integrations, and automated pipelines

### Module Structure

Each domain (mechanics, electromagnetism, quantum) follows the same pattern:

```python
@dataclass
class PhysicalSystem:
    """Parameters defining the system."""
    
    # Analytic solutions (when available)
    def some_quantity_analytic(self) -> float:
        """Ground truth from physics textbooks."""
        return ...
    
    # Numerical solutions
    def solve(self) -> dict:
        """ODE/eigenvalue solver results."""
        return {...}
    
    # Visualization (optional)
    def plot(self, ax=None, show=True):
        """Matplotlib plotting."""
        return ax
```

This architecture ensures:
- ✅ Physics correctness is testable without visualization
- ✅ New visualization methods don't require physics changes
- ✅ Results are composable and reusable

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style and testing
- Physics validation requirements  
- Adding new modules and domains
- Documenting and visualizing results

## License

MIT — see [LICENSE](LICENSE).
