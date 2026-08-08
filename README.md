# PhysPy

**An open-source Python toolkit for computational physics.**

PhysPy lets you simulate, visualize, and compute physics concepts interactively.
It builds on NumPy and SciPy to provide physics-specific tools across mechanics,
electromagnetism, quantum mechanics, and more — with every numerical result
validated against a known analytic solution wherever one exists.

> 🚧 **Status: early development.** Mechanics (projectile motion) and 
> Electromagnetism (statics & circuits) are implemented and tested.
> More domains are actively being added — see [Roadmap](#roadmap).

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

Run the examples:

```bash
python examples/projectile_demo.py
python examples/electric_field_demo.py
python examples/circuit_demo.py
```

## Running tests

```bash
pytest
```

Every physics result in PhysPy is checked against at least one of:
a hand-derivable analytic formula, agreement between the numerical and
analytic solvers in the limiting case, or a physical law that must hold
regardless of the exact parameters (e.g. "adding drag can only reduce range").

## Roadmap

### ✅ Completed
- [x] **Mechanics**: projectile motion (analytic + numerical, with drag)
- [x] **Electromagnetism**: static field visualization (point charges, dipoles)
- [x] **Electromagnetism**: DC circuit solver (Kirchhoff's laws, MNA)

### 📋 Planned
- [ ] **Mechanics**: N-body gravitational simulation
- [ ] **Electromagnetism**: AC circuit analysis (frequency-dependent)
- [ ] **Electromagnetism**: electromagnetic wave propagation
- [ ] **Quantum mechanics**: 1D time-independent Schrödinger equation solver
- [ ] **Quantum mechanics**: wavepacket time evolution / animation
- [ ] **Particle physics**: relativistic kinematics, decay simulations
- [ ] **Thermodynamics**: ideal gas simulations, phase diagrams

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style and testing
- Physics validation requirements
- Adding new modules and domains
- Documenting and visualizing results

## License

MIT — see [LICENSE](LICENSE).
