# PhysPy

**An open-source Python toolkit for computational physics.**

PhysPy lets you simulate, visualize, and compute physics concepts interactively.
It builds on NumPy and SciPy to provide physics-specific tools across mechanics,
electromagnetism, quantum mechanics, and more — with every numerical result
validated against a known analytic solution wherever one exists.

> 🚧 **Status: early development.** The mechanics module (projectile motion)
> is implemented and tested. More domains are actively being added — see
> [Roadmap](#roadmap).

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

```python
from physpy.mechanics.kinematics import Projectile

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
print(f"Range without drag (textbook formula): {baseball.range_analytic():.2f} m")

baseball.plot()  # visualizes both trajectories for comparison
```

Run the full example:

```bash
python examples/projectile_demo.py
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

- [x] **Mechanics**: projectile motion (analytic + numerical, with drag)
- [ ] **Mechanics**: N-body gravitational simulation
- [ ] **Electromagnetism**: static field visualization (point charges, dipoles)
- [ ] **Electromagnetism**: circuit solver
- [ ] **Quantum mechanics**: 1D time-independent Schrödinger equation solver
- [ ] **Quantum mechanics**: wavepacket time evolution / animation
- [ ] **Particle physics**: relativistic kinematics, decay simulations

## Contributing

Issues and pull requests are welcome once the initial modules stabilize.
See `CONTRIBUTING.md` (coming soon) for guidelines.

## License

MIT — see [LICENSE](LICENSE).
