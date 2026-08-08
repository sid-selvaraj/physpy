# Contributing to PhysPy

Thank you for your interest in contributing to PhysPy! This document outlines how to contribute code, fixes, and new features.

## Core Principles

PhysPy is built on a foundation of **physical validation**. Every numerical result must be:

1. **Validated against an analytic solution** wherever one exists
2. **Checked against known physical laws** (e.g., conservation laws)
3. **Compared with the no-interaction / limiting case** to verify correctness

This means your contributions should include:
- Clear documentation of what physics is being modeled
- Tests that verify the calculations match known results
- Discussion of any assumptions or approximations

## Getting Started

1. **Fork and clone** the repository
2. **Set up a development environment:**
   ```bash
   git clone https://github.com/yourusername/physpy.git
   cd physpy
   pip install -e ".[dev]"
   ```

3. **Run existing tests** to verify setup:
   ```bash
   pytest -v
   ```

## Code Style

- **Python version:** 3.10+
- **Formatting:** Follow PEP 8. We recommend using `black` and `ruff`
  ```bash
  black src/physpy tests/
  ruff check src/physpy tests/
  ```
- **Type hints:** Use type annotations (e.g., `def solve() -> dict[str, np.ndarray]`)
- **Docstrings:** Use NumPy docstring format for modules and classes
  ```python
  def my_function(x: float) -> float:
      """One-line summary of what this does.
      
      Parameters
      ----------
      x : float
          Description of x.
      
      Returns
      -------
      float
          Description of return value.
      """
  ```

## Architecture: Adding a New Module

PhysPy modules follow a consistent pattern. Here's how to add a new physics domain:

### 1. Create the module structure

```
src/physpy/new_domain/
├── __init__.py          # Exports public classes
├── physics_concept.py   # Implementation
└── (optional) utils.py
```

### 2. Implement with validation in mind

Each class should:
- Have **analytic methods** for simple cases or known solutions
- Have **numerical methods** that can handle more complex scenarios
- **Compare both** in tests to ensure the numerical solver is correct

Example structure (from `mechanics.kinematics`):

```python
@dataclass
class MyPhysicalObject:
    """Docstring with parameters and examples."""
    
    # ---- Analytic (known-exact) solutions ----
    def some_quantity_analytic(self) -> float:
        """Return exact value from textbook formula."""
        return ...
    
    # ---- Numerical solutions ----
    def solve(self) -> dict:
        """Return numerical solution."""
        return ...
    
    # ---- Visualization ----
    def plot(self, ax=None, show: bool = True):
        """Visualize the result."""
        return ax
```

### 3. Write comprehensive tests

Tests should validate:

```python
def test_matches_known_formula():
    """Result matches hand-derivable analytic value."""
    obj = MyClass(...)
    result = obj.solve()
    expected = ... # from physics textbook
    assert np.isclose(result, expected, rtol=1e-6)

def test_numerical_matches_analytic_when_no_perturbation():
    """When perturbations are off, numerical = analytic."""
    obj = MyClass(..., perturbation_strength=0.0)
    numerical = obj.solve()
    analytic = obj.solve_analytic()
    assert np.isclose(numerical, analytic, rtol=1e-2)

def test_physical_law_always_holds():
    """Some physical law that must hold regardless of parameters."""
    obj1 = MyClass(param=small_value)
    obj2 = MyClass(param=large_value)
    # e.g., energy conservation, causality, monotonicity
    assert obj2.result < obj1.result  # e.g., more drag → less range
```

### 4. Add examples

Create `examples/your_domain_demo.py` showing:
- Basic usage
- Comparison with known results
- Visualization

### 5. Update documentation

- Add entries to `README.md` roadmap
- Link examples in the main docstring
- Update `src/physpy/__init__.py` exports

## Testing

Run tests before submitting:

```bash
# All tests
pytest

# With coverage
pytest --cov=src/physpy tests/

# Specific test file
pytest tests/test_your_module.py -v
```

**Coverage goal:** Aim for >90% coverage on new code.

## Common Patterns

### Constants
Import from `physpy.constants` (which wraps `scipy.constants`):

```python
from physpy.constants import epsilon_0, mu_0, e_charge
```

### Numerical integration
Use `scipy.integrate.solve_ivp` for ODEs (see `mechanics.kinematics` for example).

### Visualization
All visualization should use `matplotlib`. Make plots:
- Readable in both light and dark themes
- Include axes labels, units, and title
- Support optional `ax` parameter for integration into larger figures

### Error handling
- Validate inputs at module boundaries (user-facing functions)
- Trust internal code and framework guarantees
- Raise `ValueError` for physically impossible configurations with a clear message

## Submission Process

1. **Create a feature branch:** `git checkout -b feature/your-feature`
2. **Make your changes** with descriptive commits
3. **Run tests and linting** (see Testing above)
4. **Submit a pull request** with:
   - Clear title and description
   - Link to any related issues
   - Explanation of the physics being implemented
   - Note on validation approach

## Questions?

- Check existing modules (`mechanics.kinematics`, `electromagnetism.statics`) for patterns
- Open an issue to discuss major changes before implementing
- Reference physics textbooks or papers for validation approach

Thank you for contributing to making physics simulation more accessible and trustworthy!
