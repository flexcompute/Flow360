import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.models.solver_numerics import (
    LinearSolver,
    LineSearch,
    NavierStokesSolver,
)
from flow360.component.simulation.models.surface_models import Freestream, Wall
from flow360.component.simulation.models.volume_models import Fluid
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.primitives import Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Steady, Unsteady
from flow360.component.simulation.translator.solver_translator import get_solver_json
from flow360.component.simulation.unit_system import SI_unit_system

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_sim_params(navier_stokes_solver=None, time_stepping=None):
    """Build a minimal SimulationParams with the given NS solver and time stepping."""
    with SI_unit_system:
        ns = navier_stokes_solver or NavierStokesSolver()
        ts = time_stepping or Steady()
        return SimulationParams(
            operating_condition=AerospaceCondition.from_mach(mach=0.5),
            models=[
                Fluid(navier_stokes_solver=ns),
                Wall(entities=Surface(name="fluid/wall")),
                Freestream(entities=Surface(name="fluid/farfield")),
            ],
            time_stepping=ts,
        )


# ── LineSearch field constraints ─────────────────────────────────────────────


class TestLineSearchValidation:
    def test_defaults(self):
        ls = LineSearch()
        assert ls.residual_growth_threshold == 0.85
        assert ls.max_residual_growth == 1.1
        assert ls.activation_step == 100

    def test_residual_growth_threshold_bounds(self):
        LineSearch(residual_growth_threshold=0.0)
        LineSearch(residual_growth_threshold=1.0)
        with pytest.raises(Exception):
            LineSearch(residual_growth_threshold=-0.1)
        with pytest.raises(Exception):
            LineSearch(residual_growth_threshold=1.1)

    def test_max_residual_growth_must_be_ge_one(self):
        LineSearch(max_residual_growth=1.0)
        LineSearch(max_residual_growth=5.0)
        with pytest.raises(Exception):
            LineSearch(max_residual_growth=0.99)

    def test_activation_step_must_be_positive(self):
        LineSearch(activation_step=1)
        with pytest.raises(Exception):
            LineSearch(activation_step=0)
        with pytest.raises(Exception):
            LineSearch(activation_step=-1)


# ── NavierStokesSolver Krylov default population ────────────────────────────


class TestKrylovDefaults:
    def test_krylov_enabled_populates_defaults(self):
        ns = NavierStokesSolver(use_krylov_solver=True)
        assert ns.linear_solver.max_preconditioner_iterations == 25
        assert ns.linear_solver.krylov_relative_tolerance == 0.05
        assert ns.linear_solver.max_iterations == 15
        assert ns.line_search is not None
        assert ns.line_search.residual_growth_threshold == 0.85

    def test_krylov_enabled_respects_user_overrides(self):
        ns = NavierStokesSolver(
            use_krylov_solver=True,
            linear_solver=LinearSolver(
                max_iterations=10,
                max_preconditioner_iterations=30,
                krylov_relative_tolerance=0.01,
            ),
            line_search=LineSearch(residual_growth_threshold=0.5, max_residual_growth=2.0),
        )
        assert ns.linear_solver.max_preconditioner_iterations == 30
        assert ns.linear_solver.krylov_relative_tolerance == 0.01
        assert ns.linear_solver.max_iterations == 10
        assert ns.line_search.residual_growth_threshold == 0.5
        assert ns.line_search.max_residual_growth == 2.0

    def test_krylov_disabled_is_default(self):
        ns = NavierStokesSolver()
        assert ns.use_krylov_solver is False
        assert ns.linear_solver.max_preconditioner_iterations is None
        assert ns.linear_solver.krylov_relative_tolerance is None
        assert ns.line_search is None

    def test_krylov_enabled_with_empty_linear_solver_overrides_max_iterations(self):
        ns = NavierStokesSolver(
            use_krylov_solver=True,
            linear_solver=LinearSolver(),
        )
        assert ns.linear_solver.max_iterations == 15


# ── NavierStokesSolver Krylov error paths ────────────────────────────────────


class TestKrylovDisabledErrors:
    def test_error_max_preconditioner_iterations_without_krylov(self):
        with pytest.raises(ValueError, match="max_preconditioner_iterations"):
            NavierStokesSolver(
                use_krylov_solver=False,
                linear_solver=LinearSolver(max_preconditioner_iterations=10),
            )

    def test_error_krylov_relative_tolerance_without_krylov(self):
        with pytest.raises(ValueError, match="krylov_relative_tolerance"):
            NavierStokesSolver(
                use_krylov_solver=False,
                linear_solver=LinearSolver(krylov_relative_tolerance=0.1),
            )

    def test_error_line_search_without_krylov(self):
        with pytest.raises(ValueError, match="line_search"):
            NavierStokesSolver(
                use_krylov_solver=False,
                line_search=LineSearch(),
            )

    def test_error_krylov_fields_on_base_solver(self):
        with pytest.raises(ValueError):
            _make_sim_params(
                navier_stokes_solver=NavierStokesSolver(
                    use_krylov_solver=False,
                    linear_solver=LinearSolver(max_preconditioner_iterations=10),
                ),
            )
        with pytest.raises(ValueError):
            _make_sim_params(
                navier_stokes_solver=NavierStokesSolver(
                    use_krylov_solver=False,
                    linear_solver=LinearSolver(krylov_relative_tolerance=0.1),
                ),
            )
        with pytest.raises(ValueError):
            _make_sim_params(
                navier_stokes_solver=NavierStokesSolver(
                    use_krylov_solver=False,
                    line_search=LineSearch(),
                ),
            )

    def test_explicit_max_iterations_not_overridden_by_krylov_default(self):
        ns = NavierStokesSolver(
            use_krylov_solver=True,
            linear_solver=LinearSolver(max_iterations=30),
        )
        assert ns.linear_solver.max_iterations == 30

    def test_krylov_disabled_no_cap_on_max_iterations(self):
        ns = NavierStokesSolver(
            use_krylov_solver=False,
            linear_solver=LinearSolver(max_iterations=100),
        )
        assert ns.linear_solver.max_iterations == 100

    def test_krylov_disabled_default_max_iterations_unchanged(self):
        ns = NavierStokesSolver(use_krylov_solver=False)
        assert ns.linear_solver.max_iterations == 30

    def test_krylov_enabled_min_max_iterations(self):
        ns = NavierStokesSolver(
            use_krylov_solver=True,
            linear_solver=LinearSolver(max_iterations=1),
        )
        assert ns.linear_solver.max_iterations == 1

    def test_error_max_iterations_exceeds_limit(self):
        with pytest.raises(ValueError, match="max_iterations cannot exceed 50"):
            NavierStokesSolver(
                use_krylov_solver=True,
                linear_solver=LinearSolver(max_iterations=51),
            )

    def test_max_iterations_at_limit_is_ok(self):
        ns = NavierStokesSolver(
            use_krylov_solver=True,
            linear_solver=LinearSolver(max_iterations=50),
        )
        assert ns.linear_solver.max_iterations == 50


# ── Simulation-level Krylov restrictions ─────────────────────────────────────


class TestKrylovSimulationRestrictions:
    def test_error_krylov_with_limit_velocity(self):
        with pytest.raises(ValueError, match="limit_velocity"):
            _make_sim_params(
                navier_stokes_solver=NavierStokesSolver(
                    use_krylov_solver=True, limit_velocity=True
                ),
            )

    def test_error_krylov_with_limit_pressure_density(self):
        with pytest.raises(ValueError, match="limit_pressure_density"):
            _make_sim_params(
                navier_stokes_solver=NavierStokesSolver(
                    use_krylov_solver=True, limit_pressure_density=True
                ),
            )

    def test_error_krylov_with_unsteady(self):
        with SI_unit_system:
            with pytest.raises(ValueError, match="Unsteady"):
                _make_sim_params(
                    navier_stokes_solver=NavierStokesSolver(use_krylov_solver=True),
                    time_stepping=Unsteady(steps=100, step_size=0.1),
                )

    def test_krylov_with_steady_is_ok(self):
        param = _make_sim_params(
            navier_stokes_solver=NavierStokesSolver(use_krylov_solver=True),
            time_stepping=Steady(),
        )
        assert param is not None


# ── Translator: Krylov field handling ────────────────────────────────────────


class TestKrylovTranslation:
    def test_krylov_enabled_includes_fields(self):
        param = _make_sim_params(
            navier_stokes_solver=NavierStokesSolver(use_krylov_solver=True),
        )
        translated = get_solver_json(param, mesh_unit=1 * u.m)
        ns = translated["navierStokesSolver"]
        ls = ns["linearSolver"]

        assert "useKrylovSolver" not in ns
        assert ls["maxPreconditionerIterations"] == 25
        assert ls["krylovRelativeTolerance"] == 0.05
        assert ls["maxIterations"] == 15
        assert "lineSearch" in ns
        assert ns["lineSearch"]["residualGrowthThreshold"] == 0.85
        assert ns["lineSearch"]["maxResidualGrowth"] == 1.1
        assert ns["lineSearch"]["activationStep"] == 100

    def test_krylov_disabled_strips_fields(self):
        param = _make_sim_params(
            navier_stokes_solver=NavierStokesSolver(use_krylov_solver=False),
        )
        translated = get_solver_json(param, mesh_unit=1 * u.m)
        ns = translated["navierStokesSolver"]
        ls = ns.get("linearSolver", {})

        assert "useKrylovSolver" not in ns
        assert "maxPreconditionerIterations" not in ls
        assert "krylovRelativeTolerance" not in ls
        assert "lineSearch" not in ns

    def test_krylov_enabled_user_overrides_in_json(self):
        param = _make_sim_params(
            navier_stokes_solver=NavierStokesSolver(
                use_krylov_solver=True,
                linear_solver=LinearSolver(
                    max_iterations=20,
                    max_preconditioner_iterations=40,
                    krylov_relative_tolerance=0.1,
                ),
                line_search=LineSearch(
                    residual_growth_threshold=0.9,
                    max_residual_growth=1.5,
                    activation_step=200,
                ),
            ),
        )
        translated = get_solver_json(param, mesh_unit=1 * u.m)
        ns = translated["navierStokesSolver"]
        ls = ns["linearSolver"]

        assert ls["maxIterations"] == 20
        assert ls["maxPreconditionerIterations"] == 40
        assert ls["krylovRelativeTolerance"] == 0.1
        assert ns["lineSearch"]["residualGrowthThreshold"] == 0.9
        assert ns["lineSearch"]["maxResidualGrowth"] == 1.5
        assert ns["lineSearch"]["activationStep"] == 200
