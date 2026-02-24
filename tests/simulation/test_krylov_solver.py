import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.models.solver_numerics import (
    KrylovLinearSolver,
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
        LineSearch(residual_growth_threshold=0.5)
        LineSearch(residual_growth_threshold=1.0)
        with pytest.raises(Exception):
            LineSearch(residual_growth_threshold=0.49)
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


# ── KrylovLinearSolver defaults and validation ───────────────────────────────


class TestKrylovLinearSolverDefaults:
    def test_defaults(self):
        kls = KrylovLinearSolver()
        assert kls.max_iterations == 15
        assert kls.max_preconditioner_iterations == 25
        assert kls.krylov_relative_tolerance == 0.05

    def test_user_overrides(self):
        kls = KrylovLinearSolver(
            max_iterations=10,
            max_preconditioner_iterations=30,
            krylov_relative_tolerance=0.01,
        )
        assert kls.max_iterations == 10
        assert kls.max_preconditioner_iterations == 30
        assert kls.krylov_relative_tolerance == 0.01

    def test_max_iterations_at_limit(self):
        kls = KrylovLinearSolver(max_iterations=50)
        assert kls.max_iterations == 50

    def test_max_iterations_exceeds_limit(self):
        with pytest.raises(ValueError, match="max_iterations cannot exceed 50"):
            KrylovLinearSolver(max_iterations=51)

    def test_max_iterations_minimum(self):
        kls = KrylovLinearSolver(max_iterations=1)
        assert kls.max_iterations == 1

    def test_inherits_linear_solver(self):
        assert issubclass(KrylovLinearSolver, LinearSolver)

    def test_type_name(self):
        assert KrylovLinearSolver().type_name == "KrylovLinearSolver"
        assert LinearSolver().type_name == "LinearSolver"

    def test_type_name_in_dump(self):
        dump = LinearSolver().model_dump()
        assert dump["type_name"] == "LinearSolver"
        dump = KrylovLinearSolver().model_dump()
        assert dump["type_name"] == "KrylovLinearSolver"

    def test_type_name_frozen(self):
        with pytest.raises(Exception):
            LinearSolver(type_name="KrylovLinearSolver")
        with pytest.raises(Exception):
            KrylovLinearSolver(type_name="LinearSolver")


# ── NavierStokesSolver with different linear solvers ─────────────────────────


class TestNavierStokesLinearSolverTypes:
    def test_default_is_linear_solver(self):
        ns = NavierStokesSolver()
        assert isinstance(ns.linear_solver, LinearSolver)
        assert not isinstance(ns.linear_solver, KrylovLinearSolver)

    def test_accepts_krylov_linear_solver(self):
        ns = NavierStokesSolver(linear_solver=KrylovLinearSolver())
        assert isinstance(ns.linear_solver, KrylovLinearSolver)

    def test_accepts_plain_linear_solver(self):
        ns = NavierStokesSolver(linear_solver=LinearSolver(max_iterations=50))
        assert isinstance(ns.linear_solver, LinearSolver)
        assert ns.linear_solver.max_iterations == 50

    def test_plain_solver_no_cap_on_max_iterations(self):
        ns = NavierStokesSolver(linear_solver=LinearSolver(max_iterations=100))
        assert ns.linear_solver.max_iterations == 100

    def test_krylov_default_max_iterations_is_15(self):
        ns = NavierStokesSolver(linear_solver=KrylovLinearSolver())
        assert ns.linear_solver.max_iterations == 15

    def test_krylov_explicit_max_iterations_respected(self):
        ns = NavierStokesSolver(linear_solver=KrylovLinearSolver(max_iterations=30))
        assert ns.linear_solver.max_iterations == 30

    def test_plain_solver_default_max_iterations_is_30(self):
        ns = NavierStokesSolver()
        assert ns.linear_solver.max_iterations == 30

    def test_dict_with_type_name_krylov(self):
        ns = NavierStokesSolver(
            linear_solver={"type_name": "KrylovLinearSolver", "max_iterations": 10}
        )
        assert isinstance(ns.linear_solver, KrylovLinearSolver)
        assert ns.linear_solver.max_iterations == 10

    def test_dict_with_type_name_linear(self):
        ns = NavierStokesSolver(linear_solver={"type_name": "LinearSolver", "max_iterations": 80})
        assert isinstance(ns.linear_solver, LinearSolver)
        assert not isinstance(ns.linear_solver, KrylovLinearSolver)
        assert ns.linear_solver.max_iterations == 80

    def test_dict_without_type_name_defaults_to_linear(self):
        ns = NavierStokesSolver(linear_solver={"max_iterations": 40})
        assert isinstance(ns.linear_solver, LinearSolver)
        assert not isinstance(ns.linear_solver, KrylovLinearSolver)

    def test_dict_without_type_name_krylov_fields_detected(self):
        ns = NavierStokesSolver(
            linear_solver={"max_preconditioner_iterations": 30, "max_iterations": 10}
        )
        assert isinstance(ns.linear_solver, KrylovLinearSolver)

    def test_line_search_allowed_with_krylov(self):
        ns = NavierStokesSolver(linear_solver=KrylovLinearSolver(), line_search=LineSearch())
        assert ns.line_search is not None

    def test_line_search_rejected_with_plain_linear_solver(self):
        with pytest.raises(ValueError, match="line_search can only be set"):
            NavierStokesSolver(linear_solver=LinearSolver(), line_search=LineSearch())

    def test_line_search_default_is_none(self):
        ns = NavierStokesSolver()
        assert ns.line_search is None

    def test_line_search_none_with_krylov_is_ok(self):
        ns = NavierStokesSolver(linear_solver=KrylovLinearSolver())
        assert ns.line_search is None


# ── Simulation-level Krylov restrictions ─────────────────────────────────────


class TestKrylovSimulationRestrictions:
    def test_error_krylov_with_limit_velocity(self):
        with pytest.raises(ValueError, match="limit_velocity"):
            _make_sim_params(
                navier_stokes_solver=NavierStokesSolver(
                    linear_solver=KrylovLinearSolver(), limit_velocity=True
                ),
            )

    def test_error_krylov_with_limit_pressure_density(self):
        with pytest.raises(ValueError, match="limit_pressure_density"):
            _make_sim_params(
                navier_stokes_solver=NavierStokesSolver(
                    linear_solver=KrylovLinearSolver(), limit_pressure_density=True
                ),
            )

    def test_error_krylov_with_unsteady(self):
        with SI_unit_system:
            with pytest.raises(ValueError, match="Unsteady"):
                _make_sim_params(
                    navier_stokes_solver=NavierStokesSolver(linear_solver=KrylovLinearSolver()),
                    time_stepping=Unsteady(steps=100, step_size=0.1),
                )

    def test_krylov_with_steady_is_ok(self):
        param = _make_sim_params(
            navier_stokes_solver=NavierStokesSolver(linear_solver=KrylovLinearSolver()),
            time_stepping=Steady(),
        )
        assert param is not None

    def test_plain_solver_with_limiters_is_ok(self):
        param = _make_sim_params(
            navier_stokes_solver=NavierStokesSolver(
                limit_velocity=True, limit_pressure_density=True
            ),
        )
        assert param is not None


# ── Translator: Krylov field handling ────────────────────────────────────────


class TestKrylovTranslation:
    def test_krylov_enabled_includes_fields(self):
        param = _make_sim_params(
            navier_stokes_solver=NavierStokesSolver(
                linear_solver=KrylovLinearSolver(), line_search=LineSearch()
            ),
        )
        translated = get_solver_json(param, mesh_unit=1 * u.m)
        ns = translated["navierStokesSolver"]
        ls = ns["linearSolver"]

        assert ls["maxPreconditionerIterations"] == 25
        assert ls["krylovRelativeTolerance"] == 0.05
        assert ls["maxIterations"] == 15
        assert "lineSearch" not in ls
        assert "lineSearch" in ns
        assert ns["lineSearch"]["residualGrowthThreshold"] == 0.85
        assert ns["lineSearch"]["maxResidualGrowth"] == 1.1
        assert ns["lineSearch"]["activationStep"] == 100

    def test_plain_solver_has_no_krylov_fields(self):
        param = _make_sim_params(
            navier_stokes_solver=NavierStokesSolver(),
        )
        translated = get_solver_json(param, mesh_unit=1 * u.m)
        ns = translated["navierStokesSolver"]
        ls = ns.get("linearSolver", {})

        assert "maxPreconditionerIterations" not in ls
        assert "krylovRelativeTolerance" not in ls
        assert "lineSearch" not in ns
        assert "useKrylovSolver" not in ns

    def test_type_name_in_translated_json(self):
        param = _make_sim_params(
            navier_stokes_solver=NavierStokesSolver(linear_solver=KrylovLinearSolver()),
        )
        translated = get_solver_json(param, mesh_unit=1 * u.m)
        ls = translated["navierStokesSolver"]["linearSolver"]
        assert ls["typeName"] == "KrylovLinearSolver"
        assert "type_name" not in ls

    def test_krylov_without_line_search_no_line_search_in_json(self):
        param = _make_sim_params(
            navier_stokes_solver=NavierStokesSolver(linear_solver=KrylovLinearSolver()),
        )
        translated = get_solver_json(param, mesh_unit=1 * u.m)
        ns = translated["navierStokesSolver"]
        assert "lineSearch" not in ns

    def test_krylov_user_overrides_in_json(self):
        param = _make_sim_params(
            navier_stokes_solver=NavierStokesSolver(
                linear_solver=KrylovLinearSolver(
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
