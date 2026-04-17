import flow360.component.simulation.units as u
from flow360.component.simulation.models.solver_numerics import (
    KrylovLinearSolver,
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
from flow360.component.simulation.time_stepping.time_stepping import Steady
from flow360.component.simulation.translator.solver_translator import get_solver_json
from flow360.component.simulation.unit_system import SI_unit_system


def _make_sim_params(navier_stokes_solver=None):
    with SI_unit_system:
        ns = navier_stokes_solver or NavierStokesSolver()
        return SimulationParams(
            operating_condition=AerospaceCondition.from_mach(mach=0.5),
            models=[
                Fluid(navier_stokes_solver=ns),
                Wall(entities=Surface(name="fluid/wall")),
                Freestream(entities=Surface(name="fluid/farfield")),
            ],
            time_stepping=Steady(),
        )


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
        assert ls["relativeTolerance"] == 0.05
        assert ls["maxIterations"] == 15
        assert "lineSearch" not in ls
        assert "lineSearch" in ns
        assert ns["lineSearch"]["residualGrowthThreshold"] == 0.85
        assert ns["lineSearch"]["maxResidualGrowth"] == 1.1
        assert ns["lineSearch"]["activationStep"] == 100

    def test_plain_solver_has_no_krylov_fields(self):
        param = _make_sim_params(navier_stokes_solver=NavierStokesSolver())
        translated = get_solver_json(param, mesh_unit=1 * u.m)
        ns = translated["navierStokesSolver"]
        ls = ns.get("linearSolver", {})

        assert "maxPreconditionerIterations" not in ls
        assert "relativeTolerance" not in ls
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
                    relative_tolerance=0.1,
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
        assert ls["relativeTolerance"] == 0.1
        assert ns["lineSearch"]["residualGrowthThreshold"] == 0.9
        assert ns["lineSearch"]["maxResidualGrowth"] == 1.5
        assert ns["lineSearch"]["activationStep"] == 200
