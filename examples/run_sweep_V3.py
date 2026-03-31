"""
Run a parameter sweep of Flow360 cases from a JSON config file.

Usage:
    python run_sweep_V3.py [config_file]

If config_file is not provided, defaults to ./run_config/config.json.

Config fields:
    parent_case_id  : ID of the parent case to fork or use as template
    version         : solver version string (e.g. "release-25.9")
    root            : root folder for output (e.g. "release_test")
    caseID          : case label used in the output ID file name
    caseIDpost      : suffix for the case ID file and case names (config key "casepost" kept for compatibility)
    caseIDpre       : prefix for case names (config key "casepre" kept for compatibility)
    sweepvar        : sweep variable, one of "AOA", "RPM", "VELOCITY"
    sweepvalue      : list of values to sweep over
    warmstart       : if true, fork from parent_case for each run
    GAI             : if true, enable geometry AI (use_geometry_AI=True, use_betamesher=True); default false
    feature_krylov  : if true, switch Navier-Stokes solver to KrylovLinearSolver; default false
    feature_gravity : if true, enable gravitational body force (fl.Gravity()) on the Fluid model; default false
    limit_velocity  : if true, enable limit_velocity on the Navier-Stokes solver; default false
    testflag        : if true, overlay wind-tunnel reference data in post-processing figures; default false
"""

import json
import os
import sys

import flow360 as fl
from flow360 import u


def load_config(config_file):
    with open(config_file, "r") as f:
        return json.load(f)



def main():
    config_file = sys.argv[1] if len(sys.argv) > 1 else "./run_config/config.json"
    config = load_config(config_file)

    parent_case_id = config["parent_case_id"]
    version        = config["version"]
    root           = config["root"]
    caseID         = config["caseID"]
    caseIDpost     = config["casepost"]
    casepre        = config["casepre"]
    sweepvar       = config["sweepvar"]
    sweepvalue     = config["sweepvalue"]
    warmstart      = config["warmstart"]
    anticlock       = config.get("anticlock", False)
    GAI             = config.get("GAI", False)
    feature_krylov  = config.get("feature_krylov", False)
    feature_gravity = config.get("feature_gravity", False)
    limit_velocity         = config.get("limit_velocity", False)
    limit_pressure_density = config.get("limit_pressure_density", False)
    pseudo_step            = config.get("pseudo_step", None)

    parent_case = fl.Case.from_cloud(parent_case_id)
    param = parent_case.params

    ss = param.operating_condition.thermal_state.speed_of_sound
    print("speed of sound:", ss)

    project = fl.Project.from_cloud(parent_case.project_id)

    if feature_krylov:
        fluid_models = [m for m in param.models if isinstance(m, fl.Fluid)]
        if not fluid_models:
            raise ValueError("No Fluid model found in param.models for feature_krylov.")
        for m in fluid_models:
            m.navier_stokes_solver.linear_solver = fl.KrylovLinearSolver()
        print("feature_krylov=True: NavierStokesSolver linear_solver set to KrylovLinearSolver")

    if feature_gravity:
        fluid_models = [m for m in param.models if isinstance(m, fl.Fluid)]
        if not fluid_models:
            raise ValueError("No Fluid model found in param.models for feature_gravity.")
        for m in fluid_models:
            m.gravity = fl.Gravity()
        print("feature_gravity=True: Gravity() applied to Fluid model(s)")

    fluid_models = [m for m in param.models if isinstance(m, fl.Fluid)]
    if not fluid_models:
        raise ValueError("No Fluid model found in param.models.")
    for m in fluid_models:
        m.navier_stokes_solver.limit_velocity = limit_velocity
        m.navier_stokes_solver.limit_pressure_density = limit_pressure_density
    print(f"limit_velocity={limit_velocity}, limit_pressure_density={limit_pressure_density}: NavierStokesSolver limits set")

    if pseudo_step is not None:
        param.time_stepping.max_steps = pseudo_step
        print(f"pseudo_step={pseudo_step}: time_stepping.max_steps set")

    krylov_tag  = "krylov_"  if feature_krylov  else ""
    gravity_tag = "gravity_" if feature_gravity else ""

    caseIDfiles_dir = os.path.join(root, "caseIDfiles")
    if not os.path.exists(caseIDfiles_dir):
        os.makedirs(caseIDfiles_dir)
    caseIDfile = os.path.join(caseIDfiles_dir, f"{caseID}_{version}_{caseIDpost}.txt")
    print(f"Writing case IDs to: {caseIDfile}")
    print(f"warmstart={warmstart}")

    with open(caseIDfile, "w") as f:
        if sweepvar == "AOA":
            for AOA in sweepvalue:
                param.operating_condition.alpha = AOA * u.degree
                casename = f"{sweepvar}{AOA}_{krylov_tag}{gravity_tag}{caseIDpost}_{version}"
                print(casename)
                if warmstart:
                    #case = project.run_case(params=param, fork_from=parent_case, name=casename, solver_version=version, tags=["!80.XA100"])
                    case = project.run_case(params=param, fork_from=parent_case, name=casename,
                                            solver_version=version,
                                            use_geometry_AI=GAI, use_betamesher=GAI)
                else:
                    #case = project.run_case(params=param, name=casename, solver_version=version, use_geometry_AI=False, tags=["!80.XA100"])
                    case = project.run_case(params=param, name=casename, solver_version=version,
                                            use_geometry_AI=GAI, use_betamesher=GAI)
                # Flow360 returns the existing case if the same setup was already submitted
                parent_case = case
                print(case.id, file=f)

        elif sweepvar == "RPM":
            rotation_models = [m for m in param.models if isinstance(m, fl.Rotation)]
            if not rotation_models:
                raise ValueError("No Rotation model found in param.models for RPM sweep.")
            for RPM in sweepvalue:
                for m in rotation_models:
                    if anticlock:
                        m.spec.value = -RPM * u.rpm
                    else:
                        m.spec.value = RPM * u.rpm
                casename = f"{sweepvar}{RPM}_{krylov_tag}{gravity_tag}{caseIDpost}_{version}"
                print(casename)
                #case = project.run_case(params=param, name=casename, solver_version=version, tags=["!80.XA100"])
                case = project.run_case(params=param, name=casename, solver_version=version,
                                        use_geometry_AI=GAI, use_betamesher=GAI)
                # Flow360 returns the existing case if the same setup was already submitted
                print(case.id, file=f)

        elif sweepvar == "VELOCITY":
            for speed in sweepvalue:
                with u.SI_unit_system:
                    param.operating_condition = fl.AerospaceCondition(
                        velocity_magnitude=speed * fl.u.m / fl.u.s,
                        alpha=param.operating_condition.alpha,
                        thermal_state=param.operating_condition.thermal_state,
                        reference_velocity_magnitude=param.operating_condition.reference_velocity_magnitude,
                    )
                casename = f"{sweepvar}_{speed}_{krylov_tag}{gravity_tag}{caseIDpost}_{version}"
                print(casename)
                #case = project.run_case(params=param, name=casename, solver_version=version, tags=["!80.XA100"])
                case = project.run_case(params=param, name=casename, solver_version=version,
                                        use_geometry_AI=GAI, use_betamesher=GAI)
                # Flow360 returns the existing case if the same setup was already submitted
                print(case.id, file=f)

        else:
            raise ValueError(f"Unknown sweepvar: {sweepvar!r}. Expected 'AOA', 'RPM', or 'VELOCITY'.")


if __name__ == "__main__":
    main()
