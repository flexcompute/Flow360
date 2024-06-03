import flow360
from flow360 import Flow360Params
from flow360.component.flow360_params.services import params_to_dict

data = {
    "unitSystem": {"name": "Flow360"},
    "version": "0.2.0b18",
    "geometry": {
        "refArea": {"value": 45.604, "units": "flow360_area_unit"},
        "momentCenter": {"value": [0, 0, 0], "units": "flow360_length_unit"},
        "momentLength": {"value": [3.81, 3.81, 3.81], "units": "flow360_length_unit"},
    },
    "boundaries": {
        "farField/farField": {"type": "Freestream"},
        "innerRotating/blade": {"type": "NoSlipWall"},
    },
    "timeStepping": {
        "maxPseudoSteps": 35,
        "CFL": {
            "type": "adaptive",
            "min": 0.1,
            "max": 10000,
            "maxRelativeChange": 1,
            "convergenceLimitingFactor": 0.25,
        },
        "modelType": "Unsteady",
        "physicalSteps": 120,
        "timeStepSize": {"value": 0.2835, "units": "flow360_time_unit"},
    },
    "turbulenceModelSolver": {
        "absoluteTolerance": 1e-8,
        "relativeTolerance": 0.01,
        "modelType": "SpalartAllmaras",
        "updateJacobianFrequency": 4,
        "equationEvalFrequency": 1,
        "maxForceJacUpdatePhysicalSteps": 0,
        "orderOfAccuracy": 2,
        "DDES": True,
        "gridSizeForLES": "maxEdgeLength",
        "quadraticConstitutiveRelation": False,
        "linearSolverConfig": {"maxIterations": 25, "absoluteTolerance": 1e-10},
        "rotationCorrection": True,
    },
    "freestream": {
        "modelType": "FromMach",
        "alphaAngle": -90,
        "betaAngle": 0,
        "Mach": 0.0146972,
        "MachRef": 0.7,
        "muRef": 4.29279e-8,
        "Temperature": 288.15,
    },
    "surfaceOutput": {
        "animationFrequency": 10000,
        "outputFormat": "paraview",
        "outputFields": [
            "Cp",
            "primitiveVars",
            "Cf",
            "CfNormal",
            "CfTangent",
            "yPlus",
            "nodeForcesPerUnitArea",
        ],
    },
    "volumeOutput": {
        "animationFrequency": 10000,
        "outputFormat": "paraview",
        "computeTimeAverages": False,
        "startAverageIntegrationStep": 100000,
        "outputFields": ["Cp", "Mach", "primitiveVars", "qcriterion", "T"],
    },
    "volumeZones": {
        "innerRotating": {
            "modelType": "FluidDynamics",
            "referenceFrame": {
                "modelType": "OmegaRadians",
                "omegaRadians": 0.184691,
                "centerOfRotation": {"value": [0, 0, 0], "units": "flow360_length_unit"},
                "axisOfRotation": [0, 0, -1],
            },
        }
    },
    "navierStokesSolver": {
        "absoluteTolerance": 1e-9,
        "relativeTolerance": 0.01,
        "CFLMultiplier": 1,
        "kappaMUSCL": -1,
        "updateJacobianFrequency": 4,
        "equationEvalFrequency": 1,
        "maxForceJacUpdatePhysicalSteps": 0,
        "orderOfAccuracy": 2,
        "numericalDissipationFactor": 1,
        "limitVelocity": False,
        "limitPressureDensity": False,
        "linearSolverConfig": {"maxIterations": 35, "absoluteTolerance": 1e-10},
    },
}

with flow360.flow360_unit_system:
    params = Flow360Params(**data)

    dictionary = params_to_dict(params)

    print(dictionary)
