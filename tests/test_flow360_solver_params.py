from flow360.component.flow360_solver_params import (
    Flow360MeshParams,
    Flow360Params,
    MeshBoundary,
    TimeStepping,
)


def test_boundary():
    assert MeshBoundary.parse_raw(
        """
        {
        "noSlipWalls": [
            "fluid/fuselage",
            "fluid/leftWing",
            "fluid/rightWing"
        ]
    }
        """
    )
    assert MeshBoundary.parse_raw(
        """
        {
        "noSlipWalls": [
            1,
            2,
            3
        ]
    }
        """
    )


def test_flow360param():
    assert Flow360MeshParams.parse_raw(
        """
    {
    "boundaries": {
        "noSlipWalls": [
            "fluid/fuselage",
            "fluid/leftWing",
            "fluid/rightWing"
        ]
    }
}
    """
    )

    assert Flow360MeshParams.parse_raw(
        """
        {
        "boundaries": {
            "noSlipWalls": [
                1,
                2,
                3
            ]
        }
    }
        """
    )


def test_flow360param():
    mesh = Flow360Params.parse_raw(
        """
        {
    "boundaries": {
        "fluid/fuselage": {
            "type": "NoSlipWall"
        },
        "fluid/leftWing": {
            "type": "NoSlipWall"
        },
        "fluid/rightWing": {
            "type": "NoSlipWall"
        },
        "fluid/farfield": {
            "type": "Freestream"
        }
    },
    "actuatorDisks": [
        {
            "center": [
                3.6,
                -5.08354845,
                0
            ],
            "axisThrust": [
                -0.96836405,
                -0.06052275,
                0.24209101
            ],
            "thickness": 0.42,
            "forcePerArea": {
                "radius": [],
                "thrust": [],
                "circumferential": []
            }
        },
        {
            "center": [
                3.6,
                5.08354845,
                0
            ],
            "axisThrust": [
                -0.96836405,
                0.06052275,
                0.24209101
            ],
            "thickness": 0.42,
            "forcePerArea": {
                "radius": [],
                "thrust": [],
                "circumferential": []
            }
        }
    ]
}
        """
    )

    assert mesh


def test_flow360param_stepping():
    params = Flow360Params()
    assert params.time_stepping is None
    params.time_stepping = TimeStepping(max_physical_steps=100)
    assert params
