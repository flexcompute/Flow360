from .actuator_disk import ActuatorDisk
from .airplane import Airplane
from .bet_disk import BETDisk
from .bet_line import BETLine
from .convergence import Convergence
from .cylinder2D import Cylinder2D
from .cylinder3D import Cylinder3D
from .monitors import MonitorsAndSlices
from .om6wing import OM6wing
from .rotating_spheres import RotatingSpheres
from .tutorial_2D_30p30n import Tutorial2D30p30n
from .tutorial_2D_crm import Tutorial2DCRM
from .tutorial_2D_gaw2 import Tutorial2DGAW2
from .tutorial_auto_meshing_internal_flow import TutorialAutoMeshingInternalFlow
from .tutorial_bet_disk import TutorialBETDisk
from .tutorial_cht_solver import TutorialCHTSolver
from .tutorial_dynamic_derivatives import TutorialDynamicDerivatives
from .tutorial_periodic_BC import TutorialPeriodicBC
from .tutorial_RANS_xv15 import TutorialRANSXv15
from .tutorial_UDD_forces_moments import TutorialUDDForcesMoments

__all__ = [
    "ActuatorDisk",
    "Airplane",
    "BETDisk",
    "BETLine",
    "Convergence",
    "Cylinder2D",
    "Cylinder3D",
    "MonitorsAndSlices",
    "OM6wing",
    "RotatingSpheres",
    "Tutorial2DCRM",
    "Tutorial2D30p30n",
    "Tutorial2DGAW2",
    "TutorialBETDisk",
    "TutorialCHTSolver",
    "TutorialPeriodicBC",
    "TutorialAutoMeshingInternalFlow",
    "TutorialDynamicDerivatives",
    "TutorialRANSXv15",
    "TutorialUDDForcesMoments",
]
