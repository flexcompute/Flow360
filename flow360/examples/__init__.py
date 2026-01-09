from .actuator_disk import ActuatorDisk
from .airplane import Airplane
from .bet_evtol import BETEVTOL
from .bet_example_data import BETExampleData
from .bet_line import BETLine
from .convergence import Convergence
from .cube import Cube
from .cylinder2D import Cylinder2D
from .cylinder3D import Cylinder3D
from .DARPA import DARPA_SUBOFF
from .drivaer import DrivAer
from .evtol import EVTOL
from .f1_2025 import F1_2025
from .isolated_propeller import IsolatedPropeller
from .monitors import MonitorsAndSlices
from .NLF_airfoil import NLFAirfoil2D
from .oblique_channel import ObliqueChannel
from .om6wing import OM6wing
from .quadcopter import Quadcopter
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
from .tutorial_UDD_structural import TutorialUDDStructural
from .XV15_csm import XV15_CSM
from .windsor import Windsor

__all__ = [
    "ActuatorDisk",
    "Airplane",
    "BETEVTOL",
    "BETExampleData",
    "BETLine",
    "Convergence",
    "Cube",
    "Cylinder2D",
    "Cylinder3D",
    "DARPA_SUBOFF",
    "DrivAer",
    "EVTOL",
    "F1_2025",
    "IsolatedPropeller",
    "MonitorsAndSlices",
    "NLFAirfoil2D",
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
    "TutorialUDDStructural",
    "Quadcopter",
    "XV15_CSM",
    "ObliqueChannel",
    "Windsor",
]
