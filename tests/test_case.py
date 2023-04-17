from flow360 import Case, Flow360Params
from flow360.log import log

from .utils import mock_id


def test_case():
    case = Case.create(name="hi", params=Flow360Params(), volume_mesh_id=mock_id)
    case_2 = case.copy()
    case_3 = case.retry()
    case_4 = case.fork()
    case_5 = case.continuation()
    print(case_5)
