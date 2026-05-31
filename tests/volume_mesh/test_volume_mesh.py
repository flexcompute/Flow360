import pytest

import flow360 as fl


@pytest.mark.usefixtures("s3_download_override")
def test_bounding_box():

    vm = fl.VolumeMesh(id="vm-1cfdec99-3ce3-428c-85f8-2054812b2ddc")

    assert vm.bounding_box.length == 1340.713806228357
    assert vm.bounding_box.width == 1340.6863368189502
    assert vm.bounding_box.height == 1341.071651886667

    vm.bounding_box.filter(include=["rotating*"])
    assert vm.bounding_box.length == 15.99937337657778
    assert vm.bounding_box.width == 15.999686685221198
    assert vm.bounding_box.height == 3.000000000000001

    vm.bounding_box.filter(include=["*/blade1"])
    assert vm.bounding_box.length == 5.690080000000009
    assert vm.bounding_box.width == 0.6854140939740105
    assert vm.bounding_box.height == 0.09453280280563113

    vm.bounding_box.filter(exclude=["*"])
    assert vm.bounding_box.length == 0
    assert vm.bounding_box.width == 0
    assert vm.bounding_box.height == 0

    vm.bounding_box.filter(exclude=["stationaryBlock/farfield"])
    assert vm.bounding_box.length == 15.99937337657778
    assert vm.bounding_box.width == 15.999686685221198
    assert vm.bounding_box.height == 3.000000000000001
