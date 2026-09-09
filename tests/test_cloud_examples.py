"""Unit tests for cloud examples functionality"""

import pytest

from flow360.cloud.responses import ExampleItem
from flow360.component.cloud_examples import find_example_by_name
from flow360.exceptions import Flow360ValueError


@pytest.fixture
def sample_examples():
    """Fixture providing sample example data"""
    return [
        ExampleItem(
            tags=["Steady", "k-Omega SST"],
            id="eedc5f6d-f877-4b30-b307-0d6ef4713f78",
            type="flow360-project-examples",
            resourceId="prj-7fb80c26-6565-4ea5-97b6-9bf5e87882f2",
            s3path="flow360-project-examples/images/eedc5f6d-f877-4b30-b307-0d6ef4713f78_1764590248025.jpeg",
            title="DrivAer Steady",
            createdAt="2025-12-01T11:57:27.625338Z",
            order=0,
        ),
        ExampleItem(
            tags=["AngleExpression", "Steady", "Unsteady"],
            id="9e59d33e-f84c-4c8f-8aa9-02683be8dc58",
            type="flow360-project-examples",
            resourceId="prj-345c32ae-d80e-4753-a017-dbdf82884709",
            s3path="flow360-project-examples/images/9e59d33e-f84c-4c8f-8aa9-02683be8dc58_1764590348065.jpeg",
            title="Calculating Dynamic Derivatives using Sliding Interfaces",
            createdAt="2025-12-01T11:59:07.654252Z",
            order=1,
        ),
        ExampleItem(
            tags=["steady", "SA"],
            id="9bcf55e0-55c4-4182-9814-2147870cbd00",
            type="flow360-project-examples",
            resourceId="prj-261fdfa7-025e-40d7-9c5d-52c0f206f782",
            s3path="flow360-project-examples/images/9bcf55e0-55c4-4182-9814-2147870cbd00_1727702467098.jpeg",
            title="Simple Airplane",
            createdAt="2024-09-16T10:02:54.267602Z",
            order=2,
        ),
        ExampleItem(
            tags=["SA", "quasi-3D model"],
            id="9e1ef593-ee6e-4817-bc33-2473d205a2a9",
            type="flow360-project-examples",
            resourceId="prj-36b4faeb-f55b-4899-803f-35b6739a818b",
            s3path="flow360-project-examples/images/9e1ef593-ee6e-4817-bc33-2473d205a2a9_1765382688440.jpeg",
            title="NACA 0012 Low Speed Airfoil",
            createdAt="2025-12-10T16:04:48.118213Z",
            order=3,
        ),
        ExampleItem(
            tags=["BET", "DDES", "unsteady", "SA"],
            id="3fc65cbd-c474-4a1f-b986-5716d14bec38",
            type="flow360-project-examples",
            resourceId="prj-207be482-26c0-4211-9834-753ae5ff3149",
            s3path="flow360-project-examples/images/3fc65cbd-c474-4a1f-b986-5716d14bec38_1727790871461.jpeg",
            title="eVTOL with BET line",
            createdAt="2024-10-01T13:54:31.028901Z",
            order=5,
        ),
        ExampleItem(
            tags=["DDES", "rotation", "unsteady", "SA"],
            id="804b41ca-605b-49da-91b7-87c8567d8f63",
            type="flow360-project-examples",
            resourceId="prj-a0dd6591-7f9e-4436-a676-a9b80d68d103",
            s3path="flow360-project-examples/images/804b41ca-605b-49da-91b7-87c8567d8f63_1727785798140.jpeg",
            title="Isolated Propeller",
            createdAt="2024-10-01T12:29:57.702452Z",
            order=6,
        ),
        ExampleItem(
            tags=["Conjugate Heat Transfer", "SA", "Steady"],
            id="3ce2a6c2-2c9c-49f7-89db-6c183a484ee8",
            type="flow360-project-examples",
            resourceId="prj-90433d78-a5a9-4b95-b748-d58072576fb2",
            s3path="flow360-project-examples/images/3ce2a6c2-2c9c-49f7-89db-6c183a484ee8_1765382630413.jpeg",
            title="Conjugate Heat Transfer for Cooling Fins",
            createdAt="2025-12-10T16:03:50.127476Z",
            order=7,
        ),
    ]


def test_find_example_exact_match(sample_examples):
    """Test finding example with exact match"""
    matched, score = find_example_by_name("Simple Airplane", sample_examples)
    assert matched.title == "Simple Airplane"
    assert matched.id == "9bcf55e0-55c4-4182-9814-2147870cbd00"
    assert score == 1.0


def test_find_example_case_insensitive(sample_examples):
    """Test finding example with case-insensitive matching"""
    matched, score = find_example_by_name("simple airplane", sample_examples)
    assert matched.title == "Simple Airplane"
    assert score == 1.0

    matched, score = find_example_by_name("SIMPLE AIRPLANE", sample_examples)
    assert matched.title == "Simple Airplane"
    assert score == 1.0

    matched, score = find_example_by_name("DrivAer Steady", sample_examples)
    assert matched.title == "DrivAer Steady"
    assert score == 1.0


def test_find_example_with_typo(sample_examples):
    """Test finding example with typo"""
    matched, score = find_example_by_name("Simple Airplne", sample_examples)
    assert matched.title == "Simple Airplane"
    assert score < 1.0
    assert score >= 0.3

    matched, score = find_example_by_name("DrivAer Stedy", sample_examples)
    assert matched.title == "DrivAer Steady"
    assert score < 1.0
    assert score >= 0.3


def test_find_example_partial_match(sample_examples):
    """Test finding example with partial match"""
    matched, score = find_example_by_name("Simple", sample_examples)
    assert matched.title == "Simple Airplane"
    assert score >= 0.3

    matched, score = find_example_by_name("Airplane", sample_examples)
    assert matched.title == "Simple Airplane"
    assert score >= 0.3


def test_find_example_long_title(sample_examples):
    """Test finding example with long title"""
    matched, score = find_example_by_name("Calculating Dynamic Derivatives", sample_examples)
    assert matched.title == "Calculating Dynamic Derivatives using Sliding Interfaces"
    assert score >= 0.3

    matched, score = find_example_by_name("Dynamic Derivatives", sample_examples)
    assert matched.title == "Calculating Dynamic Derivatives using Sliding Interfaces"
    assert score >= 0.3


def test_find_example_whitespace_handling(sample_examples):
    """Test finding example with extra whitespace"""
    matched, score = find_example_by_name("  Simple Airplane  ", sample_examples)
    assert matched.title == "Simple Airplane"
    assert score == 1.0


def test_find_example_no_match(sample_examples):
    """Test that no match raises appropriate error"""
    with pytest.raises(Flow360ValueError) as exc_info:
        find_example_by_name("xyzabc123def456", sample_examples)
    assert "No matching example found" in str(exc_info.value)
    assert "xyzabc123def456" in str(exc_info.value)


def test_find_example_poor_match(sample_examples):
    """Test that very poor match raises error"""
    with pytest.raises(Flow360ValueError) as exc_info:
        find_example_by_name("xyzabc123", sample_examples)
    assert "No matching example found" in str(exc_info.value)


def test_find_example_empty_list():
    """Test that empty examples list raises error"""
    with pytest.raises(Flow360ValueError) as exc_info:
        find_example_by_name("Simple Airplane", [])
    assert "No examples available" in str(exc_info.value)


def test_find_example_best_match_selection(sample_examples):
    """Test that the best match is selected when multiple partial matches exist"""
    matched, score = find_example_by_name("Propeller", sample_examples)
    assert matched.title == "Isolated Propeller"
    assert score >= 0.3

    matched, score = find_example_by_name("Heat Transfer", sample_examples)
    assert matched.title == "Conjugate Heat Transfer for Cooling Fins"
    assert score >= 0.3
