"""
report item
"""

from numbers import Number
from typing import Any, List, Optional

from pydantic import BaseModel, NonNegativeInt

# this plugin is optional, thus pylatex is not required: TODO add handling of installation of pylatex
# pylint: disable=import-error
from pylatex import NoEscape, Package, Tabular

from flow360 import Case
from flow360.component.results import case_results
from flow360.log import log

# pylint: disable=protected-access
_requirements_mapping = {
    "params": "simulation.json",
    "total_forces": case_results.TotalForcesResultCSVModel()._remote_path(),
    "nonlinear_residuals": case_results.NonlinearResidualsResultCSVModel()._remote_path(),
    "x_slicing_force_distribution": case_results.XSlicingForceDistributionResultCSVModel()._remote_path(),
    "y_slicing_force_distribution": case_results.YSlicingForceDistributionResultCSVModel()._remote_path()
}


def get_requirements_from_data_path(data_path):
    """
    Retrieves requirements based on data path entries by mapping root paths
    to their corresponding requirements.

    Parameters
    ----------
    data_path : iterable
        An iterable containing data path entries to be checked.

    Returns
    -------
    list
        A list of unique requirements derived from the data path.

    Raises
    ------
    ValueError
        If a root path in the data path does not have a corresponding requirement.
    """

    requirements = set()
    for item in data_path:
        root_path = get_root_path(item)
        requirement = _requirements_mapping.get(root_path)
        if requirement is None:
            raise ValueError(f"Unknown result type: {item}")
        requirements.add(requirement)
    return list(requirements)


def check_landscape(doc):
    """
    Checks if a document is in landscape orientation based on geometry package options.

    Parameters
    ----------
    doc : Document
        The LaTeX document to check for landscape orientation.

    Returns
    -------
    bool
        True if the document is in landscape orientation, False otherwise.
    """

    for package in doc.packages:
        if "geometry" in str(package.arguments):
            return "landscape" in str(package.options)
    return False


def get_case_from_id(case_id: str, cases: list[Case]) -> Case:
    """
    Retrieves a case by its unique identifier from a list of cases.

    Parameters
    ----------
    case_id : str
        The unique identifier of the case to retrieve.
    cases : list[Case]
        A list of `Case` objects to search.

    Returns
    -------
    Case
        The `Case` object matching the specified `case_id`.

    Raises
    ------
    ValueError
        If no cases are provided or the specified `case_id` is not found.
    """
    if len(cases) == 0:
        raise ValueError("No cases provided for `get_case_from_id`.")
    for case in cases:
        if case.id == case_id:
            return case
    raise ValueError(f"{case_id=} not found in {cases=}")


def get_root_path(data_path):
    """
    Extracts the root path from a given data path.

    If the provided `data_path` is of type `Delta`, the function retrieves
    the path from `data_path.data`.

    Parameters
    ----------
    data_path : str or Delta or None
        The data path to parse or a `Delta` object containing a data path.

    Returns
    -------
    str or None
        The root path as a string if `data_path` is valid, otherwise `None`.
    """

    if data_path is not None:
        if isinstance(data_path, Delta):
            data_path = data_path.data
        return data_path.split("/")[0]
    return None


# pylint: disable=too-many-return-statements
def data_from_path(
    case: Case, path: str, cases: list[Case] = None, case_by_case: bool = False
) -> Any:
    """
    Retrieves data from a specified path within a `Case` object, with optional delta calculations.

    Parameters
    ----------
    case : Case
        The primary `Case` object to search.
    path : str
        The path string indicating the nested attributes or dictionary keys.
    cases : list[Case], optional
        List of additional cases for delta calculations, default is an empty list.
    case_by_case : bool, default=False
        Flag for enabling case-by-case delta calculation when `path` is a `Delta` object.

    Returns
    -------
    Any
        The data extracted from the specified path within the `Case` object or calculated delta.

    Raises
    ------
    ValueError
        If a specified path component is not found or cannot be accessed.

    Notes
    -----
    This function splits the path into components and recursively searches through attributes,
    dictionary keys, or list indices as indicated in each component. Supports delta calculation
    using `Delta` objects and error handling for invalid paths.
    """
    if cases is None:
        cases = []

    if isinstance(path, Delta):
        if case_by_case:
            return path.model_copy(update={"ref_index": None}).calculate(case, cases)
        return path.calculate(case, cases)

    # Split path into components
    path_components = path.split("/")

    def _search_path(case: Case, component: str) -> Any:
        """
        Case starts as a `Case` object but changes as it recurses through the path components
        """
        # Check if component is an attribute
        try:
            return getattr(case, component)
        except AttributeError:
            pass

        # Check if component is an attribute of case.results
        # Convenience feature so the user doesn't have to include "results" in path
        try:
            return getattr(case.results, component)
        except AttributeError:
            pass

        # Check if component is a key for a dictionary
        try:
            case = case[component]
            # Have to test for int or str here otherwise...
            if isinstance(case, (int, str)):
                return case
            # .. this raises a KeyError.
            # This is a convenience that may be removed for if people want something other than the value
            if "value" in case:
                return case["value"]
            return case
        except TypeError:
            pass

        # Check if case is a list and interpret component as an int index
        # E.g. in user defined functions
        if isinstance(case, list):
            try:
                return case[int(component)]
            except (ValueError, IndexError):
                pass

        # Check if case is a number
        if isinstance(case, Number):
            return case

        # Check if component is a key of a value
        try:
            return case.values[component]
        except KeyError as err:
            raise ValueError(
                f"Could not find path component: '{component}', available: {case.values.keys()}"
            ) from err
        except AttributeError:
            log.warning(f"unknown value for path: {case=}, {component=}")

        return None

    # Case variable is slightly misleading as this is only a case on the first iteration
    for component in path_components:
        case = _search_path(case, component)

    return case


class Delta(BaseModel):
    """
    Represents a delta calculation between a reference case and a target case based on specified data.

    Parameters
    ----------
    data : str
        Path to the data item used for delta calculation.
    ref_index : Optional[NonNegativeInt], default=0
        Index of the reference case in the list of cases for comparison.
    """

    data: str
    ref_index: Optional[NonNegativeInt] = 0

    def calculate(self, case: Case, cases: List[Case]) -> float:
        """
        Calculates the delta between the specified case and the reference case.

        Parameters
        ----------
        case : Case
            The target case for which the delta is calculated.
        cases : List[Case]
            A list of available cases, including the reference case.

        Returns
        -------
        float
            The computed delta value between the case and reference case data.

        Raises
        ------
        ValueError
            If `ref_index` is out of bounds or `None`, indicating a missing reference.
        """

        if self.ref_index is None or self.ref_index >= len(cases):
            return "Ref not found."
        ref = cases[self.ref_index]
        case_result = data_from_path(case, self.data)
        ref_result = data_from_path(ref, self.data)
        return case_result - ref_result

    def __str__(self):
        return f"Delta {self.data.split('/')[-1]}"


# pylint: disable=too-few-public-methods
class Tabulary(Tabular):
    """The `tabulary` package works better than the existing pylatex implementations so this includes it in pylatex"""

    packages = [Package("tabulary")]

    def __init__(self, *args, width_argument=NoEscape(r"\linewidth"), **kwargs):
        """
        Args
        ----
        width_argument:
            The width of the table. By default the table is as wide as the
            text.
        """
        super().__init__(*args, start_arguments=width_argument, **kwargs)
