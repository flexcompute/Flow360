from typing import Any, Optional, List
from pydantic import BaseModel, NonNegativeInt
from pylatex import NoEscape, Package, Tabular
from numbers import Number

from flow360 import Case


def check_landscape(doc):
    for package in doc.packages:
        if "geometry" in str(package.arguments):
            if "landscape" in str(package.options):
                return True
            else:
                return False


def get_case_from_id(id: str, cases: list[Case]) -> Case:
    # This can happen if Delta has no ref_case
    if len(cases) == 0:
        raise ValueError("No cases provided for `get_case_from_id`.")
    for case in cases:
        if case.id == id:
            return case


def data_from_path(case: Case, path: str, cases: list[Case] = []) -> Any:
    # Handle Delta values
    if isinstance(path, Delta):
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
            elif "value" in case:
                return case["value"]
            else:
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
        except KeyError:
            raise ValueError(f"Could not find path component: '{component}'")

    # Case variable is slightly misleading as this is only a case on the first iteration
    for component in path_components:
        case = _search_path(case, component)

    return case


class Delta(BaseModel):
    data_path: str
    ref_index: Optional[NonNegativeInt] = 0

    def calculate(self, case: Case, cases: List[Case]) -> float:
        # Used when trying to do a Delta in a case_by_case or if ref ID is bad
        if self.ref_index is None or self.ref_index >= len(cases):
            return "Ref not found."
        ref = cases[self.ref_index]
        case_result = data_from_path(case, self.data_path)
        ref_result = data_from_path(ref, self.data_path)
        return case_result - ref_result

    __str__ = lambda self: f"Delta {self.data_path.split('/')[-1]}"


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
