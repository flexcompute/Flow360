"""
report utils, utils.py
"""
from  __future__ import annotations
from abc import ABCMeta, abstractmethod
import os
import posixpath
from numbers import Number
from typing import Any, List, Optional, Literal, Union, Annotated

import pydantic as pd

# this plugin is optional, thus pylatex is not required: TODO add handling of installation of pylatex
# pylint: disable=import-error
from pylatex import NoEscape, Package, Tabular

from flow360 import Case
from flow360.component.simulation.framework.base_model import Flow360BaseModel, Conflicts
from flow360.component.results import case_results
from flow360.component.volume_mesh import VolumeMeshV2
from flow360.log import log

import ast
import numexpr as ne
import uuid

class RequirementItem(pd.BaseModel):
    resource_type: Literal['case', 'volume_mesh', 'surface_mesh', 'geometry'] = 'case'
    filename: str

    model_config = {"frozen": True}


# pylint: disable=protected-access
_requirements_mapping = {
    "params": RequirementItem(filename="simulation.json"),
    "total_forces": RequirementItem(filename=case_results.TotalForcesResultCSVModel()._remote_path()),
    "surface_forces": RequirementItem(filename=case_results.SurfaceForcesResultCSVModel()._remote_path()),
    "nonlinear_residuals": RequirementItem(filename=case_results.NonlinearResidualsResultCSVModel()._remote_path()),
    "x_slicing_force_distribution": RequirementItem(filename=case_results.XSlicingForceDistributionResultCSVModel()._remote_path()),
    "y_slicing_force_distribution": RequirementItem(filename=case_results.YSlicingForceDistributionResultCSVModel()._remote_path()),
    "volume_mesh": RequirementItem(resource_type='volume_mesh', filename="simulation.json"),
    "volume_mesh/stats": RequirementItem(resource_type='volume_mesh', filename=VolumeMeshV2._mesh_stats_file),
}


def get_requirements_from_data_path(data_path)->List[RequirementItem]:
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
        matched_requirement = None
        for key in _requirements_mapping:
            if root_path.startswith(key):
                matched_requirement = _requirements_mapping[key]
                requirements.add(matched_requirement)
        if matched_requirement is None:
            raise ValueError(f"Unknown result type: {item}")
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
        if isinstance(data_path, (Delta, DataItem)):
            data_path = data_path.data
        if isinstance(data_path, DataItem):
            data_path = data_path.data     
        return data_path
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
    
    if isinstance(path, DataItem):
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

        if isinstance(case, case_results.PerEntityResultCSVModel):
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


class GenericOperation(Flow360BaseModel, metaclass=ABCMeta):
    @abstractmethod
    def calculate(self, data, cases, new_variable_name):
        pass


class Average(GenericOperation):
    start_step: Optional[pd.NonNegativeInt] = None
    end_step: Optional[pd.NonNegativeInt] = None
    start_time: Optional[pd.NonNegativeFloat] = None
    end_time: Optional[pd.NonNegativeFloat] = None
    fraction: Optional[pd.PositiveFloat] = pd.Field(None, le=1)
    type_name: Literal['Average'] = pd.Field('Average', frozen=True)


    model_config = pd.ConfigDict(
        conflicting_fields=[
            Conflicts(field1="start_step", field2="start_time"),
            Conflicts(field1="start_step", field2="fraction"),
            Conflicts(field1="start_time", field2="fraction"),
            Conflicts(field1="end_step", field2="end_time"),
            Conflicts(field1="end_step", field2="fraction"),
            Conflicts(field1="end_time", field2="fraction"),
        ],
        require_one_of=["start_step", "start_time", "fraction"],
    )

    def calculate(self, data, cases, new_variable_name):
        if isinstance(data, case_results.ResultCSVModel):
            if self.fraction is None:
                raise NotImplementedError(f'Only "fraction" average method implemented.')
            averages = data.get_averages(avarage_fraction=self.fraction)
            return data, cases, averages

        raise NotImplementedError(f'{self.__class__.__name__} not implemented for data type: {type(data)=}')


class Expression(GenericOperation):
    expr: str
    type_name: Literal['Expression'] = pd.Field('Expression', frozen=True)


    @classmethod
    def get_variables(cls, expr):
        """
        Parse the expression and return a set of variable names.
        """
        tree = ast.parse(expr, mode='eval')
        return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    @classmethod
    def evaluate_expression(cls, df, expr, new_variable_name):
        """
        Evaluate the expression on the dataframe and add the result as a new column.
        """
        # Extract variable names from the expression
        variables = cls.get_variables(expr)
        
        # Check for missing variables in the dataframe
        missing_vars = variables - set(df.columns)
        if missing_vars:
            raise ValueError(f"The following variables are missing in the dataframe: {', '.join(missing_vars)}")
        
        # Prepare the local dictionary for numexpr
        local_dict = {var: df[var].values for var in variables}
        
        # Evaluate the expression safely
        try:
            result = ne.evaluate(expr, local_dict)
        except Exception as e:
            raise ValueError(f"Error evaluating expression: {e}")
        
        df[new_variable_name] = result
        return df



    def calculate(self, data, cases, new_variable_name):
        log.debug(f"evaluating expression {self.expr}")

        if isinstance(data, case_results.SurfaceForcesResultCSVModel):
            df = self.evaluate_expression(data.as_dataframe(), self.expr, new_variable_name)
            data.update(df)
            return data, cases, data.values

        raise NotImplementedError(f'{self.__class__.__name__} not implemented for data type: {type(data)=}')





OperationTypes = Annotated[Union[Average, Expression], pd.Field(discriminator="type_name")]



class Delta(pd.BaseModel):
    """
    Represents a delta calculation between a reference case and a target case based on specified data.

    Parameters
    ----------
    data : str
        Path to the data item used for delta calculation.
    ref_index : Optional[NonNegativeInt], default=0
        Index of the reference case in the list of cases for comparison.
    """

    data: Union[str, DataItem]
    ref_index: Optional[pd.NonNegativeInt] = 0
    type_name: Literal["Delta"] = pd.Field("Delta", frozen=True)

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
        if isinstance(self.data, str):
            data_str = self.data.split('/')[-1]
        else:
            data_str = str(self.data)
        return f"Delta {data_str}"


class DataItem(pd.BaseModel):
    """
    Represents a delta calculation between a reference case and a target case based on specified data.

    Parameters
    ----------
    data : str
        Path to the data item used for delta calculation.
    ref_index : Optional[NonNegativeInt], default=0
        Index of the reference case in the list of cases for comparison.
    exclude : Optional[List[str]]
        List of boundaries to exclude from data. Applicable to:
        x_slicing_force_distribution, y_slicing_force_distribution, surface_forces
    """

    data: str
    title: Optional[str] = None
    exclude: Optional[List[str]] = None
    operations: Optional[List[OperationTypes]] = None
    type_name: Literal["DataItem"] = pd.Field("DataItem", frozen=True)

    @pd.model_validator(mode='before')
    def validate_operations(cls, values):
        operations = values.get("operations")
        if operations is None:
            values["operations"] = []
        elif not isinstance(operations, list):
            values["operations"] = [operations]
        return values


    def _preprocess_data(self, case):
        source = data_from_path(case, self.data)

        if isinstance(source, case_results.SurfaceForcesResultCSVModel):
            full_path = self.data.split("/")
            new_variable_name = "opr_" + uuid.uuid4().hex[:8]
            variable_name = None
            if len(full_path) == 1:
                pass
            elif len(full_path) == 2:
                variable_name = full_path[-1]
                self.operations.insert(0, Expression(expr=variable_name))
            else:
                raise ValueError(f'{self.__class__.__name__}, unknown input: data={self.data}, allowed single <source> or <source>/<variable>')
            
            return source, new_variable_name

        raise NotImplementedError(f'{self.__class__.__name__} not implemented for data type: data={self.data}, {type(source)=}')


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

        source = data_from_path(case, self.data)
        if isinstance(source, case_results.SurfaceForcesResultCSVModel):
            if self.exclude is not None:
                source.filter(exclude=self.exclude)

            source, new_variable_name = self._preprocess_data(case)
            if len(self.operations) > 0:
                for opr in self.operations:
                    source, cases, result = opr.calculate(source, cases, new_variable_name)
                return result[new_variable_name]

            return source


    def __str__(self):
        if self.title is not None:
            return self.title
        return self.data.split('/')[-1]




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



here = os.path.dirname(os.path.abspath(__file__))
font_path = posixpath.join(here, 'fonts/')
font_definition = r'''
\setmainfont{TWKEverett}[
    Path = ''' + font_path + r''',
    Extension = .otf,
    UprightFont = *-Regular,
    ItalicFont = *-RegularItalic,
    BoldFont = *-Bold,
    BoldItalicFont = *-BoldItalic,
]
'''