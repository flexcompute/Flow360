""" Module for ReportContext to hold shared configurations between Report and ReportItem
"""

from typing import List, Union

# pylint: disable=import-error
from pylatex import (
    Document,
    Section,
    Subsection,
)

from flow360 import Case


class ReportContext:
    """
    Context for report data to be used in conjunction with report item
    """

    def __init__(
        self,
        cases: List[Case],
        doc: Document,
        section_func: Union[Section, Subsection] = Section,
        case_by_case: bool = False,
        data_storage: str = ".",
        access_token: str = "",
    ):  # pylint: disable=R0913
        self.cases = cases
        self.doc = doc
        self.section_func = section_func
        self.case_by_case = case_by_case
        self.data_storage = data_storage
        self.access_token = access_token
