"""
Module for ReportContext to hold shared configurations between Report and ReportItem
"""

from typing import Callable, List, Optional, Type, Union

import pydantic as pd

# pylint: disable=import-error
from pylatex import Document, Section, Subsection

from flow360.component.case import Case


class ReportContext(pd.BaseModel):
    """
    Context for report data to be used in conjunction with report item.
    """

    cases: List[Case]
    doc: Document = Document()
    section_func: Union[Type[Section], Type[Subsection]] = Section
    case_by_case: bool = False
    data_storage: str = "."
    process_screenshot_in_parallel: bool = True

    shutter_url: Optional[str] = None
    shutter_access_token: Optional[str] = None
    shutter_screenshot_process_function: Optional[Callable] = None

    model_config = pd.ConfigDict(arbitrary_types_allowed=True)
