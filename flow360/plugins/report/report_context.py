""" 
Module for ReportContext to hold shared configurations between Report and ReportItem
"""

import pydantic as pd
from typing import List, Union, Optional, Type

# pylint: disable=import-error
from pylatex import (
    Document,
    Section,
    Subsection,
)

from flow360 import Case


class ReportContext(pd.BaseModel):
    """
    Context for report data to be used in conjunction with report item.
    """

    cases: List[Case]
    doc: Document = Document()
    section_func: Union[Type[Section], Type[Subsection]] = Section
    case_by_case: bool = False
    data_storage: str = "."
    shutter_url: Optional[str] = None
    shutter_access_token: Optional[str] = None

    model_config = pd.ConfigDict(arbitrary_types_allowed=True)
