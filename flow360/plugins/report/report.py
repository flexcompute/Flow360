"""
Report generation interface
"""

import os
from typing import List, Union, Set

from flow360 import Case
from flow360.component.resource_base import Flow360Resource, AssetMetaBaseModel
from flow360.cloud.requests import NewReportRequest
from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import ReportInterface
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.plugins.report.report_context import ReportContext
from flow360.plugins.report.report_items import Chart, Chart2D, Chart3D, Inputs, Summary, Table, FileNameStr
from pydantic import Field, validate_call, model_validator

# this plugin is optional, thus pylatex is not required: TODO add handling of installation of pylatex
# pylint: disable=import-error
from pylatex import (
    Command,
    Document,
    Foot,
    Head,
    NewPage,
    NoEscape,
    Package,
    PageStyle,
    Section,
    Subsection,
)


class Report(Flow360Resource):
    # pylint: disable=redefined-builtin
    def __init__(self, id: str):
        super().__init__(
            interface=ReportInterface,
            meta_class=AssetMetaBaseModel,
            id=id,
        )

    def download(self, file_name, to_file=None, to_folder=".", overwrite: bool = True):
        """
        Download the file to the specified location.

        Parameters
        ----------
        file_name : str
            File name to download.
        to_file : str, optional
            The name of the file after downloading.
        to_folder : str, optional
            The folder where the file will be downloaded.
        overwrite : bool, optional
            Flag indicating whether to overwrite existing files.
        """
        # pylint: disable=no-member
        return self._download_file(
            file_name='results/' + file_name, to_file=to_file, to_folder=to_folder, overwrite=overwrite
        )


# pylint: disable=too-few-public-methods
class ReportApi:
    """
    ReportApi interface
    """

    _webapi = RestApi(ReportInterface.endpoint)

    # pylint: disable=too-many-arguments
    @classmethod
    def submit(
        cls,
        name: str,
        case_ids: List[str],
        config: str,
        landscape: bool = True,  # pylint: disable=unused-argument
        solver_version: str = None,
    ):
        """
        Submits a new report request.

        Parameters
        ----------
        name : str
            The name of the report.
        case_ids : List[str]
            List of case IDs to include in the report.
        config : str
            JSON configuration for report settings.
        landscape : bool, default=True
            Whether the report should be landscape-oriented (unused argument).
        solver_version : str, optional
            Version of the solver for report generation.

        Returns
        -------
        Response
            The response object from the web API post request.
        """

        request = NewReportRequest(
            name=name,
            resources=[{"type": "Case", "id": id} for id in case_ids],
            config_json=config,
            solver_version=solver_version,
        )
        resp = cls._webapi.post(json=request.dict())
        return Report(resp['id'])


class ReportTemplate(Flow360BaseModel):
    """
    A model representing a report containing various components such as summaries, inputs, tables,
    and charts in both 2D and 3D.

    Parameters
    ----------
    items : List[Union[Summary, Inputs, Table, Chart2D, Chart3D]]
        A list of report items, each of which can be a summary, input data, table, 2D chart, or 3D chart.
        The `type` field acts as a discriminator for differentiating item types.
    include_case_by_case : bool, default=True
        Flag indicating whether to include a case-by-case analysis in the report.
    """

    items: List[Union[Summary, Inputs, Table, Chart2D, Chart3D]] = Field(discriminator="type")
    include_case_by_case: bool = False


    @model_validator(mode='after')
    def check_fig_names(cls, model):
        """Validate and assign unique fig_names to report items."""
        used_fig_names: Set[str] = set()
        for idx, item in enumerate(model.items):
            if hasattr(item, 'fig_name'):
                fig_name = getattr(item, 'fig_name', None)
                if not fig_name:
                    class_name = item.__class__.__name__
                    fig_name = f"{class_name}_{idx}"
                    item.fig_name = fig_name 
                else:
                    fig_name = item.fig_name
                if fig_name in used_fig_names:
                    raise ValueError(f"Duplicate fig_name '{fig_name}' found in item at index {idx}")
                used_fig_names.add(fig_name)
        return model


    def _create_header_footer(self) -> PageStyle:
        header = PageStyle("header")

        with header.create(Head("R")):
            header.append(
                NoEscape(
                    r"\includegraphics[width=2cm]{"
                    f"{os.path.join(os.path.dirname(__file__), 'img', 'flow360.png')}"
                    "}"
                )
            )

        # Footer date and page number
        with header.create(Foot("R")):
            header.append(NoEscape(r"\thepage"))

        with header.create(Foot("L")):
            header.append(NoEscape(r"\today"))

        return header

    def _define_preamble(self, doc, landscape):
        # Package info
        doc.packages.append(Package("float"))
        doc.packages.append(Package("caption"))
        doc.packages.append(
            Package("graphicx")
        )  # Included here as it's sometimes not included automatically when needed
        doc.packages.append(Package("placeins"))  # For FloatBarrier
        doc.packages.append(Package("xcolor", options="table"))  # For coloring inc Table
        # doc.packages.append(Package("opensans", options="default"))  # For changing font

        geometry_options = ["a4paper", "margin=1in"]
        if landscape:
            geometry_options.append("landscape")
        doc.packages.append(Package("geometry", options=geometry_options))

    def _make_title(self, doc):

        # Title page
        # NOTE: using NewLine inside a titlepage causes centering to fail. MUST use "\\" instead.
        doc.append(Command("begin", "titlepage"))
        doc.append(Command("centering"))

        # Title
        doc.append(Command("vspace*{3cm}"))
        doc.append(Command("huge", "Flow 360 Report"))
        doc.append(NoEscape(r"\\"))

        # Date
        doc.append(Command("large", NoEscape(r"\today")))  # Current date
        doc.append(NoEscape(r"\\"))

        # Image
        doc.append(
            NoEscape(
                r"\includegraphics[width=0.4\textwidth]{"
                + os.path.join(os.path.dirname(__file__), "img", "flow360.png")
                + "}"
            )
        )
        doc.append(Command("end", "titlepage"))

    def get_requirements(self):
        """
        Collects and returns unique requirements from all items.

        This method iterates over all items, retrieves each item's requirements,
        and aggregates them into a unique list.

        Returns
        -------
        list
            A list of unique requirements aggregated from all items.
        """

        requirements = set()
        for item in self.items:  # pylint: disable=not-an-iterable
            for req in item.get_requirements():
                requirements.add(req)
        return list(requirements)

    # pylint: disable=unused-argument
    def create_in_cloud(
        self, name: str, cases: list[Case], landscape: bool = False, solver_version: str = None
    ):
        """
        Creates a report in the cloud for a specified set of cases.

        Parameters
        ----------
        name : str
            The name of the report to create.
        cases : list[Case]
            A list of `Case` instances to include in the report.
        landscape : bool, default=False
            Orientation of the report, where `True` represents landscape.
        solver_version : str, optional
            Version of the solver for report generation.

        Returns
        -------
        Response
            The response from the Report API submission.
        """

        return ReportApi.submit(
            name=name,
            case_ids=[case.id for case in cases],
            config=self.model_dump_json(),
            solver_version=solver_version,
        )

    @validate_call(config={"arbitrary_types_allowed": True})
    def create_pdf(
        self,
        filename: FileNameStr,
        cases: list[Case],
        landscape: bool = False,
        data_storage: str = ".",
        use_cache: bool = True,
        shutter_url: str = None,
        shutter_access_token: str = None
    ) -> None:
        """
        Generates a PDF report for a specified set of cases.

        Parameters
        ----------
        filename : str
            The name of the output PDF file.
        cases : list[Case]
            A list of `Case` instances to include in the PDF report.
        landscape : bool, default=False
            Orientation of the report, where `True` represents landscape.
        data_storage : str, default="."
            Directory where the PDF file will be saved.
        use_cache : bool
            Whether to force generate data or use cached data

        Returns
        -------
        None
        """
        os.makedirs(data_storage, exist_ok=True)
        doc = Document(document_options=["10pt"])
        self._define_preamble(doc, landscape)
        self._make_title(doc)

        doc.append(NewPage())

        # Preamble info
        doc.preamble.append(self._create_header_footer())
        doc.preamble.append(NoEscape(r"\setlength{\headheight}{20pt}"))
        doc.preamble.append(NoEscape(r"\addtolength{\topmargin}{-5pt}"))

        doc.preamble.append(
            NoEscape(r"\DeclareCaptionLabelFormat{graybox}{\colorbox{gray!20}{#1 #2} }")
        )
        doc.preamble.append(
            NoEscape(
                r"\captionsetup{position=bottom, font=large, labelformat=graybox, "
                r"labelsep=none, justification=raggedright, singlelinecheck=false}"
            )
        )

        doc.change_document_style("header")

        context = ReportContext(
            cases=cases, doc=doc, data_storage=data_storage, shutter_url=shutter_url, shutter_access_token=shutter_access_token
        )
        # Iterate through all cases together
        for item in self.items:  # pylint: disable=not-an-iterable
            item.get_doc_item(context)

        # Iterate each case one at a time
        if self.include_case_by_case is True:
            with doc.create(Section("Appendix", numbering=False)):
                for case in cases:
                    with doc.create(Section(f"Case: {case.id}")):
                        case_context = ReportContext(
                            cases=[case],
                            doc=doc,
                            section_func=Subsection,
                            case_by_case=True,
                            data_storage=data_storage,
                            shutter_url=shutter_url,
                            shutter_access_token=shutter_access_token,
                        )
                        for item in self.items:  # pylint: disable=not-an-iterable
                            # Don't attempt to create ReportItems that have a
                            # select_case_ids which don't include the current case.id
                            # Checks for valid selecte_case_ids can be done later
                            if isinstance(item, Chart) and item.select_indices is not None:
                                selected_case_ids = [cases[i].id for i in item.select_indices]
                                if case.id not in selected_case_ids:
                                    continue

                            item.get_doc_item(case_context)

        # Generate the PDF
        doc.generate_pdf(os.path.join(data_storage, filename), clean_tex=False)
