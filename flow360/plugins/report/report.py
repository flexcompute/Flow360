import os
from typing import List, Union

from pydantic import Field
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

from flow360 import Case
from flow360.cloud.requests import NewReportRequest
from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import ReportInterface
from flow360.component.simulation.framework.base_model import Flow360BaseModel

from .report_items import Chart, Chart2D, Chart3D, Inputs, Summary, Table


class ReportApi:
    _webapi = RestApi(ReportInterface.endpoint)

    @classmethod
    def submit(
        cls,
        name: str,
        case_ids: List[str],
        config: str,
        landscape: bool = True,
        solver_version: str = None,
    ):
        request = NewReportRequest(
            name=name,
            resources=[{"type": "Case", "id": id} for id in case_ids],
            config_json=config,
            solver_version=solver_version,
        )
        return cls._webapi.post(json=request.dict())


class Report(Flow360BaseModel):
    items: List[Union[Summary, Inputs, Table, Chart2D, Chart3D]] = Field(discriminator="type")
    include_case_by_case: bool = True

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
        requirements = set()
        for item in self.items:
            [requirements.add(req) for req in item.get_requirements()]
        return list(requirements)

    def create_in_cloud(
        self, name: str, cases: list[Case], landscape: bool = False, solver_version: str = None
    ):
        return ReportApi.submit(
            name=name,
            case_ids=[case.id for case in cases],
            config=self.model_dump_json(),
            solver_version=solver_version,
        )

    def create_pdf(
        self, filename: str, cases: list[Case], landscape: bool = False, data_storage: str = "."
    ) -> None:
        # Create a new LaTeX document
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
                r"\captionsetup{position=bottom, font=large, labelformat=graybox, labelsep=none, justification=raggedright, singlelinecheck=false}"
            )
        )
        doc.change_document_style("header")

        # Iterate through all cases together
        for item in self.items:
            item.get_doc_item(cases, doc, case_by_case=False, data_storage=data_storage)

        # Iterate each case one at a time
        if self.include_case_by_case is True:
            with doc.create(Section("Appendix", numbering=False)):
                for case in cases:
                    with doc.create(Section(f"Case: {case.id}")):
                        for item in self.items:
                            # Don't attempt to create ReportItems that have a select_case_ids which don't include the current case.id
                            # Checks for valid selecte_case_ids can be done later
                            if isinstance(item, Chart) and item.select_indices is not None:
                                selected_case_ids = [cases[i].id for i in item.select_indices]
                                if case.id not in selected_case_ids:
                                    continue

                            item.get_doc_item(
                                [case],
                                doc,
                                Subsection,
                                self.include_case_by_case,
                                data_storage=data_storage,
                            )

        # Generate the PDF
        doc.generate_pdf(os.path.join(data_storage, filename), clean_tex=False)
