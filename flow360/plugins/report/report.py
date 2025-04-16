"""
Report generation interface
"""

import os
from typing import Callable, List, Optional, Set, Union

import pydantic as pd

# this plugin is optional, thus pylatex is not required: TODO add handling of installation of pylatex
# pylint: disable=import-error
from pylatex import Section, Subsection

from flow360 import Case
from flow360.cloud.flow360_requests import NewReportRequest
from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import ReportInterface
from flow360.component.resource_base import AssetMetaBaseModel, Flow360Resource
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.utils import validate_type
from flow360.plugins.report.report_context import ReportContext
from flow360.plugins.report.report_doc import ReportDoc
from flow360.plugins.report.report_items import (
    Chart,
    Chart2D,
    Chart3D,
    FileNameStr,
    Inputs,
    NonlinearResiduals,
    Settings,
    Summary,
    Table,
)
from flow360.plugins.report.utils import (
    RequirementItem,
    get_requirements_from_data_path,
)
from flow360.plugins.report.uvf_shutter import ShutterBatchService


class Report(Flow360Resource):
    """
    Report component for interacting with cloud
    """

    # pylint: disable=redefined-builtin
    def __init__(self, id: str):
        super().__init__(
            interface=ReportInterface,
            meta_class=AssetMetaBaseModel,
            id=id,
        )

    @classmethod
    def _from_meta(cls, meta: AssetMetaBaseModel):
        validate_type(meta, "meta", AssetMetaBaseModel)
        report = cls(id=meta.id)
        report._set_meta(meta)
        return report

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
            file_name="results/" + file_name,
            to_file=to_file,
            to_folder=to_folder,
            overwrite=overwrite,
        )


# pylint: disable=too-few-public-methods
class ReportDraft:
    """
    ReportDraft interface
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
        return Report(resp["id"])


class ReportTemplate(Flow360BaseModel):
    """
    A model representing a report containing various components such as summaries, inputs, tables,
    and charts in both 2D and 3D.

    Parameters
    ----------
    title: str, optional
        Title of report, shown on the first page
    items : List[Union[Summary, Inputs, Table, Chart2D, Chart3D]]
        A list of report items, each of which can be a summary, input data, table, 2D chart, or 3D chart.
        The `type` field acts as a discriminator for differentiating item types.
    include_case_by_case : bool, default=True
        Flag indicating whether to include a case-by-case analysis in the report.
    """

    title: Optional[str] = None
    items: List[Union[Summary, Inputs, Table, NonlinearResiduals, Chart2D, Chart3D]] = pd.Field(
        discriminator="type_name"
    )
    include_case_by_case: bool = False
    settings: Optional[Settings] = Settings()

    @pd.model_validator(mode="after")
    def check_fig_names(self):
        """Validate and assign unique fig_names to report items."""
        used_fig_names: Set[str] = set()
        for idx, item in enumerate(self.items):
            if hasattr(item, "fig_name"):
                fig_name = getattr(item, "fig_name", None)
                if not fig_name:
                    class_name = item.__class__.__name__
                    fig_name = f"{class_name}_{idx}"
                    item.fig_name = fig_name
                else:
                    fig_name = item.fig_name
                if fig_name in used_fig_names:
                    raise ValueError(
                        f"Duplicate fig_name '{fig_name}' found in item at index {idx}"
                    )
                used_fig_names.add(fig_name)
        # return model

    # pylint: disable=protected-access
    def _generate_shutter_screenshots(self, context: ReportContext):
        service = ShutterBatchService()

        for chart in self.items:  # pylint: disable=not-an-iterable
            if isinstance(chart, Chart3D):
                for case in context.cases:
                    if not chart._fig_exist(case.id, context.data_storage):
                        req = chart._get_shutter_request(case)
                        service.add_request(req)
            elif isinstance(chart, Chart2D):
                chart3d, reference_case = chart.get_background_chart3d(context.cases)
                if chart3d is not None:
                    if not chart3d._fig_exist(reference_case.id, context.data_storage):
                        req = chart3d._get_shutter_request(reference_case)
                        if req is not None:
                            service.add_request(req)

        service.process_requests(context)

    def _get_baseline_requirements(self):
        return get_requirements_from_data_path(["volume_mesh", "surface_mesh", "geometry"])

    def get_requirements(self) -> List[RequirementItem]:
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
        for req in self._get_baseline_requirements():
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

        return ReportDraft.submit(
            name=name,
            case_ids=[case.id for case in cases],
            config=self.model_dump_json(),
            solver_version=solver_version,
        )

    # pylint: disable=too-many-arguments
    @pd.validate_call(config={"arbitrary_types_allowed": True})
    def create_pdf(
        self,
        filename: FileNameStr,
        cases: list[Case],
        landscape: bool = True,
        data_storage: str = ".",
        shutter_url: str = None,
        shutter_access_token: str = None,
        shutter_screenshot_process_function: Callable = None,
        process_screenshot_in_parallel: bool = True,
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

        Returns
        -------
        None
        """
        os.makedirs(data_storage, exist_ok=True)
        report_doc = ReportDoc(title=self.title, landscape=landscape)

        context = ReportContext(
            cases=cases,
            doc=report_doc.doc,
            data_storage=data_storage,
            process_screenshot_in_parallel=process_screenshot_in_parallel,
            shutter_url=shutter_url,
            shutter_access_token=shutter_access_token,
            shutter_screenshot_process_function=shutter_screenshot_process_function,
        )
        case_context = context.model_copy(
            update={
                "section_func": Subsection,
                "case_by_case": True,
            }
        )

        self._generate_shutter_screenshots(context)

        for item in self.items:  # pylint: disable=not-an-iterable
            item.get_doc_item(context, self.settings)

        if self.include_case_by_case is True:
            with report_doc.doc.create(Section("Appendix", numbering=False)):
                for case in cases:
                    with report_doc.doc.create(Section(f"Case: {case.id}")):
                        case_context = case_context.model_copy(update={"cases": [case]})
                        for item in self.items:  # pylint: disable=not-an-iterable
                            if isinstance(item, Chart) and item.select_indices is not None:
                                selected_case_ids = [cases[i].id for i in item.select_indices]
                                if case.id not in selected_case_ids:
                                    continue

                            item.get_doc_item(case_context, self.settings)

        report_doc.generate_pdf(os.path.join(data_storage, filename))
