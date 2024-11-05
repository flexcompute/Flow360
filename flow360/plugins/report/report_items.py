"""
Module containg detailed report items
"""

import os
from typing import List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
from pydantic import Field, NonNegativeInt, model_validator

# this plugin is optional, thus pylatex is not required: TODO add handling of installation of pylatex
# pylint: disable=import-error
from pylatex import (
    Command,
    Document,
    Figure,
    NewPage,
    NoEscape,
    Section,
    SubFigure,
    Subsection,
)

# pylint: disable=import-error
from pylatex.utils import bold, escape_latex

from flow360 import Case
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.outputs.outputs import SurfaceFieldNames
from flow360.plugins.report.utils import (
    Delta,
    Tabulary,
    _requirements_mapping,
    data_from_path,
    get_requirements_from_data_path,
    get_root_path,
)
from flow360.plugins.report.uvf_shutter import (
    ActionPayload,
    Scene,
    ScenesData,
    SetFieldPayload,
    SetObjectVisibilityPayload,
    SourceContext,
    TakeScreenshotPayload,
    UvfObjectTypes,
    UVFshutter,
)

here = os.path.dirname(os.path.abspath(__file__))


class ReportItem(Flow360BaseModel):
    """
    Base class for for all report items
    """

    boundaries: Union[Literal["ALL"], List[str]] = "ALL"
    _requirements: List[str] = None

    # pylint: disable=unused-argument,too-many-arguments
    def get_doc_item(
        self,
        cases: List[Case],
        doc: Document,
        section_func: Union[Section, Subsection] = Section,
        case_by_case=False,
        data_storage: str = ".",
        access_token: str = "",
    ) -> None:
        """
        returns doc item for report item
        """
        with doc.create(section_func(self.__class__.__name__)):
            doc.append(f"this is {self.__class__.__name__}")

    def get_requirements(self):
        """
        Returns requirements for this item
        """
        if self._requirements is not None:
            return self._requirements
        raise NotImplementedError(
            f"Internal error: get_requirements() not implemented for {self.__class__.__name__}"
        )


class Summary(ReportItem):
    """
    Represents a summary item in a report.

    Parameters
    ----------
    text : str
        The main content or text of the summary.
    type : Literal["Summary"], default="Summary"
        Indicates that this item is of type "Summary"; this field is immutable.
    _requirements : List[str], default=[]
        List of specific requirements associated with the summary item.
    """

    text: str
    type: Literal["Summary"] = Field("Summary", frozen=True)
    _requirements: List[str] = []

    # pylint: disable=too-many-arguments
    def get_doc_item(
        self,
        cases: List[Case],
        doc: Document,
        section_func: Union[Section, Subsection] = Section,
        case_by_case=False,
        data_storage: str = ".",
        access_token: str = "",
    ) -> None:
        """
        returns doc item for report item
        """
        section = section_func("Summary")
        doc.append(section)
        doc.append(f"{self.text}\n")
        Table(data=["name"], section_title=None, headers=["Case Name"]).get_doc_item(
            cases, doc, section_func, case_by_case
        )


class Inputs(ReportItem):
    """
    Inputs is a wrapper for a specific Table setup that details key inputs from the simulation
    """

    type: Literal["Inputs"] = Field("Inputs", frozen=True)
    _requirements: List[str] = [_requirements_mapping["params"]]

    # pylint: disable=too-many-arguments
    def get_doc_item(
        self,
        cases: List[Case],
        doc: Document,
        section_func: Union[Section, Subsection] = Section,
        case_by_case=False,
        data_storage: str = ".",
        access_token: str = "",
    ) -> None:
        """
        returns doc item for inputs
        """
        Table(
            data=[
                "params/version",
                "params/time_stepping/type_name",
                "params/outputs/0/output_format",
                "params/operating_condition/velocity_magnitude",
                "params/operating_condition/alpha",
            ],
            section_title="Inputs",
            headers=[
                "Version",
                "Time stepping",
                "Output Format",
                "Velocity",
                "Alpha",
            ],
        ).get_doc_item(cases, doc, section_func, case_by_case, data_storage)


class Table(ReportItem):
    """
    Represents a table within a report, with configurable data and headers.

    Parameters
    ----------
    data : list[Union[str, Delta]]
        A list of table data entries, which can be either strings or `Delta` objects.
    section_title : Union[str, None]
        The title of the table section.
    headers : Union[list[str], None], optional
        List of column headers for the table, default is None.
    type : Literal["Table"], default="Table"
        Specifies the type of report item as "Table"; this field is immutable.
    """

    data: list[Union[str, Delta]]
    section_title: Union[str, None]
    headers: Union[list[str], None] = None
    type: Literal["Table"] = Field("Table", frozen=True)

    @model_validator(mode="after")
    def _check_custom_heading_count(self) -> None:
        if self.headers is not None:
            if len(self.data) != len(self.headers):
                raise ValueError(
                    "Suppled `headers` must be the same length as `data`: "
                    f"{len(self.headers)} instead of {len(self.data)}"
                )
        return self

    def get_requirements(self):
        """
        Returns requirements for this item
        """
        return get_requirements_from_data_path(self.data)

    # pylint: disable=too-many-arguments
    def get_doc_item(
        self,
        cases: List[Case],
        doc: Document,
        section_func: Union[Section, Subsection] = Section,
        case_by_case=False,
        data_storage: str = ".",
        access_token: str = "",
    ) -> None:
        """
        Returns doc item for table
        """
        if self.section_title is not None:
            section = section_func(self.section_title)
            doc.append(section)

        # Getting tables to wrap is a pain - Tabulary seems the best approach
        with doc.create(
            Tabulary("|C" * (len(self.data) + 1) + "|", width=len(self.data) + 1)
        ) as table:
            table.add_hline()

            # Manage column headings
            field_titles = [bold("Case No.")]
            if self.headers is None:
                for path in self.data:
                    if isinstance(path, Delta):
                        field = path
                    else:
                        field = path.split("/")[-1]

                    field_titles.append(bold(str(field)))
            else:
                field_titles.extend([bold(heading) for heading in self.headers])

            table.append(Command("rowcolor", "gray!20"))
            table.add_row(field_titles)
            table.add_hline()

            # Build data rows
            for idx, case in enumerate(cases):
                row_list = [
                    data_from_path(case, path, cases, case_by_case=case_by_case)
                    for path in self.data
                ]
                row_list.insert(0, str(idx + 1))  # Case numbers
                table.add_row(row_list)
                table.add_hline()


class Chart(ReportItem):
    """
    Represents a chart in a report, with options for layout and display properties.

    Parameters
    ----------
    section_title : Union[str, None]
        The title of the chart section.
    fig_name : str
        Name of the figure file or identifier for the chart.
    fig_size : float, default=0.7
        Relative size of the figure as a fraction of text width.
    items_in_row : Union[int, None], optional
        Number of items to display in a row within the chart section.
    select_indices : Optional[List[NonNegativeInt]], optional
        Specific indices to select for the chart.
    single_plot : bool, default=False
        If True, display as a single plot; otherwise, use multiple plots.
    force_new_page : bool, default=False
        If True, starts the chart on a new page in the report.
    """

    section_title: Union[str, None]
    fig_name: str
    fig_size: float = 0.7  # Relates to fraction of the textwidth
    items_in_row: Union[int, None] = None
    select_indices: Optional[List[NonNegativeInt]] = None
    single_plot: bool = False
    force_new_page: bool = False

    @model_validator(mode="after")
    def _check_chart_args(self) -> None:
        if self.items_in_row is not None and self.items_in_row != -1:
            if self.items_in_row < 1:
                raise ValueError(
                    "`Items_in_row` should be greater than 1. Use -1 to include all "
                    "cases on a single row. Use `None` to disable the argument."
                )
        if self.items_in_row is not None and self.single_plot:
            raise ValueError("`Items_in_row` and `single_plot` cannot be used together.")
        return self

    def _assemble_fig_rows(self, img_list: list[str], doc: Document, fig_caption: str):
        """
        Build a figure from SubFigures which displays images in rows

        Using Doc manually here may be uncessary - but it does allow for more control
        """

        # Smaller than 1 to avoid overflowing - single subfigure sizing seems to be weird
        minipage_size = 0.95 / self.items_in_row if self.items_in_row != 1 else 0.7
        doc.append(NoEscape(r"\begin{figure}[h!]"))
        doc.append(NoEscape(r"\centering"))

        # Build list of indices to combine into rows
        indices = list(range(len(img_list)))
        idx_list = [
            indices[i : i + self.items_in_row] for i in range(0, len(indices), self.items_in_row)
        ]
        for row_idx in idx_list:
            for idx in row_idx:
                sub_fig = SubFigure(position="t", width=NoEscape(rf"{minipage_size}\textwidth"))
                sub_fig.add_image(filename=img_list[idx], width=NoEscape(r"\textwidth"))

                # Stop caption for single subfigures - happens when include_case_by_case
                if self.items_in_row != 1:
                    sub_fig.add_caption(idx)

                doc.append(sub_fig)

                doc.append(NoEscape(r"\hfill"))

            doc.append(NoEscape(r"\\"))

        doc.append(NoEscape(r"\caption{" + escape_latex(fig_caption) + "}"))
        doc.append(NoEscape(r"\end{figure}"))


class Chart2D(Chart):
    """
    Represents a 2D chart within a report, plotting x and y data.

    Parameters
    ----------
    x : Union[str, Delta]
        The data source for the x-axis, which can be a string path or a `Delta` object.
    y : Union[str, Delta]
        The data source for the y-axis, which can be a string path or a `Delta` object.
    background : Union[Literal["geometry"], None], optional
        Background type for the chart; set to "geometry" or None.
    _requirements : List[str]
        Internal list of requirements associated with the chart.
    type : Literal["Chart2D"], default="Chart2D"
        Specifies the type of report item as "Chart2D"; this field is immutable.
    """

    x: Union[str, Delta]
    y: Union[str, Delta]
    background: Union[Literal["geometry"], None] = None
    _requirements: List[str] = [_requirements_mapping["total_forces"]]
    type: Literal["Chart2D"] = Field("Chart2D", frozen=True)

    def get_requirements(self):
        """
        Returns requirements for this item
        """
        return get_requirements_from_data_path([self.x, self.y])

    def is_log_plot(self):
        """
        Determines if the plot is logarithmic based on the data path of the y-axis.

        Returns
        -------
        bool
            True if the root path of `self.y` corresponds to "nonlinear_residuals",
            indicating a logarithmic plot; False otherwise.
        """
        root_path = get_root_path(self.y)
        return root_path == "nonlinear_residuals"

    # pylint: disable=unused-argument,too-many-arguments
    def _create_fig(
        self, x_data: list, y_data: list, x_lab: str, y_lab: str, legend: str, save_name: str
    ) -> None:
        """Create a simple matplotlib figure"""
        if self.is_log_plot():
            plt.semilogy(x_data, y_data)
        else:
            plt.plot(x_data, y_data)
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        if self.single_plot:
            plt.legend([val + 1 for val in range(len(x_data))])

        plt.savefig(save_name)
        if not self.single_plot:
            plt.close()

    def _handle_title(self, doc, section_func):
        if self.section_title is not None:
            section = section_func(self.section_title)
            doc.append(section)
        return doc

    # pylint: disable=too-many-arguments,too-many-locals
    def get_doc_item(
        self,
        cases: List[Case],
        doc: Document,
        section_func: Union[Section, Subsection] = Section,
        case_by_case=False,
        data_storage: str = ".",
        access_token: str = "",
    ) -> None:
        """
        returns doc item for chart
        """

        if self.force_new_page:
            doc.append(NewPage())

        # Change items in row to be the number of cases if higher number is supplied
        if self.items_in_row is not None:
            if self.items_in_row > len(cases) or self.items_in_row == -1:
                self.items_in_row = len(cases)

        doc = self._handle_title(doc, section_func)

        x_lab = self.x.split("/")[-1]
        y_lab = self.y.split("/")[-1]

        figure_list = []
        # pylint: disable=not-an-iterable
        if case_by_case is False:
            cases = (
                [cases[i] for i in self.select_indices]
                if self.select_indices is not None
                else cases
            )
        for case in cases:

            # Extract data from the Case
            x_data = data_from_path(case, self.x, cases)
            y_data = data_from_path(case, self.y, cases)

            # Create the figure using basic matplotlib
            cbc_str = "_cbc_" if case_by_case else ""
            save_name = os.path.join(data_storage, self.fig_name + cbc_str + case.name + ".png")
            legend = case.name
            self._create_fig(x_data, y_data, x_lab, y_lab, legend, save_name)

            # Allow for handling the figures later inside a subfig
            if self.items_in_row is not None:
                figure_list.append(save_name)

            elif self.single_plot:
                continue

            else:
                # Fig is added to doc later to facilitate method of creating single_plot
                fig = Figure(position="h!")
                fig.add_image(save_name, width=NoEscape(rf"{self.fig_size}\textwidth"))
                fig.add_caption(
                    NoEscape(f"{bold(y_lab)} against {bold(x_lab)} for {bold(case.name)}.")
                )
                figure_list.append(fig)

        if self.items_in_row is not None:
            fig_caption = NoEscape(f'{bold(y_lab)} against {bold(x_lab)} for {bold("all cases")}.')
            self._assemble_fig_rows(figure_list, doc, fig_caption)

        elif self.single_plot:
            # Takes advantage of plot cached by matplotlib and that the last save_name is the full plot
            fig = Figure(position="h!")
            fig.add_image(save_name, width=NoEscape(rf"{self.fig_size}\textwidth"))
            fig.add_caption(
                NoEscape(f'{bold(y_lab)} against {bold(x_lab)} for {bold("all cases")}.')
            )
            doc.append(fig)
        else:
            for fig in figure_list:
                doc.append(fig)

        # Stops figures floating away from their sections
        doc.append(NoEscape(r"\FloatBarrier"))
        doc.append(NoEscape(r"\clearpage"))

        # Clear the matplotlib cache to be certain figure won't appear
        plt.close()


class Chart3D(Chart):
    """
    Represents a 3D chart within a report, displaying a specific surface field.

    Parameters
    ----------
    field : Optional[SurfaceFieldNames], default=None
        The name of the surface field to display in the chart.
    limits : Optional[Tuple[float, float]], default=None
        Optional limits for the field values, specified as a tuple (min, max).
    show : UvfObjectTypes
        Type of object to display in the 3D chart.
    """

    field: Optional[SurfaceFieldNames] = None
    # camera: List[float]
    limits: Optional[Tuple[float, float]] = None
    show: UvfObjectTypes

    _requirements: List[str] = [Case._manifest_path]  # pylint: disable=protected-access
    type: Literal["Chart3D"] = Field("Chart3D", frozen=True)

    def _get_uvf_qcriterion_script(
        self, script: List = None, field: str = None, limits: Tuple[float, float] = None
    ):
        if script is None:
            script = []

        script += [
            ActionPayload(
                action="set-object-visibility",
                payload=SetObjectVisibilityPayload(object_ids=["slices"], visibility=False),
            ),
            ActionPayload(action="focus"),
            ActionPayload(
                action="set-field",
                payload=SetFieldPayload(object_id="qcriterion", field_name=field, min_max=limits),
            ),
        ]
        return script

    def _get_uvf_screenshot_script(self, script, screenshot_name):
        script += [
            ActionPayload(
                action="take-screenshot",
                payload=TakeScreenshotPayload(file_name=screenshot_name, type="png"),
            )
        ]

        return script

    def _get_uvf_boundary_script(
        self, script: List = None, field: str = None, limits: Tuple[float, float] = None
    ):
        if script is None:
            script = []
        script += [
            ActionPayload(
                action="set-object-visibility",
                payload=SetObjectVisibilityPayload(
                    object_ids=["slices", "qcriterion"], visibility=False
                ),
            ),
            ActionPayload(action="focus"),
        ]
        if field is not None:
            script += [
                ActionPayload(
                    action="set-field",
                    payload=SetFieldPayload(
                        object_id="boundaries", field_name="yPlus", min_max=limits
                    ),
                )
            ]
        return script

    def _get_uvf_request(self, fig_name, user_id, case_id):

        if self.show == "qcriterion":
            script = self._get_uvf_qcriterion_script(field=self.field, limits=self.limits)
        elif self.show == "boundaries":
            script = self._get_uvf_boundary_script(field=self.field, limits=self.limits)
        elif self.show == "slices":
            raise NotImplementedError("Slices not implemented yet")
        else:
            raise ValueError(f'"{self.show}" is not corect type for 3D chart.')

        script = self._get_uvf_screenshot_script(script=script, screenshot_name=fig_name)

        scene = Scene(name="my-scene", script=script)
        source_context = SourceContext(user_id=user_id, case_id=case_id)
        scenes_data = ScenesData(scenes=[scene], context=source_context)
        return scenes_data

    def _get_images(self, cases: List[Case], data_storage, access_token: str):
        fig_name = self.fig_name
        uvf_requests = []
        for case in cases:
            uvf_requests.append(self._get_uvf_request(fig_name, case.info.user_id, case.id))
        img_files = UVFshutter(
            cases=cases, data_storage=data_storage, access_token=access_token
        ).get_images(fig_name, uvf_requests)
        # taking "first" image from returned images as UVF-shutter supports many screenshots generation on one call
        img_list = [img_files[case.id][0] for case in cases]
        return img_list

    # pylint: disable=too-many-arguments
    def get_doc_item(
        self,
        cases: List[Case],
        doc: Document,
        section_func: Union[Section, Subsection] = Section,
        case_by_case: bool = False,
        data_storage: str = ".",
        access_token: str = "",
    ):
        # Create new page is user requests one
        if self.force_new_page:
            doc.append(NewPage())

        # Change items in row to be the number of cases if higher number is supplied
        if self.items_in_row is not None:
            if self.items_in_row > len(cases) or self.items_in_row == -1:
                self.items_in_row = len(cases)

        # Only create a title if specified
        if self.section_title is not None:
            section = section_func(self.section_title)
            doc.append(section)

        # Reduce the case list by the selected IDs
        # pylint: disable=not-an-iterable
        cases = (
            [cases[i] for i in self.select_indices] if self.select_indices is not None else cases
        )

        img_list = self._get_images(cases, data_storage, access_token)

        if self.items_in_row is not None:
            fig_caption = "Chart3D Row"
            self._assemble_fig_rows(img_list, doc, fig_caption)

        else:
            for filename in img_list:
                fig = Figure(position="h!")
                fig.add_image(filename, width=NoEscape(rf"{self.fig_size}\textwidth"))
                fig.add_caption("A Chart3D test picture.")
                doc.append(fig)

        # Stops figures floating away from their sections
        doc.append(NoEscape(r"\FloatBarrier"))
        doc.append(NoEscape(r"\clearpage"))
