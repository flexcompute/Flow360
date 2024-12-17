"""
Module containg detailed report items
"""

from __future__ import annotations

import os
from typing import Annotated, List, Literal, Optional, Tuple, Union

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pydantic as pd
import unyt
from pydantic import (
    BaseModel,
    Field,
    NonNegativeInt,
    StringConstraints,
    field_validator,
    model_validator,
)

# this plugin is optional, thus pylatex is not required: TODO add handling of installation of pylatex
# pylint: disable=import-error
from pylatex import Command, Document, Figure, NewPage, NoEscape, SubFigure

# pylint: disable=import-error
from pylatex.utils import bold, escape_latex

from flow360 import Case, SimulationParams
from flow360.component.results import case_results
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.outputs.output_fields import (
    SurfaceFieldNames,
    IsoSurfaceFieldNames,
)
from flow360.component.simulation.unit_system import (
    DimensionedTypes,
    unyt_quantity,
    is_flow360_unit,
)
from flow360.log import log
from flow360.plugins.report.report_context import ReportContext
from flow360.plugins.report.utils import (
    DataItem,
    Delta,
    OperationTypes,
    Tabulary,
    _requirements_mapping,
    data_from_path,
    generate_colorbar_from_image,
    get_requirements_from_data_path,
    get_root_path,
    split_path,
)
from flow360.plugins.report.uvf_shutter import (
    ActionPayload,
    Camera,
    FocusPayload,
    ResetFieldPayload,
    Resource,
    Scene,
    ScenesData,
    SetCameraPayload,
    SetFieldPayload,
    SetLICPayload,
    SetObjectVisibilityPayload,
    Shutter,
    ShutterObjectTypes,
    TakeScreenshotPayload,
)

here = os.path.dirname(os.path.abspath(__file__))


FileNameStr = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9._-]+$")]

FIG_ASPECT_RATIO = 16 / 9


class ReportItem(Flow360BaseModel):
    """
    Base class for for all report items
    """

    boundaries: Union[Literal["ALL"], List[str]] = "ALL"
    _requirements: List[str] = None

    # pylint: disable=unused-argument,too-many-arguments
    def get_doc_item(
        self,
        context: ReportContext,
    ) -> None:
        """
        returns doc item for report item
        """
        with context.doc.create(context.section_func(self.__class__.__name__)):
            context.doc.append(f"this is {self.__class__.__name__}")

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
    text : str, optional
        The main content or text of the summary.
    type_name : Literal["Summary"], default="Summary"
        Indicates that this item is of type "Summary"; this field is immutable.
    _requirements : List[str], default=[]
        List of specific requirements associated with the summary item.
    """

    text: Optional[str] = None
    type_name: Literal["Summary"] = Field("Summary", frozen=True)
    _requirements: List[str] = []

    # pylint: disable=too-many-arguments
    def get_doc_item(
        self,
        context: ReportContext,
    ) -> None:
        """
        returns doc item for report item
        """
        section = context.section_func("Summary")
        context.doc.append(section)
        if self.text is not None:
            context.doc.append(f"{self.text}\n")
        Table(data=["name", "id"], section_title=None, headers=["Case Name", "Id"]).get_doc_item(
            context
        )


class Inputs(ReportItem):
    """
    Inputs is a wrapper for a specific Table setup that details key inputs from the simulation
    """

    type_name: Literal["Inputs"] = Field("Inputs", frozen=True)
    _requirements: List[str] = [_requirements_mapping["params"]]

    # pylint: disable=too-many-arguments
    def get_doc_item(
        self,
        context: ReportContext,
    ) -> None:
        """
        returns doc item for inputs
        """
        Table(
            data=[
                "params/version",
                "params/operating_condition/velocity_magnitude",
                "params/time_stepping/type_name",
            ],
            section_title="Inputs",
            headers=[
                "Version",
                "Velocity",
                "Time stepping",
            ],
        ).get_doc_item(context)


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
    type_name : Literal["Table"], default="Table"
        Specifies the type of report item as "Table"; this field is immutable.
    """

    data: list[Union[str, Delta, DataItem]]
    section_title: Union[str, None]
    headers: Union[list[str], None] = None
    type_name: Literal["Table"] = Field("Table", frozen=True)

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
        context: ReportContext,
    ) -> None:
        """
        Returns doc item for table
        """
        if self.section_title is not None:
            section = context.section_func(self.section_title)
            context.doc.append(section)

        # Getting tables to wrap is a pain - Tabulary seems the best approach
        with context.doc.create(
            Tabulary("|C" * (len(self.data) + 1) + "|", width=len(self.data) + 1)
        ) as table:
            table.add_hline()

            # Manage column headings
            field_titles = ["Case No."]
            if self.headers is None:
                for path in self.data:
                    if isinstance(path, (Delta, DataItem)):
                        field = path
                    else:
                        field = split_path(path)[-1]

                    field_titles.append(str(field))
            else:
                field_titles.extend([heading for heading in self.headers])

            table.append(Command("rowcolor", "labelgray"))
            table.add_row(field_titles)
            table.add_hline()

            # Build data rows
            for idx, case in enumerate(context.cases):
                row_list = [
                    data_from_path(case, path, context.cases, case_by_case=context.case_by_case)
                    for path in self.data
                ]
                row_list = [f"{x:.5g}" if isinstance(x, (int, float)) else x for x in row_list]

                row_list.insert(0, str(idx + 1))  # Case numbers
                table.add_row(row_list)
                table.add_hline()


class Chart(ReportItem):
    """
    Represents a chart in a report, with options for layout and display properties.

    Parameters
    ----------
    section_title : str, optional
        The title of the chart section.
    fig_name : FileNameStr, optional
        Name of the figure file or identifier for the chart (). Only '^[a-zA-Z0-9._-]+$' allowed.
    fig_size : float, default=0.7
        Relative size of the figure as a fraction of text width.
    items_in_row : Union[int, None], optional
        Number of items to display in a row within the chart section.
    select_indices : Optional[List[NonNegativeInt]], optional
        Specific indices to select for the chart.
    separate_plots : bool, default=None
        If True, display as multiple plots; otherwise single plot.
    force_new_page : bool, default=False
        If True, starts the chart on a new page in the report.
    """

    section_title: Optional[str] = None
    fig_name: Optional[FileNameStr] = None
    fig_size: float = 0.7  # Relates to fraction of the textwidth
    items_in_row: Union[int, None] = None
    select_indices: Optional[List[NonNegativeInt]] = None
    separate_plots: Optional[bool] = None
    force_new_page: bool = False
    caption: Optional[str] = ""

    @model_validator(mode="after")
    def _check_chart_args(self) -> None:
        if self.items_in_row is not None and self.items_in_row != -1:
            if self.items_in_row < 1:
                raise ValueError(
                    "`Items_in_row` should be greater than 1. Use -1 to include all "
                    "cases on a single row. Use `None` to disable the argument."
                )
        if self.items_in_row is not None:
            if self.separate_plots is False:
                raise ValueError(
                    "`Items_in_row` and `separate_plots=False` cannot be used together."
                )
            elif self.separate_plots is None:
                self.separate_plots = True
        return self

    def _handle_title(self, doc, section_func):
        if self.section_title is not None:
            section = section_func(self.section_title)
            doc.append(section)

    def _handle_new_page(self, doc):
        if self.force_new_page:
            doc.append(NewPage())

    def _handle_grid_input(self, cases):
        # Change items in row to be the number of cases if higher number is supplied
        if self.items_in_row is not None:
            if self.items_in_row > len(cases) or self.items_in_row == -1:
                self.items_in_row = len(cases)

    def _filter_input_cases(self, cases, case_by_case):
        # Reduce the case list by the selected IDs
        # pylint: disable=not-an-iterable
        if case_by_case is False:
            cases = (
                [cases[i] for i in self.select_indices]
                if self.select_indices is not None
                else cases
            )

        return cases

    def _fig_exist(self, id, data_storage="."):
        img_folder = os.path.join(data_storage, id)
        img_name = self.fig_name + ".png"
        img_full_path = os.path.join(img_folder, img_name)
        if os.path.exists(img_full_path):
            log.debug(f"File: {img_name=} exists in cache, reusing.")
            return True
        return False

    def _add_figure(self, doc: Document, file_name, caption, legend_filename=None):
        if legend_filename is None:
            fig = Figure(position="!ht")
            fig.add_image(file_name, width=NoEscape(rf"{self.fig_size}\textwidth"))
            fig.add_caption(caption)
            doc.append(fig)
        else:
            doc.append(NoEscape(r"\begin{figure}[!ht]"))

            sub_fig = SubFigure(position="t", width=NoEscape(r"\textwidth"))
            sub_fig.add_image(
                file_name,
                width=NoEscape(rf"{self.fig_size}\textwidth"),
                placement=NoEscape(r"\centering"),
            )
            doc.append(sub_fig)
            doc.append(NoEscape(r"\\"))
            fig = self._add_legend(legend_filename=legend_filename)
            doc.append(fig)

            doc.append(NoEscape(r"\caption{" + escape_latex(caption) + "}"))
            doc.append(NoEscape(r"\end{figure}"))

    def _add_legend(self, legend_filename: str, minipage_width=1, legend_width=0.45):
        sub_fig = SubFigure(position="t", width=NoEscape(rf"{minipage_width}\textwidth"))
        sub_fig.add_image(
            filename=legend_filename,
            width=NoEscape(rf"{legend_width}\textwidth"),
            placement=NoEscape(r"\centering"),
        )
        return sub_fig

    def _add_row_figure(
        self,
        doc: Document,
        img_list: list[str],
        fig_caption: str,
        sub_fig_captions: List[str] = None,
        legend_filename=None,
    ):
        """
        Build a figure from SubFigures which displays images in rows

        Using Doc manually here may be uncessary - but it does allow for more control
        """

        # Smaller than 1 to avoid overflowing - single subfigure sizing seems to be weird
        minipage_size = 0.86 / self.items_in_row if self.items_in_row != 1 else 0.8
        doc.append(NoEscape(r"\begin{figure}[h!]"))

        if sub_fig_captions is None:
            sub_fig_captions = range(1, len(img_list) + 1)

        figures = []
        for img, caption in zip(img_list, sub_fig_captions):
            sub_fig = SubFigure(position="t", width=NoEscape(rf"{minipage_size}\textwidth"))
            sub_fig.add_image(
                filename=img, width=NoEscape(r"\textwidth"), placement=NoEscape(r"\centering")
            )

            # Stop caption for single subfigures - happens when include_case_by_case
            if self.items_in_row != 1 and caption is not None:
                sub_fig.add_caption(caption)
            figures.append(sub_fig)

        if legend_filename is not None:
            img_list.append(legend_filename)
            legend_width = 0.9 if self.items_in_row != 1 else 0.45
            sub_fig = self._add_legend(
                legend_filename=legend_filename,
                minipage_width=minipage_size,
                legend_width=legend_width,
            )
            figures.append(sub_fig)

        # Build list of indices to combine into rows
        indices = list(range(len(img_list)))
        idx_list = [
            indices[i : i + self.items_in_row] for i in range(0, len(indices), self.items_in_row)
        ]

        for row_idx in idx_list:
            for idx in row_idx:
                doc.append(figures[idx])
                doc.append(NoEscape(r"\hfill"))
            doc.append(NoEscape(r"\\"))
        doc.append(NoEscape(r"\caption{" + escape_latex(fig_caption) + "}"))
        doc.append(NoEscape(r"\end{figure}"))


class PlotModel(BaseModel):
    x_data: Union[List[float], List[List[float]]]
    y_data: Union[List[float], List[List[float]]]
    x_label: str
    y_label: str
    legend: Optional[List[str]] = None
    is_log: bool = False
    style: str = "-"
    backgroung_png: Optional[str] = None
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None

    @field_validator("x_data", "y_data", mode="before")
    def ensure_y_data_is_list_of_lists(cls, v):
        if isinstance(v, list):
            if all(isinstance(item, list) for item in v):
                return v
            else:
                return [v]
        else:
            raise ValueError("x_data/y_data must be a list")

    @model_validator(mode="after")
    def check_lengths(self):
        if len(self.x_data) == 1:
            self.x_data = [self.x_data[0] for _ in self.y_data]
        if len(self.x_data) != len(self.y_data):
            raise ValueError(
                f"Number of x_data series but be one or equal to number of y_data series."
            )
        for idx, (x_series, y_series) in enumerate(zip(self.x_data, self.y_data)):
            if len(x_series) != len(y_series):
                raise ValueError(
                    f"Length of y_data series at index {idx} ({len(y_series)}) does not match length of x_data ({len(x_series)})"
                )
        if self.legend is not None:
            if len(self.legend) != len(self.y_data):
                raise ValueError(
                    f"Length of legend ({len(self.legend)}) must match number of y_data series ({len(self.y_data)})"
                )

        return self

    @property
    def x_data_as_np(self):
        return [np.array(x_series) for x_series in self.x_data]

    @property
    def y_data_as_np(self):
        return [np.array(y_series) for y_series in self.y_data]

    def _get_extent_for_background(self):
        """
        calculates good extend for background image
        """
        extent = [
            np.amin(self.x_data_as_np[0]),
            np.amax(self.x_data_as_np[0]),
            np.amin(self.y_data_as_np),
            np.amax(self.y_data_as_np),
        ]
        y_extent = extent[3] - extent[2]
        extent[2] -= y_extent * 0.1
        extent[3] += y_extent * 0.1
        return extent

    def get_plot(self):
        figsize = 8
        fig, ax = plt.subplots(figsize=(figsize, figsize / FIG_ASPECT_RATIO))
        num_series = len(self.y_data)

        if self.backgroung_png is not None:
            background_img = mpimg.imread(self.backgroung_png)
            extent = self._get_extent_for_background()
            ax.imshow(background_img, extent=extent, aspect="auto", alpha=0.7, zorder=0)
            ax.grid(True)

        for idx in range(num_series):
            x_series = self.x_data_as_np[idx]
            y_series = self.y_data_as_np[idx]
            label = (
                self.legend[idx] if self.legend and idx < len(self.legend) else f"Series {idx+1}"
            )
            if self.is_log:
                ax.semilogy(x_series, y_series, self.style, label=label)
                ax.grid(axis="y")
            else:
                ax.plot(x_series, y_series, self.style, label=label)

        if self.xlim is not None:
            ax.set_xlim(self.xlim)
        if self.ylim is not None:
            ax.set_ylim(self.ylim)

        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        if self.legend:
            ax.legend()

        return fig


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
    type_name : Literal["Chart2D"], default="Chart2D"
        Specifies the type of report item as "Chart2D"; this field is immutable.
    exclude : Optional[List[str]]
        List of boundaries to exclude from data. Applicable to:
        x_slicing_force_distribution, y_slicing_force_distribution, surface_forces
    focus_x : Optional[Tuple[float, float]]
        A tuple defining a fractional focus range along the x-axis for y-limit
        adjustments. For example, `focus_x = (0.5, 1.0)` will consider the
        subset of data corresponding to the top 50% to 100% range of x-values.
        The y-limits of the chart will be determined based on this data subset:
        the minimum and maximum y-values in this range will be found, then a
        25% margin is added above and below. This adjusted y-limit helps to
        highlight the specified portion of the chart.         
    """

    x: Union[str, Delta, DataItem]
    y: Union[str, Delta, DataItem]
    background: Union[Literal["geometry"], None] = None
    _requirements: List[str] = [_requirements_mapping["total_forces"]]
    type_name: Literal["Chart2D"] = Field("Chart2D", frozen=True)
    operations: Optional[Union[List[OperationTypes], OperationTypes]] = None
    exclude: Optional[List[str]] = None
    focus_x: Optional[Tuple[float, float]] = None

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
        return root_path.startswith("nonlinear_residuals")

    def _check_dimensions_consistency(self, data):
        if any(isinstance(d, unyt.unyt_array) for d in data):
            for d in data:
                if d is None:
                    continue
                if not isinstance(d, unyt.unyt_array):
                    raise ValueError(
                        f"data: {data} contains data with units and without, cannot create plot."
                    )

            dimesions = set([v.units.dimensions for v in data if data is not None])
            if len(dimesions) > 1:
                raise ValueError(
                    f"{data} contains data with different dimensions {dimesions=}, cannot create plot."
                )
            units = [d.units for d in data if d is not None][0]
            data = [d.to(units) for d in data if data]
            return True
        return False

    def _handle_data_with_units(self, x_data, y_data, x_label, y_label):

        if self._check_dimensions_consistency(x_data) is True:
            x_unit = x_data[0].units
            x_data = [data.value for data in x_data]
            x_label += f" [{x_unit}]"

        if self._check_dimensions_consistency(y_data) is True:
            y_unit = y_data[0].units
            y_data = [data.value for data in y_data]
            y_label += f" [{y_unit}]"

        return x_data, y_data, x_label, y_label

    def _is_multiline_data(self, x_data, y_data):
        return all(not isinstance(data, list) for data in x_data) and all(
            not isinstance(data, list) for data in y_data
        )

    def _load_data(self, cases):
        x_label = split_path(self.x)[-1]
        y_label = split_path(self.y)[-1]

        x_data = [data_from_path(case, self.x, cases) for case in cases]
        y_data = [data_from_path(case, self.y, cases) for case in cases]

        x_data, y_data, x_label, y_label = self._handle_data_with_units(
            x_data, y_data, x_label, y_label
        )

        component = x_label
        for i, data in enumerate(x_data):
            if isinstance(data, case_results.PerEntityResultCSVModel):
                data.filter(exclude=self.exclude)
                x_data[i] = data.values[component]

        component = y_label
        for i, data in enumerate(y_data):
            if isinstance(data, case_results.PerEntityResultCSVModel):
                data.filter(exclude=self.exclude)
                y_data[i] = data.values[component]

        return x_data, y_data, x_label, y_label

    def _calculate_focus_y_limits(
        self,
        x_series_list: List[List[float]],
        y_series_list: List[List[float]]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate the y-limits based on the focus_x range.
        The focus_x defines a fractional range along the dataset length.
        We slice the y data accordingly and find min/max in this range,
        then add a 25% margin above and below.
        """
        if self.focus_x is None:
            return None

        start_frac, end_frac = self.focus_x
        all_focused_y = []

        for xs, ys in zip(x_series_list, y_series_list):
            xs_np = np.array(xs)
            ys_np = np.array(ys)

            start_idx = int(len(xs_np) * start_frac)
            end_idx = int(len(xs_np) * end_frac)

            if end_idx > start_idx:
                focused_y = ys_np[start_idx:end_idx]
            else:
                focused_y = ys_np

            if len(focused_y) > 0:
                all_focused_y.extend(focused_y.tolist())

        if not all_focused_y:
            return None, None

        global_y_min = float(min(all_focused_y))
        global_y_max = float(max(all_focused_y))
        y_range = global_y_max - global_y_min
        y_min = global_y_min - 0.25 * y_range
        y_max = global_y_max + 0.25 * y_range
        return (y_min, y_max)

    def get_data(self, cases: List[Case], context: ReportContext) -> PlotModel:
        x_data, y_data, x_label, y_label = self._load_data(cases)
        background = self._get_background_chart(x_data)
        background_png = None
        if background is not None:
            background_png = background._get_images([cases[0]], context)[0]


        if self._is_multiline_data(x_data, y_data):
            x_data = [float(data) for data in x_data]
            y_data = [float(data) for data in y_data]
            legend = None
            style="o-"
        else:
            legend = [case.name for case in cases]
            style="-"

        ylim = self._calculate_focus_y_limits(x_data, y_data)

        return PlotModel(
            x_data=x_data,
            y_data=y_data,
            x_label=x_label,
            y_label=y_label,
            legend=legend,
            style=style,
            is_log=self.is_log_plot(),
            backgroung_png=background_png,
            xlim=None,
            ylim=ylim
        )

    def _get_background_chart(self, x_data):
        if self.background == "geometry":
            dimension = np.amax(x_data[0]) - np.amin(x_data[0])
            if self.x == "x_slicing_force_distribution/X":
                log.warning(
                    "First case is used as a background image with dimensiones matched to the extent of X data"
                )
                camera = Camera(
                    position=(0, -1, 0), up=(0, 0, 1), dimension=dimension, dimension_dir="width"
                )
            elif self.x == "y_slicing_force_distribution/Y":
                log.warning(
                    "First case is used as a background image with dimensiones matched to the extent of X data"
                )
                camera = Camera(
                    position=(-1, 0, 0), up=(0, 0, 1), dimension=dimension, dimension_dir="width"
                )
            else:
                raise ValueError(
                    f"background={self.background} can be only used with x == x_slicing_force_distribution/X OR x == y_slicing_force_distribution/Y"
                )
            background = Chart3D(
                show="boundaries",
                camera=camera,
                fig_name="background_" + self.fig_name,
                exclude=self.exclude,
            )
            return background
        return None

    def get_background_chart3D(self, cases) -> Tuple[Chart3D, Case]:
        x_data, y_data, x_label, y_label = self._load_data(cases)
        reference_case = cases[0]
        return self._get_background_chart(x_data), reference_case

    def _get_figures(self, cases, context: ReportContext):
        file_names = []
        case_by_case, data_storage = context.case_by_case, context.data_storage
        cbc_str = "_cbc_" if case_by_case else "_"
        case_by_case, data_storage
        if self.separate_plots:
            for case in cases:
                file_name = os.path.join(data_storage, self.fig_name + cbc_str + case.id + ".png")
                data = self.get_data([case], context)
                fig = data.get_plot()
                fig.savefig(file_name)
                file_names.append(file_name)
                plt.close()

        else:
            file_name = os.path.join(data_storage, self.fig_name + cbc_str + "all_cases" + ".png")
            data = self.get_data(cases, context)
            fig = data.get_plot()
            fig.savefig(file_name)
            file_names.append(file_name)
            plt.close()

        return file_names, data.x_label, data.y_label

    # pylint: disable=too-many-arguments,too-many-locals
    def get_doc_item(
        self,
        context: ReportContext,
    ) -> None:
        """
        returns doc item for chart
        """
        self._handle_new_page(context.doc)
        self._handle_grid_input(context.cases)
        self._handle_title(context.doc, context.section_func)
        cases = self._filter_input_cases(context.cases, context.case_by_case)

        file_names, x_lab, y_lab = self._get_figures(cases, context)

        caption = NoEscape(f'{bold(y_lab)} against {bold(x_lab)} for {bold("all cases")}.')

        if self.items_in_row is not None:
            self._add_row_figure(context.doc, file_names, caption)
        else:
            if self.separate_plots is True:
                for case, file_name in zip(cases, file_names):
                    caption = NoEscape(
                        f"{bold(y_lab)} against {bold(x_lab)} for {bold(case.name)}."
                    )
                    self._add_figure(context.doc, file_name, caption)
            else:
                self._add_figure(context.doc, file_names[-1], caption)

        context.doc.append(NoEscape(r"\FloatBarrier"))
        context.doc.append(NoEscape(r"\clearpage"))


class Chart3D(Chart):
    """
    Represents a 3D chart within a report, displaying a specific surface field.

    Parameters
    ----------
    field : Optional[SurfaceFieldNames], default=None
        The name of the surface field to display in the chart.
    limits : Optional[Tuple[float, float]], default=None
        Optional limits for the field values, specified as a tuple (min, max).
    camera: Camera
        Camera settings: camera position, look at, up.
    show : ShutterObjectTypes
        Type of object to display in the 3D chart.
    exclude : List[str], optional
        Exclude boundaries from screenshot,
    """

    field: Optional[Union[SurfaceFieldNames, str]] = None
    camera: Optional[Camera] = Camera()
    limits: Optional[Union[Tuple[float, float], Tuple[DimensionedTypes, DimensionedTypes]]] = None
    is_log_scale: bool = False
    show: Union[ShutterObjectTypes, Literal["isosurface"]]
    iso_field: Optional[Union[IsoSurfaceFieldNames, str]] = None
    exclude: Optional[List[str]] = None
    include: Optional[List[str]] = None
    type_name: Literal["Chart3D"] = Field("Chart3D", frozen=True)

    def _get_limits(self, case: Case):
        if self.limits is not None and not isinstance(self.limits[0], float):
            params: SimulationParams = case.params
            if is_flow360_unit(self.limits[0]):
                return (self.limits[0].value, self.limits[1].value)

            if isinstance(self.limits[0], unyt_quantity):
                min = params.convert_unit(self.limits[0], target_system="flow360")
                max = params.convert_unit(self.limits[1], target_system="flow360")
                return (float(min.value), float(max.value))

        return self.limits

    def _get_shutter_exclude_visibility(self):
        if self.exclude is not None:
            return [
                ActionPayload(
                    action="set-object-visibility",
                    payload=SetObjectVisibilityPayload(object_ids=self.exclude, visibility=False),
                )
            ]
        return []

    def _get_shutter_include_visibility(self):
        if self.include is not None:
            return [
                ActionPayload(
                    action="set-object-visibility",
                    payload=SetObjectVisibilityPayload(object_ids=self.include, visibility=True),
                )
            ]
        return []

    def _get_shutter_qcriterion_script(
        self,
        script: List = None,
        field: str = None,
        limits: Tuple[float, float] = None,
        is_log_scale: bool = None,
    ):
        if script is None:
            script = []

        script += [
            ActionPayload(
                action="reset-field",
                payload=ResetFieldPayload(
                    object_id="boundaries",
                ),
            ),
            ActionPayload(
                action="set-object-visibility",
                payload=SetObjectVisibilityPayload(
                    object_ids=["slices", "isosurfaces"], visibility=False
                ),
            ),
            ActionPayload(
                action="set-object-visibility",
                payload=SetObjectVisibilityPayload(
                    object_ids=["qcriterion", "boundaries"], visibility=True
                ),
            ),
        ]
        script += self._get_shutter_exclude_visibility()
        script += [
            ActionPayload(
                action="set-field",
                payload=SetFieldPayload(
                    object_id="qcriterion",
                    field_name=field,
                    min_max=limits,
                    is_log_scale=is_log_scale,
                ),
            ),
        ]
        return script

    def _get_shutter_isosurface_script(
        self,
        iso_field: str,
        script: List = None,
        field: str = None,
        limits: Tuple[float, float] = None,
        is_log_scale: bool = None,
    ):
        if script is None:
            script = []

        if iso_field != "qcriterion":
            iso_field = f"isosurface-{iso_field.lower()}"

        script += [
            ActionPayload(
                action="reset-field",
                payload=ResetFieldPayload(
                    object_id="boundaries",
                ),
            ),
            ActionPayload(
                action="set-object-visibility",
                payload=SetObjectVisibilityPayload(
                    object_ids=["slices", "isosurfaces"], visibility=False
                ),
            ),
            ActionPayload(
                action="set-object-visibility",
                payload=SetObjectVisibilityPayload(
                    object_ids=[iso_field, "boundaries"], visibility=True
                ),
            ),
        ]
        script += self._get_shutter_include_visibility()
        script += self._get_shutter_exclude_visibility()

        if field is not None:
            script += [
                ActionPayload(
                    action="set-field",
                    payload=SetFieldPayload(
                        object_id=iso_field,
                        field_name=field,
                        min_max=limits,
                        is_log_scale=is_log_scale,
                    ),
                ),
            ]
        return script

    def _get_shutter_screenshot_script(self, script, screenshot_name):
        script += [
            ActionPayload(
                action="take-screenshot",
                payload=TakeScreenshotPayload(file_name=screenshot_name, type="png"),
            )
        ]
        return script

    def _get_shutter_set_camera(self, script, camera: Camera):
        script += [
            ActionPayload(
                action="set-camera",
                payload=SetCameraPayload(**camera.model_dump(exclude_none=True)),
            )
        ]
        return script

    def _get_focus_camera(self, script):
        script += [
            ActionPayload(action="focus", payload=FocusPayload(object_ids=["boundaries"], zoom=1.5))
        ]
        return script

    def _get_shutter_boundary_script(
        self,
        script: List = None,
        field: str = None,
        limits: Tuple[float, float] = None,
        is_log_scale: bool = None,
    ):
        if script is None:
            script = []

        if self.include is not None:
            script += self._get_shutter_include_visibility()
        else:
            script += [
                ActionPayload(
                    action="set-object-visibility",
                    payload=SetObjectVisibilityPayload(object_ids=["boundaries"], visibility=True),
                )
            ]
        script += [
            ActionPayload(
                action="set-object-visibility",
                payload=SetObjectVisibilityPayload(
                    object_ids=["slices", "isosurfaces"], visibility=False
                ),
            )
        ]
        script += self._get_shutter_exclude_visibility()

        if field is None:
            pass
            # get geometry id?
        else:
            script += [
                ActionPayload(
                    action="set-field",
                    payload=SetFieldPayload(
                        object_id="boundaries",
                        field_name=field,
                        min_max=limits,
                        is_log_scale=is_log_scale,
                    ),
                )
            ]
            if field == "CfVec":
                script += [
                    ActionPayload(
                        action="set-lic",
                        payload=SetLICPayload(object_id="boundaries", visibility=True),
                    )
                ]

        return script

    def _get_shutter_slice_script(
        self,
        script: List = None,
        field: str = None,
        limits: Tuple[float, float] = None,
        is_log_scale: bool = None,
    ):
        if script is None:
            script = []

        script += [
            ActionPayload(
                action="set-object-visibility",
                payload=SetObjectVisibilityPayload(
                    object_ids=["boundaries", "isosurfaces", "slices"], visibility=False
                ),
            ),
        ]
        script += self._get_shutter_include_visibility()
        script += self._get_shutter_exclude_visibility()

        if field is None:
            pass
            # get geometry id?
        else:
            script += [
                ActionPayload(
                    action="set-field",
                    payload=SetFieldPayload(
                        object_id="slices",
                        field_name=field,
                        min_max=limits,
                        is_log_scale=is_log_scale,
                    ),
                )
            ]
            if field == "CfVec":
                script += [
                    ActionPayload(
                        action="set-lic",
                        payload=SetLICPayload(object_id="slices", visibility=True),
                    )
                ]

        return script

    def _get_shutter_request(self, case: Case):

        if self.show == "qcriterion":
            script = self._get_shutter_qcriterion_script(
                field=self.field, limits=self._get_limits(case), is_log_scale=self.is_log_scale
            )
        elif self.show == "isosurface":
            script = self._get_shutter_isosurface_script(
                iso_field=self.iso_field,
                field=self.field,
                limits=self._get_limits(case),
                is_log_scale=self.is_log_scale,
            )
        elif self.show == "boundaries":
            script = self._get_shutter_boundary_script(
                field=self.field, limits=self._get_limits(case), is_log_scale=self.is_log_scale
            )
        elif self.show == "slices":
            script = self._get_shutter_slice_script(
                field=self.field, limits=self._get_limits(case), is_log_scale=self.is_log_scale
            )
        else:
            raise ValueError(f'"{self.show}" is not corect type for 3D chart.')

        script = self._get_shutter_set_camera(script, self.camera)
        if self.camera.dimension is None:
            script = self._get_focus_camera(script)

        script = self._get_shutter_screenshot_script(script=script, screenshot_name=self.fig_name)

        scene = Scene(name="my-scene", script=script)
        path_prefix = case.get_cloud_path_prefix()
        if self.field is None and self.show == "boundaries":
            log.debug(
                "Not implemented: getting geometry resource for showing geometry. Currently using case resource."
            )
            # path_prefix = f"s3://flow360meshes-v1/users/{user_id}"
            # resource = Resource(path_prefix=path_prefix, id="geo-21a4cfb4-84c7-413f-b9ea-136ad9c2fed5")
        resource = Resource(path_prefix=path_prefix, id=case.id)
        scenes_data = ScenesData(scenes=[scene], resource=resource)
        return scenes_data

    def get_requirements(self):
        """get requirements"""
        return []

    def _get_images(self, cases: List[Case], context: ReportContext):
        shutter_requests = []
        for case in cases:
            shutter_requests.append(self._get_shutter_request(case))

        context_data = {
            "data_storage": context.data_storage,
            "url": context.shutter_url,
            "access_token": context.shutter_access_token,
            "screeshot_process_function": context.shutter_screeshot_process_function,
        }
        context_data = {k: v for k, v in context_data.items() if v is not None}
        img_files = Shutter(**context_data).get_images(self.fig_name, shutter_requests)
        # taking "first" image from returned images as UVF-shutter
        # supports many screenshots generation on one call
        img_list = [img_files[case.id][0] for case in cases if img_files[case.id] is not None]
        return img_list

    def _get_legend(self, context: ReportContext):
        if self.field is not None:
            legend_filename = os.path.join(context.data_storage, f"{self.fig_name}_legend.png")
            field = self.field
            limits = self.limits
            if isinstance(self.limits[0], unyt_quantity):
                field += f" [{self.limits[0].units}]"
                limits = (float(self.limits[0].value), float(self.limits[1].value))
            generate_colorbar_from_image(
                limits=limits,
                output_filename=legend_filename,
                field_name=field,
                is_log_scale=self.is_log_scale,
            )
            return legend_filename
        return None

    # pylint: disable=too-many-arguments
    def get_doc_item(
        self,
        context: ReportContext,
    ):
        """
        returns doc item for 3D chart
        """
        self._handle_new_page(context.doc)
        self._handle_grid_input(context.cases)
        self._handle_title(context.doc, context.section_func)
        cases = self._filter_input_cases(context.cases, context.case_by_case)

        img_list = self._get_images(cases, context)
        legend_filename = self._get_legend(context)

        if self.items_in_row is not None:
            self._add_row_figure(
                context.doc,
                img_list,
                self.caption,
                sub_fig_captions=[case.name for case in cases],
                legend_filename=legend_filename,
            )
        else:
            for filename in img_list:
                self._add_figure(
                    context.doc, filename, self.caption, legend_filename=legend_filename
                )

        # Stops figures floating away from their sections
        context.doc.append(NoEscape(r"\FloatBarrier"))
        context.doc.append(NoEscape(r"\clearpage"))
