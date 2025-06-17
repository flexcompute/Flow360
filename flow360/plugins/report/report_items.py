"""
Module containg detailed report items
"""

# pylint: disable=too-many-lines
from __future__ import annotations

import os
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from typing import Annotated, List, Literal, Optional, Tuple, Union

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pydantic as pd
import unyt
from matplotlib.ticker import FuncFormatter
from pandas import DataFrame
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

from flow360.component.case import Case
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.outputs.output_fields import (
    IsoSurfaceFieldNames,
    SurfaceFieldNames,
    get_unit_for_field,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Unsteady
from flow360.component.simulation.unit_system import (
    DimensionedTypes,
    is_flow360_unit,
    unyt_quantity,
)
from flow360.exceptions import Flow360ValidationError
from flow360.log import log
from flow360.plugins.report.report_context import ReportContext
from flow360.plugins.report.utils import (
    DataItem,
    Delta,
    Grouper,
    OperationTypes,
    Tabulary,
    _requirements_mapping,
    data_from_path,
    downsample_image_to_relative_width,
    generate_colorbar_from_image,
    get_requirements_from_data_path,
    path_variable_name,
    split_path,
)
from flow360.plugins.report.uvf_shutter import (
    ActionPayload,
    BottomCamera,
    Camera,
    FocusPayload,
    FrontCamera,
    FrontLeftBottomCamera,
    FrontLeftTopCamera,
    LeftCamera,
    RearCamera,
    RearLeftTopCamera,
    RearRightBottomCamera,
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
    TopCamera,
    make_shutter_context,
)

here = os.path.dirname(os.path.abspath(__file__))


FileNameStr = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9._-]+$")]

FIG_ASPECT_RATIO = 16 / 9


class Settings(Flow360BaseModel):
    """
    Settings for controlling output properties.
    """

    # pylint: disable=fixme
    # TODO: Create a setting class for each type of report items.
    dpi: Optional[pd.PositiveInt] = Field(
        300,
        description="The resolution in dots per inch (DPI) for generated images in report (A4 assumed).",
    )
    dump_table_csv: Optional[pd.StrictBool] = Field(
        False, description="If ``True``, :class:``Table`` data will be dumped into a csv file."
    )


class ReportItem(Flow360BaseModel):
    """
    Base class for for all report items.
    """

    _requirements: List[str] = None

    # pylint: disable=unused-argument,too-many-arguments
    def get_doc_item(self, context: ReportContext, settings: Settings = None) -> None:
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
    """

    text: Optional[str] = Field(None, description="The main content or text of the summary.")
    type_name: Literal["Summary"] = Field("Summary", frozen=True)
    _requirements: List[str] = []

    # pylint: disable=too-many-arguments
    def get_doc_item(self, context: ReportContext, settings: Settings = None) -> None:
        """
        Returns doc item for report item.
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
    Inputs is a wrapper for a specific Table setup that details key inputs from the simulation.
    """

    type_name: Literal["Inputs"] = Field("Inputs", frozen=True)
    _requirements: List[str] = [_requirements_mapping["params"]]

    # pylint: disable=too-many-arguments
    def get_doc_item(self, context: ReportContext, settings: Settings = None) -> None:
        """
        Returns doc item for inputs.
        """
        Table(
            data=[
                "params/operating_condition/velocity_magnitude",
                "params/time_stepping/type_name",
            ],
            section_title="Inputs",
            headers=[
                "Velocity",
                "Time stepping",
            ],
        ).get_doc_item(context)


def human_readable_formatter(value):
    """
    Custom formatter that uses k/M suffixes with a human-readable style.
    For large numbers, it attempts to show a concise representation without
    scientific notation:

    - For millions, it will show something like 225M (no decimals if >100),
      22.5M (one decimal if between 10 and 100), or 2.3M (two decimals if <10).
    - For thousands, it follows a similar pattern for k.
    - For numbers < 1000, it shows up to two decimal places if needed.
    """
    if not isinstance(value, (int, float)):
        return str(value)

    abs_value = abs(value)

    def strip_trailing_zeros(s):
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s

    if abs_value >= 1e6:
        scale = 1e6
        symbol = "M"
    elif abs_value >= 1e3:
        scale = 1e3
        symbol = "k"
    else:
        return str(value)

    scaled = value / scale
    if abs(scaled) < 10:
        # e.g., 2.3M
        formatted = f"{scaled:.1f}"
    elif abs(scaled) < 100:
        # e.g., 23M
        formatted = f"{scaled:.0f}"
    else:
        # e.g., 225M
        formatted = f"{scaled:.0f}"
    formatted = strip_trailing_zeros(formatted)
    return formatted + symbol


_SPECIAL_FORMATING_MAP = {"volume_mesh/stats/n_nodes": human_readable_formatter}


class Table(ReportItem):
    """
    Represents a table within a report, with configurable data and headers.
    """

    data: List[Union[str, Delta, DataItem]] = Field(
        description="A list of table data entries, which can be either strings or `Delta` objects."
    )
    section_title: Union[str, None] = Field(description="The title of the table section.")
    headers: Union[list[str], None] = Field(
        None, description="List of column headers for the table, default is None."
    )
    type_name: Literal["Table"] = Field("Table", frozen=True)
    select_indices: Optional[List[NonNegativeInt]] = Field(
        None, description="Specific indices to select for the chart."
    )
    formatter: Optional[Union[str, List[Union[str, None]]]] = Field(
        None,
        description='Formatter can be a single str (e.g. ".4g") or a list of str of the same length as ``data``',
    )

    @model_validator(mode="before")
    @classmethod
    def _process_formatter(cls, values):
        if not isinstance(values, dict):
            return values
        data = values.get("data")
        if data is None:
            return values

        formatter = values.get("formatter")

        if isinstance(formatter, str) or formatter is None:
            formatter = [formatter for _ in data]

        if len(formatter) != len(data):
            raise ValueError("List of formatters must match the length of data, or ")
        values["formatter"] = formatter

        return values

    @model_validator(mode="after")
    def _check_custom_heading_count(self) -> None:
        if self.headers is not None:
            if len(self.data) != len(self.headers):
                raise ValueError(
                    "Supplied `headers` must be the same length as `data`: "
                    f"{len(self.headers)} instead of {len(self.data)}"
                )
        return self

    # pylint: disable=unsupported-assignment-operation
    def _get_formatters(self):
        formatters = self.formatter
        for i, (fmt, data) in enumerate(zip(formatters, self.data)):
            if fmt is None:
                if isinstance(data, str) and data in _SPECIAL_FORMATING_MAP:
                    formatters[i] = _SPECIAL_FORMATING_MAP[data]
                else:
                    formatters[i] = ".5g"

        def make_callable(fmt):
            if callable(fmt):
                return fmt
            if fmt == "bool":
                return lambda x: f"{bool(x)}"
            return lambda x, ff=fmt: f"{x:{ff}}"

        formatters = [make_callable(f) for f in formatters]  # pylint: disable=not-an-iterable
        return formatters

    def get_requirements(self):
        """
        Returns requirements for this item
        """
        return get_requirements_from_data_path(self.data)

    def calculate_table_data(self, context: ReportContext) -> Tuple[List[str], List[List]]:
        """
        Calculate raw table data (headers + rows) without formatting.

        Returns:
            (headers, rows)
                headers: list of column names (strings)
                rows: list of rows, each row is a list of cell values
        """

        headers = ["Case No."]
        if self.headers is None:
            for path in self.data:
                if isinstance(path, (Delta, DataItem)):
                    field_label = str(path)
                else:
                    field_label = split_path(path)[-1]
                headers.append(field_label)
        else:
            headers.extend(self.headers)

        rows = []
        for idx, case in enumerate(context.cases):
            # pylint: disable=unsupported-membership-test
            if self.select_indices and idx not in self.select_indices:
                continue
            raw_values = []
            for path in self.data:
                value = data_from_path(
                    case,
                    path,
                    context.cases,
                    case_by_case=context.case_by_case,
                )
                raw_values.append(value)

            row_values = [str(idx + 1)] + raw_values
            rows.append(row_values)

        return (headers, rows)

    def to_dataframe(self, context: ReportContext) -> DataFrame:
        """
        Convert calculated data into a Pandas DataFrame for unit-testing
        or external usage.
        """
        headers, rows = self.calculate_table_data(context)
        df = DataFrame(rows, columns=headers)
        return df

    def get_doc_item(self, context: ReportContext, settings: Settings = None) -> None:
        """
        Returns a LaTeX doc item (Tabulary) for the table,
        using previously calculated data.
        """
        if self.section_title is not None:
            section = context.section_func(self.section_title)
            context.doc.append(section)

        headers, rows = self.calculate_table_data(context)

        num_columns = len(headers)
        with context.doc.create(Tabulary("|C" * num_columns + "|", width=num_columns)) as table:
            table.add_hline()

            table.append(Command("rowcolor", "labelgray"))
            table.add_row(headers)
            table.add_hline()

            formatters = self._get_formatters()
            for row_values in rows:

                formatted = [row_values[0]]
                for i, val in enumerate(row_values[1:], 1):
                    fmt = formatters[i - 1]
                    if isinstance(val, (int, float)):
                        formatted.append(fmt(val))
                    else:
                        formatted.append(str(val))

                table.add_row(formatted)
                table.add_hline()

        if settings is not None and settings.dump_table_csv:
            df = self.to_dataframe(context=context)
            df.to_csv(f"{self.section_title}.csv", index=False)


class PatternCaption(Flow360BaseModel):
    """
    Class for setting up chart caption.
    """

    pattern: str = Field(
        default="[case.name]",
        description="The caption pattern containing placeholders like [case.name] and [case.id]."
        + " These placeholders will be replaced with the actual case name and ID when resolving the caption."
        + ' For example, "The case is [case.name] with ID [case.id]". Defaults to ``"[case.name]"``.',
    )
    type_name: Literal["PatternCaption"] = Field("PatternCaption", frozen=True)

    # pylint: disable=no-member
    def resolve(self, case: "Case") -> str:
        """
        Resolves the pattern to the actual caption string using the provided case object.

        Parameters
        ----------
        case : Case
            The case object containing `name` and `id` attributes.

        Returns
        -------
        str
            The resolved caption string with placeholders replaced by actual values.

        Examples
        --------
        >>> caption = PatternCaption(pattern="The case is [case.name] with ID [case.id]")
        >>> case = Case(name="Example", id=123)
        >>> caption.resolve(case)
        'The case is Example with ID 123'
        """
        caption = self.pattern.replace("[case.name]", case.name)
        caption = caption.replace("[case.id]", str(case.id))
        return caption


class Chart(ReportItem):
    """
    Represents a chart in a report, with options for layout and display properties.
    """

    section_title: Optional[str] = Field(None, description="The title of the chart section.")
    fig_name: Optional[FileNameStr] = Field(
        None,
        description="Name of the figure file or identifier for the chart (). Only '^[a-zA-Z0-9._-]+$' allowed.",
    )
    fig_size: float = Field(
        0.7, description="Relative size of the figure as a fraction of text width."
    )
    items_in_row: Union[int, None] = Field(
        None, description="Number of items to display in a row within the chart section."
    )
    select_indices: Optional[List[NonNegativeInt]] = Field(
        None, description="Specific indices to select for the chart."
    )
    separate_plots: Optional[bool] = Field(
        None, description="If True, display as multiple plots; otherwise single plot."
    )
    force_new_page: bool = Field(
        False, description="If True, starts the chart on a new page in the report."
    )
    caption: Optional[Union[str, List[str], PatternCaption]] = Field(
        "", description="Caption to be shown for figures."
    )

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
            if self.separate_plots is None:
                self.separate_plots = True
            if isinstance(self.caption, List):
                raise ValueError("List of captions and items_in_row cannot be used together.")
            if isinstance(self.caption, PatternCaption):
                raise ValueError("PatternCaption and items_in_row cannot be used together.")
        return self

    def _check_caption_validity(self, cases):
        if isinstance(self.caption, List):
            if len(self.caption) != len(cases):
                raise ValueError("Caption list is not the same length as the list of cases.")

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

    def _fig_exist(self, resource_id, data_storage="."):
        img_folder = os.path.join(data_storage, resource_id)
        img_name = self.fig_name + ".png"
        img_full_path = os.path.join(img_folder, img_name)
        if os.path.exists(img_full_path):
            log.debug(f"File: {img_name=} exists in cache, reusing.")
            return True
        return False

    def _downsample_png(self, img: str, relative_width=1, dpi=None):
        if dpi is not None and img.lower().endswith(".png"):
            downsampled_img = os.path.splitext(img)[0] + f"_dpi{dpi}.png"
            downsample_image_to_relative_width(
                img, downsampled_img, relative_width=relative_width, dpi=dpi
            )
            img = downsampled_img
        return img

    # pylint: disable=too-many-arguments
    def _add_figure(self, doc: Document, file_name, caption, legend_filename=None, dpi=None):

        file_name = self._downsample_png(file_name, dpi=dpi)

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

    # pylint: disable=too-many-arguments,too-many-locals
    def _add_row_figure(
        self,
        doc: Document,
        img_list: list[str],
        fig_caption: str,
        sub_fig_captions: List[str] = None,
        legend_filename=None,
        dpi=None,
    ):
        """
        Build a figure from SubFigures which displays images in rows

        Using Doc manually here may be unnecessary - but it does allow for more control
        """

        # Smaller than 1 to avoid overflowing
        minipage_size = 0.86 / self.items_in_row if self.items_in_row != 1 else 0.8

        if sub_fig_captions is None:
            sub_fig_captions = range(1, len(img_list) + 1)

        figures = []
        for img, caption in zip(img_list, sub_fig_captions):
            sub_fig = SubFigure(position="t", width=NoEscape(rf"{minipage_size}\textwidth"))

            img = self._downsample_png(img, relative_width=minipage_size, dpi=dpi)
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
            doc.append(NoEscape(r"\begin{figure}[h!]"))
            for idx in row_idx:
                doc.append(figures[idx])
                doc.append(NoEscape(r"\hfill"))
            doc.append(NoEscape(r"\\"))
            if row_idx == idx_list[-1]:
                doc.append(NoEscape(r"\caption{" + escape_latex(fig_caption) + "}"))
            doc.append(NoEscape(r"\end{figure}"))


# pylint: disable=no-member
class PlotModel(BaseModel):
    """
    PlotModel that stores series data and configuration required to render
    a matplotlib ``Figure``.
    """

    x_data: Union[List[float], List[List[float]]] = Field(
        description="Values for the primary x-axis. Accepts a single list (one series)"
        + " or a list of lists (multiple series)."
    )
    y_data: Union[List[float], List[List[float]]] = Field(
        description="Values for the primary y-axis, matching the shape of ``x_data``."
    )
    x_label: str = Field(description="Text label for the primary x-axis.")
    y_label: str = Field(description="Text label for the primary y-axis.")
    secondary_x_data: Optional[Union[List[float], List[List[float]]]] = Field(
        None, description="Alternate x-axis values used when plotting against a secondary axis."
    )
    secondary_x_label: Optional[str] = Field(
        None,
        description="Label for the secondary x-axis (shown only if ``secondary_x_data`` is set).",
    )
    legend: Optional[List[str]] = Field(
        None,
        description="Series names to appear in the plot legend. The length should equal the number of plotted series.",
    )
    is_log: bool = Field(
        False, description="If ``True``, the y-axis is drawn on a logarithmic scale."
    )
    style: str = Field(
        "-",
        description='Matplotlib style or format string (e.g. ``"-"`` or ``"o--"``) applied to all data series.',
    )
    backgroung_png: Optional[str] = Field(
        None, description="Path to a PNG file placed behind the plot as a background image."
    )
    xlim: Optional[Tuple[float, float]] = Field(
        None, description="Axis limits for the x-axis as ``(xmin, xmax)``."
    )
    ylim: Optional[Tuple[float, float]] = Field(
        None, description="Axis limits for the y-axis as ``(ymin, ymax)``."
    )
    grid: Optional[bool] = Field(
        True, description="Show grid lines if ``True``, hide them if ``False``."
    )

    @field_validator("x_data", "y_data", mode="before")
    @classmethod
    def _ensure_y_data_is_list_of_lists(cls, v):
        if isinstance(v, list):
            if all(isinstance(item, list) for item in v):
                return v
            return [v]
        raise ValueError("x_data/y_data must be a list")

    @field_validator("secondary_x_data", mode="before")
    @classmethod
    def _ensure_secondary_x_data_identical(cls, v):
        if isinstance(v, list):
            if all((item == v[0]) for item in v):
                return v
            raise ValueError("Every series in secondary_x_data must be the same.")
        return v

    @model_validator(mode="after")
    def _check_lengths(self):
        if len(self.x_data) == 1:
            self.x_data = [self.x_data[0] for _ in self.y_data]
        if len(self.x_data) != len(self.y_data):
            raise ValueError(
                "Number of x_data series but be one or equal to number of y_data series."
            )
        for idx, (x_series, y_series) in enumerate(zip(self.x_data, self.y_data)):
            if len(x_series) != len(y_series):
                raise ValueError(
                    f"Length of y_data series at index {idx} ({len(y_series)})"
                    + f" does not match length of x_data ({len(x_series)})"
                )
        if self.legend is not None:
            if len(self.legend) != len(self.y_data):
                raise ValueError(
                    f"Length of legend ({len(self.legend)}) must match number of y_data series ({len(self.y_data)})"
                )

        return self

    @model_validator(mode="after")
    def _check_x_label_use(self):
        if (self.secondary_x_data is None) and (self.secondary_x_label is not None):
            raise ValueError(
                "Cannot define secondary x label when there is no data on secondary x axis."
            )
        return self

    @property
    def x_data_as_np(self):
        """
        returns X data as list of numpy arrays
        """
        return [np.array(x_series) for x_series in self.x_data]

    # pylint: disable=not-an-iterable
    @property
    def secondary_x_data_as_np(self):
        """
        returns secondary X data as list of numpy arrays
        """
        if self.secondary_x_data is not None:
            return [np.array(x_series) for x_series in self.secondary_x_data]
        return None

    @property
    def y_data_as_np(self):
        """
        returns Y data as list of numpy arrays
        """
        return [np.array(y_series) for y_series in self.y_data]

    def _get_extent_for_background(self):
        """
        calculates good extend for background image
        """
        extent = [
            np.min(self.x_data_as_np[0]),
            np.max(self.x_data_as_np[0]),
            np.min(np.concatenate(self.y_data_as_np)),
            np.max(np.concatenate(self.y_data_as_np)),
        ]
        y_extent = extent[3] - extent[2]
        extent[2] -= y_extent * 0.1
        extent[3] += y_extent * 0.1
        return extent

    def _calcuate_secondary_labels(self):
        locations = []
        labels = []

        curr_secondary = None

        for x_entry, sec_x_entry in zip(self.x_data_as_np[0], self.secondary_x_data_as_np[0]):
            if sec_x_entry != curr_secondary:
                locations.append(x_entry)
                labels.append(f"{sec_x_entry:g}")
                curr_secondary = sec_x_entry

        return locations, labels

    def get_plot(self):
        """
        Generates a matplotlib plot based on the provided x and y data.

        Returns
        -------
        matplotlib.figure.Figure
            A matplotlib Figure object containing the generated plot.

        Notes
        -----
        - If a background image is provided (`self.backgroung_png`), it is overlaid on the plot
        with adjusted transparency and aspect ratio.
        - The function supports both regular and logarithmic y-axis scales based on the
        `self.is_log` attribute.
        - Data series from `self.x_data` and `self.y_data` are plotted using the specified
        style (`self.style`), and legends are included if provided (`self.legend`).

        Examples
        --------
        >>> plot_model = PlotModel(
        ...     x_data=[[1, 2, 3], [1, 2, 3]],
        ...     y_data=[[4, 5, 6], [7, 8, 9]],
        ...     x_label="Time (s)",
        ...     y_label="Value",
        ...     legend=["Series A", "Series B"],
        ...     is_log=False,
        ...     style="o-"
        ... )
        >>> fig = plot_model.get_plot()
        >>> fig.show()
        """
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
            # pylint: disable=unsubscriptable-object
            label = (
                self.legend[idx] if self.legend and idx < len(self.legend) else f"Series {idx+1}"
            )

            if self.grid:
                ax.grid(True)
            if self.is_log:
                ax.semilogy(x_series, y_series, self.style, label=label)
            else:
                ax.plot(x_series, y_series, self.style, label=label)

        if self.secondary_x_data is not None:
            sec_xax = ax.secondary_xaxis(location="top")
            locations, labels = self._calcuate_secondary_labels()
            sec_xax.set_xticks(locations, labels)
            if self.secondary_x_label is not None:
                sec_xax.set_xlabel(self.secondary_x_label)

        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: format(x, "g")))

        if self.xlim is not None:
            ax.set_xlim(self.xlim)
        if self.ylim is not None:
            ax.set_ylim(self.ylim)

        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        if self.legend:
            ax.legend()

        return fig


class ManualLimit(Flow360BaseModel):
    """
    Class for setting up xlim and ylim in Chart2D by providing
    a lower and upper value of the limits.
    """

    lower: float = Field(description="Absolute value of the lower limit of an axis.")
    upper: float = Field(description="Absolute value of the upper limit of an axis.")
    type_name: Literal["ManualLimit"] = Field("ManualLimit", frozen=True)


class SubsetLimit(Flow360BaseModel):
    """
    Class for setting up ylim in Chart2D by providing
    a subset of values and an offset, which will be applied
    to the range of y values.
    """

    subset: Tuple[pd.NonNegativeFloat, pd.NonNegativeFloat] = Field(
        description="Tuple of fractions between 0 and 1 describing the lower"
        + " and upper range of the subset of values that will be used to calculate the ylim."
    )
    offset: float = Field(
        description='"Padding" that will be added to the top and bottom of the charts y_range.'
        + " It scales with with calculated range of y values. For example, if range of y value is 10,"
        + ' an offset=0.3 will "expand" the range by 0.3*10 on both sides,'
        + " resulting in a final range of y values equal to 16."
    )
    type_name: Literal["SubsetLimit"] = Field("SubsetLimit", frozen=True)

    @pd.model_validator(mode="after")
    def check_subset_values(self):
        """
        Ensure that correct subset values are provided.
        """
        lower, upper = self.subset
        if not lower < 1 or not upper <= 1:
            raise ValueError("Subset values need to be between 0 and 1 (inclusive).")
        if not lower <= upper:
            raise ValueError("Lower fraction of the subset cannot be higher than upper fraction.")
        return self


class FixedRangeLimit(Flow360BaseModel):
    """
    Class for setting up ylim in Chart2D by providing
    a fixed range of y values and strategy for centering.
    """

    fixed_range: float = Field(
        description="Range of absolute y values that will be visible on the chart."
        + " For example, fixed_range=3 means that y_max - y_min = 3."
    )
    center_strategy: Literal["last", "last_percent"] = Field(
        "last",
        description="Describes which values will be considered for calculating ylim."
        + ' "last" means that the last value will be the center. "last_percent"'
        + " means that the middle point between max and min y values"
        + " in the specified center_fraction will be the center.",
    )
    center_fraction: Optional[pd.PositiveFloat] = Field(
        None,
        description='Used alongside center_strategy="last_percent",'
        + " describes values that will be taken into account for calculating ylim."
        + " For example, center_fraction=0.3 means that the last 30% of data will be used.",
    )
    type_name: Literal["FixedRangeLimit"] = Field("FixedRangeLimit", frozen=True)

    @pd.model_validator(mode="after")
    def check_center_fraction(self):
        """Ensure that correct center fraction value is provided."""
        if self.center_strategy == "last_percent" and self.center_fraction >= 1:
            raise ValueError("Center fraction value needs to be between 0 and 1 (exclusive).")
        return self


class BaseChart2D(Chart, metaclass=ABCMeta):
    """
    Base class for Chart2D like objects - does not contain data.
    """

    operations: Optional[Union[List[OperationTypes], OperationTypes]] = Field(
        None, description="List of operations to perform on the data."
    )
    focus_x: Optional[
        Annotated[
            Tuple[float, float],
            Field(
                deprecated="focus_x is deprecated, your input was converted to a corresponding SubsetLimit. "
                + "Please use ylim=SubsetLimit instead in the future.",
            ),
        ]
    ] = None
    xlim: Optional[Union[ManualLimit, Tuple[float, float]]] = Field(
        None, description="Defines the range of x values that will be displayed on the chart."
    )
    ylim: Optional[Union[ManualLimit, SubsetLimit, FixedRangeLimit, Tuple[float, float]]] = Field(
        None,
        description="Defines the range of y values that will be displayed on the chart."
        + " This helps with highlighting a desired portion of the chart.",
    )
    y_log: Optional[bool] = Field(
        False, description="Sets the y axis to logarithmic scale. Defaults to ``False``."
    )
    show_grid: Optional[bool] = Field(
        True, description="Turns the gridlines on. Defaults to ``True``."
    )

    def is_log_plot(self):
        """
        Determines if the plot is logarithmic.

        Returns
        -------
        bool
        """
        return self.y_log is True

    # pylint: disable=unpacking-non-sequence
    @pd.model_validator(mode="after")
    def _handle_deprecated_focus_x(self):
        """
        Ensures that scripts containing deprecated focus_x will still work.
        """
        if self.focus_x is not None:
            if self.ylim is not None:
                raise ValueError("Fields ylim and focus_x cannot be used together.")
            lower, upper = self.focus_x
            self.focus_x = None
            self.ylim = SubsetLimit(subset=(lower, upper), offset=0.25)
        return self

    @pd.model_validator(mode="after")
    def _check_caption_separate_plots(self):
        if self.separate_plots is not True:
            if isinstance(self.caption, List):
                raise ValueError(
                    "List of captions is only supported for Chart2D when separate_plots is True."
                )
            if isinstance(self.caption, PatternCaption):
                raise ValueError(
                    "PatternCaption is only supported for Chart2D when separate_plots is True."
                )
        return self

    def _check_dimensions_consistency(self, data):
        if any(isinstance(d, unyt.unyt_array) for d in data):
            for d in data:
                if d is None:
                    continue
                if not isinstance(d, unyt.unyt_array):
                    raise ValueError(
                        f"data: {data} contains data with units and without, cannot create plot."
                    )

            dimesions = {v.units.dimensions for v in data if data is not None}
            if len(dimesions) > 1:
                raise ValueError(
                    f"{data} contains data with different dimensions {dimesions=}, cannot create plot."
                )
            units = [d.units for d in data if d is not None][0]
            data = [d.to(units) for d in data if data]
            return True
        return False

    def _unpack_data_to_multiline(self, x_data: list, y_data: list):
        if (
            len(x_data) == 1
            and isinstance(x_data[0], list)
            and len(y_data) == 1
            and isinstance(y_data[0], list)
        ):
            return x_data[0], y_data[0]
        return x_data, y_data

    def _is_multiline_data(self, x_data: list, y_data: list):
        return all(not isinstance(data, list) for data in x_data) and all(
            not isinstance(data, list) for data in y_data
        )

    @abstractmethod
    def _get_background_chart(self, _):
        pass

    def _handle_xlimits(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Make sure that xlim is always passed
        as a tuple of floats to the plotting tool.
        """
        xlim = self.xlim
        if xlim is None:
            return None

        if isinstance(xlim, ManualLimit):
            return (xlim.lower, xlim.upper)

        return xlim

    def _calculate_subset(self, x_series_list, y_series_list, start_frac, end_frac):
        """
        Based on provided data series and fraction start and end,
        calculate the corresponding subset of data.
        """
        all_subset_y = []

        for xs, ys in zip(x_series_list, y_series_list):
            xs_np = np.array(xs)
            ys_np = np.array(ys)

            start_idx = int(len(xs_np) * start_frac)
            end_idx = int(len(xs_np) * end_frac)

            if end_idx > start_idx:
                subset_y = ys_np[start_idx:end_idx]
            else:
                subset_y = ys_np

            if len(subset_y) > 0:
                all_subset_y.extend(subset_y.tolist())

        return all_subset_y

    def _calculate_y_min_max(
        self, all_subset_y, type_name: Literal["SubsetLimit", "FixedRangeLimit"]
    ) -> Tuple[float, float]:
        """
        Given a subset of data and ylim type,
        calculate min and max y values.
        """
        subset_y_min = float(min(all_subset_y))
        subset_y_max = float(max(all_subset_y))

        if type_name == "SubsetLimit":
            y_range = subset_y_max - subset_y_min
            y_min = subset_y_min - self.ylim.offset * y_range
            y_max = subset_y_max + self.ylim.offset * y_range

        elif type_name == "FixedRangeLimit":
            y_center = (subset_y_max + subset_y_min) / 2
            y_min = y_center - 0.5 * self.ylim.fixed_range
            y_max = y_center + 0.5 * self.ylim.fixed_range
        else:
            raise ValueError(f"Unknown type_name: {type_name}.")

        return (y_min, y_max)

    def _calculate_ylimits(
        self, x_series_list: List[List[float]], y_series_list: List[List[float]]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate ylim based on provided input.
        """
        ylim = self.ylim

        if ylim is None:
            return None

        if isinstance(ylim, Tuple):
            return ylim

        if isinstance(ylim, ManualLimit):
            y_range = (ylim.lower, ylim.upper)

        elif isinstance(ylim, SubsetLimit):
            start_frac, end_frac = ylim.subset

            all_subset_y = self._calculate_subset(
                x_series_list, y_series_list, start_frac, end_frac
            )

            if not all_subset_y:
                return (None, None)

            y_range = self._calculate_y_min_max(all_subset_y, ylim.type_name)

        else:
            if ylim.center_strategy == "last":
                all_last_y = []
                for ys in y_series_list:
                    last_y = ys[-1]
                    all_last_y.append(last_y)

            else:
                start_frac = 1 - ylim.center_fraction
                end_frac = 1
                all_last_y = self._calculate_subset(
                    x_series_list, y_series_list, start_frac, end_frac
                )

            if not all_last_y:
                return (None, None)

            y_range = self._calculate_y_min_max(all_last_y, ylim.type_name)

        return y_range

    @abstractmethod
    def _load_data(self, cases):
        pass

    @abstractmethod
    def _handle_legend(self, cases, x_data, y_data):
        pass

    def _handle_plot_style(self, x_data, y_data):
        if self._is_multiline_data(x_data, y_data):
            style = "o-"
        else:
            style = "-"

        return style

    def _cumulate_pseudo_step(self, pseudo_steps):
        cumulative = []
        last = 0
        for step in pseudo_steps:
            if (step == 0) and cumulative:
                last = cumulative[-1] + 1
            cumulative.append(step + last)

        return cumulative

    def _handle_transient_pseudo_step(self, cases, x_data, x_label):
        """
        Converts pseudo_step to cumulative pseudo_step.
        """
        if x_label == "pseudo_step" and any(
            isinstance(case.params.time_stepping, Unsteady) for case in cases
        ):
            for idx, x_series in enumerate(x_data):
                x_data[idx] = self._cumulate_pseudo_step(x_series)

    def _handle_secondary_x_axis(self, cases, x_data, x_lim, x_label):
        """
        Creates physical_step array to use on the secondary axis
        when the primary axis is pseudo_step, and the number of physical_steps visible
        within x axis limits is less or equal than 5.
        """
        if x_label == "pseudo_step" and any(
            isinstance(case.params.time_stepping, Unsteady) for case in cases
        ):
            if len(cases) == 1:
                path_to_physical_step = self.x.rstrip("pseudo_step") + "physical_step"
                sec_x_data = data_from_path(cases[0], path_to_physical_step, [])
                x_min = min(x_data[0])
                x_max = max(x_data[0])
                lower_idx = x_data[0].index(max(x_min, x_lim[0] if x_lim is not None else -np.inf))
                upper_idx = x_data[0].index(min(x_max, x_lim[1] if x_lim is not None else np.inf))
                physical_steps_to_show = sec_x_data[upper_idx] - sec_x_data[lower_idx]
                if physical_steps_to_show <= 5:
                    return [sec_x_data] * len(x_data), "physical_step"
            else:
                log.warning("Does not show physical step with multiple cases plotted.")
        return None, None

    def _remove_empty_series(self, x_data, y_data, legend):
        to_pop = []
        for idx, x_series in enumerate(x_data):
            if not x_series:
                to_pop.append(idx)

        x_data = [series for series in x_data if series]
        y_data = [series for series in y_data if series]
        legend = [item for idx, item in enumerate(legend) if idx not in to_pop]

        return x_data, y_data, legend

    def get_data(self, cases: List[Case], context: ReportContext) -> PlotModel:
        """
        Loads and processes data for creating a 2D plot model.

        Parameters
        ----------
        cases : List[Case]
            A list of simulation cases to extract data from.
        context : ReportContext
            The report context providing additional configuration and case-specific data.

        Returns
        -------
        PlotModel
            A `PlotModel` instance containing the processed x and y data, axis labels,
            legend, and optional background image for plotting.

        Notes
        -----
        - Handles data with physical units and ensures dimensional consistency.
        - Supports optional background images for geometry-related plots.
        - Automatically determines y-axis limits if `focus_x` is specified.

        Examples
        --------
        >>> chart = Chart2D(
        ...     x="x_slicing_force_distribution/X",
        ...     y="x_slicing_force_distribution/totalCumulative_CD_Curve",
        ...     background="geometry"
        ... )
        >>> plot_model = chart.get_data(cases, context)
        >>> fig = plot_model.get_plot()
        >>> fig.show()
        """
        x_data, y_data, x_label, y_label = self._load_data(cases)

        background = self._get_background_chart(x_data)
        background_png = None
        if background is not None:
            # pylint: disable=protected-access
            background_png = background._get_images([cases[0]], context)[0]

        x_data, y_data = self._unpack_data_to_multiline(x_data=x_data, y_data=y_data)

        legend = self._handle_legend(cases, x_data, y_data)

        if not self._is_multiline_data(x_data, y_data):
            x_data, y_data, legend = self._remove_empty_series(x_data, y_data, legend)

        style = self._handle_plot_style(x_data, y_data)

        xlim = self._handle_xlimits()
        ylim = self._calculate_ylimits(x_data, y_data)

        self._handle_transient_pseudo_step(x_data=x_data, x_label=x_label, cases=cases)

        secondary_x_data, seondary_x_label = self._handle_secondary_x_axis(
            x_lim=xlim, x_data=x_data, x_label=x_label, cases=cases
        )

        return PlotModel(
            x_data=x_data,
            y_data=y_data,
            x_label=x_label,
            y_label=y_label,
            legend=legend,
            style=style,
            is_log=self.is_log_plot(),
            backgroung_png=background_png,
            xlim=xlim,
            ylim=ylim,
            grid=self.show_grid,
            secondary_x_data=secondary_x_data,
            secondary_x_label=seondary_x_label,
        )

    def _get_figures(self, cases, context: ReportContext):
        file_names = []
        case_by_case, data_storage = context.case_by_case, context.data_storage
        cbc_str = "_cbc_" if case_by_case else "_"
        if self.separate_plots:
            for case in cases:
                file_name = os.path.join(data_storage, self.fig_name + cbc_str + case.id + ".pdf")
                data = self.get_data([case], context)
                fig = data.get_plot()
                fig.savefig(file_name, format="pdf", bbox_inches="tight")
                file_names.append(file_name)
                plt.close()

        else:
            file_name = os.path.join(data_storage, self.fig_name + cbc_str + "all_cases" + ".pdf")
            data = self.get_data(cases, context)
            fig = data.get_plot()
            fig.savefig(file_name, format="pdf", bbox_inches="tight")
            file_names.append(file_name)
            plt.close()

        return file_names, data.x_label, data.y_label

    # pylint: disable=too-many-return-statements
    def _handle_2d_caption(
        self, case: Case = None, x_lab: str = None, y_lab: str = None, case_number: int = None
    ):
        """
        Handle captions for Chart2D.
        """

        if self.caption == "":
            if self.items_in_row is not None:
                return f"{bold(y_lab)} against {bold(x_lab)}."
            if self.separate_plots is True:
                return f"{bold(y_lab)} against {bold(x_lab)} for {bold(case.name)}."
            if self.select_indices is not None:
                return f"{bold(y_lab)} against {bold(x_lab)} for {bold('selected cases')}."
            return f"{bold(y_lab)} against {bold(x_lab)} for {bold('all cases')}."
        if self.separate_plots is True:
            if isinstance(self.caption, List):
                return escape_latex(self.caption[case_number])
            if isinstance(self.caption, PatternCaption):
                return escape_latex(self.caption.resolve(case))
        return self.caption

    # pylint: disable=too-many-arguments,too-many-locals
    def get_doc_item(self, context: ReportContext, settings: Settings = None) -> None:
        """
        Returns doc item for chart.
        """
        self._handle_new_page(context.doc)
        self._handle_grid_input(context.cases)
        self._handle_title(context.doc, context.section_func)
        cases = self._filter_input_cases(context.cases, context.case_by_case)
        self._check_caption_validity(cases)

        file_names, x_lab, y_lab = self._get_figures(cases, context)

        if self.items_in_row is not None:
            caption = NoEscape(self._handle_2d_caption(x_lab=x_lab, y_lab=y_lab))
            self._add_row_figure(context.doc, file_names, caption, [case.name for case in cases])
        else:
            if self.separate_plots is True:
                for case_number, (case, file_name) in enumerate(zip(cases, file_names)):
                    caption = NoEscape(
                        self._handle_2d_caption(
                            case=case, x_lab=x_lab, y_lab=y_lab, case_number=case_number
                        )
                    )
                    self._add_figure(context.doc, file_name, caption)
            else:
                caption = NoEscape(self._handle_2d_caption(x_lab=x_lab, y_lab=y_lab))
                self._add_figure(context.doc, file_names[-1], caption)

        context.doc.append(NoEscape(r"\FloatBarrier"))
        context.doc.append(NoEscape(r"\clearpage"))


class Chart2D(BaseChart2D):
    """
    Represents a 2D chart within a report, plotting x and y data.

    Example
    -------

    -  Create a chart of CL for an alpha sweep case, different turbulence models

    >>> Chart2D(
    ...     x="params/operating_condition/beta",
    ...     y=DataItem(data="total_forces/CL", operations=[Average(fraction=0.1)]),
    ...     section_title="CL vs alpha",
    ...     fig_name="cl_vs_alpha",
    ...     group_by=Grouper(group_by="params/models/Fluid/turbulence_model_solver/type_name"),
    ... )

    ====
    """

    x: Union[DataItem, Delta, str] = Field(
        description="The data source for the x-axis, which can be a string path, 'DataItem', a 'Delta' object."
    )
    y: Union[DataItem, Delta, str, List[DataItem], List[Delta], List[str]] = Field(
        description="The data source for the y-axis, which can be a string path,"
        + " 'DataItem', a 'Delta' object or their list."
    )
    group_by: Optional[Union[str, Grouper]] = Field(
        Grouper(group_by=None),
        description="A grouper object or a string leading to the data by which the grouping should be done.",
    )
    include: Optional[
        Annotated[
            List[str],
            Field(
                deprecated="Include and exclude are deprecated as Chart2D options, use DataItem instead."
            ),
        ]
    ] = Field(
        None,
        description="List of boundaries to include in data. Applicable to:"
        + " x_slicing_force_distribution, y_slicing_force_distribution, surface_forces.",
    )
    exclude: Optional[
        Annotated[
            List[str],
            Field(
                deprecated="Include and exclude are deprecated as Chart2D options, use DataItem instead."
            ),
        ]
    ] = Field(
        None,
        description="List of boundaries to exclude from data. Applicable to:"
        + " x_slicing_force_distribution, y_slicing_force_distribution, surface_forces.",
    )
    background: Union[Literal["geometry"], None] = Field(
        None,
        description='Background type for the chart; set to "geometry" or None. Defaults to ``None``.',
    )
    _requirements: List[str] = [_requirements_mapping["total_forces"]]
    type_name: Literal["Chart2D"] = Field("Chart2D", frozen=True)

    @pd.model_validator(mode="after")
    def _handle_deprecated_include_exclude(self):
        include = self.include
        exclude = self.exclude
        if (include is not None) or (exclude is not None):
            self.include = None
            self.exclude = None
            self.x = self._overload_include_exclude(include, exclude, self.x)
            if isinstance(self.y, List):
                new_value = []
                for data_variable in self.y:
                    new_value.append(
                        self._overload_include_exclude(include, exclude, data_variable)
                    )
                self.y = new_value
            else:
                self.y = self._overload_include_exclude(include, exclude, self.y)
        return self

    @pd.model_validator(mode="after")
    def _create_grouper(self):
        if isinstance(self.group_by, str):
            self.group_by = Grouper(group_by=self.group_by)
        return self

    @classmethod
    def _overload_include_exclude(cls, include, exclude, data_variable):
        if isinstance(data_variable, Delta):
            raise Flow360ValidationError(
                "Delta can not be used with exclude/include options. "
                + "Specify the Delta data using DataItem."
            )
        if not isinstance(data_variable, DataItem):
            data_variable = DataItem(data=data_variable, include=include, exclude=exclude)
        else:
            data_variable.include = include
            data_variable.exclude = exclude
        return data_variable

    def get_requirements(self):
        """
        Returns requirements for this item.
        """
        if isinstance(self.y, list):
            return get_requirements_from_data_path([self.x, *self.y])
        return get_requirements_from_data_path([self.x, self.y])

    # pylint: disable=no-member
    def _handle_data_with_units(self, x_data, y_data, x_label, y_label):
        for idx, (x_series, y_series) in enumerate(zip(x_data, y_data)):
            united_array_x = unyt.unyt_array(x_series)
            united_array_y = unyt.unyt_array(y_series)
            if united_array_x.units != unyt.dimensionless:
                x_data[idx] = united_array_x
            if united_array_y.units != unyt.dimensionless:
                y_data[idx] = united_array_y

        if self._check_dimensions_consistency(x_data) is True:
            x_unit = x_data[0].units
            x_data = [data.value.tolist() for data in x_data]
            x_label += f" [{x_unit}]"

        if self._check_dimensions_consistency(y_data) is True:
            y_unit = y_data[0].units
            y_data = [data.value.tolist() for data in y_data]
            if not isinstance(self.y, list):
                y_label += f" [{y_unit}]"

        return x_data, y_data, x_label, y_label

    def _handle_legend(self, cases, x_data, y_data):
        if not self._is_series_data(cases[0], self.x):
            return self.group_by.arrange_legend()

        if self._is_multiline_data(x_data, y_data):
            x_data = [float(data) for data in x_data]
            y_data = [float(data) for data in y_data]
            legend = None
        elif isinstance(self.y, list) and (len(self.y) > 1):
            legend = []
            for case in cases:
                for y in self.y:
                    if len(cases) > 1:
                        legend.append(f"{case.name} - {path_variable_name(str(y))}")
                    else:
                        legend.append(f"{path_variable_name(str(y))}")
        else:
            legend = [case.name for case in cases]

        return legend

    def _is_series_data(self, example_case, variable):
        data_point = data_from_path(example_case, variable, None)
        if isinstance(data_point, Iterable):
            if isinstance(data_point, unyt_quantity) and data_point.shape == ():
                return False
            return True
        return False

    def _validate_variable_format(self, example_case, x_variable, y_variables):
        series = self._is_series_data(example_case, x_variable)

        for y in y_variables:
            if series != self._is_series_data(example_case, y):
                raise AttributeError(
                    "Variables incompatible - cannot plot point and series data on the same plot."
                )

    def _load_series(self, cases, x_label, y_variables):
        x_data = []
        y_data = []
        for case in cases:
            filter_physical_steps = isinstance(case.params.time_stepping, Unsteady) and (
                x_label in ["time", "physical_step"]
            )
            for y in y_variables:
                x_data.append(
                    data_from_path(
                        case, self.x, cases, filter_physical_steps_only=filter_physical_steps
                    )
                )
                y_data.append(
                    data_from_path(case, y, cases, filter_physical_steps_only=filter_physical_steps)
                )

        return x_data, y_data

    def _load_points(self, cases, y_variables):
        x_data, y_data = self.group_by.initialize_arrays(cases, y_variables)
        for case in cases:
            for y in y_variables:
                x_data_point = data_from_path(case, self.x, cases)
                y_data_point = data_from_path(case, y, cases)
                x_data, y_data = self.group_by.arrange_data(
                    case, x_data, y_data, x_data_point, y_data_point, y
                )

        return x_data, y_data

    def _load_data(self, cases):
        x_label = path_variable_name(str(self.x))

        if not isinstance(self.y, list):
            y_label = path_variable_name(str(self.y))
            y_variables = [self.y]
        else:
            y_label = "value"
            y_variables = self.y.copy()

        self._validate_variable_format(cases[0], self.x, y_variables)

        if self._is_series_data(cases[0], self.x):
            x_data, y_data = self._load_series(cases, x_label, y_variables)
        else:
            x_data, y_data = self._load_points(cases, y_variables)

        x_data, y_data, x_label, y_label = self._handle_data_with_units(
            x_data, y_data, x_label, y_label
        )

        return x_data, y_data, x_label, y_label

    def _get_background_chart(self, x_data):
        if self.background == "geometry":
            dimension = np.amax(x_data[0]) - np.amin(x_data[0])
            if self.x == "x_slicing_force_distribution/X":
                log.warning(
                    "First case is used as a background image with dimensions matched to the extent of X data"
                )
                camera = Camera(
                    position=(0, -1, 0), up=(0, 0, 1), dimension=dimension, dimension_dir="width"
                )
            elif self.x == "y_slicing_force_distribution/Y":
                log.warning(
                    "First case is used as a background image with dimensions matched to the extent of X data"
                )
                camera = Camera(
                    position=(-1, 0, 0), up=(0, 0, 1), dimension=dimension, dimension_dir="width"
                )
            else:
                raise ValueError(
                    f"background={self.background} can be only used with x == x_slicing_force_distribution/X"
                    + " OR x == y_slicing_force_distribution/Y"
                )
            background = Chart3D(
                show="boundaries",
                camera=camera,
                fig_name="background_" + self.fig_name,
                include=self.include,
                exclude=self.exclude,
            )
            return background
        return None

    def get_background_chart3d(self, cases) -> Tuple[Chart3D, Case]:
        """
        Returns Chart3D for background.
        """
        # pylint: disable=unsubscriptable-object
        reference_case_idx = self.select_indices[0] if self.select_indices is not None else 0
        reference_case = cases[reference_case_idx]
        x_data, _, _, _ = self._load_data([reference_case])
        return self._get_background_chart(x_data), reference_case


class NonlinearResiduals(BaseChart2D):
    """
    Residuals is an object for showing the solution history of nonlinear residuals.
    """

    show_grid: Optional[bool] = Field(
        True, description="If ``True``, grid lines are displayed on the plot. Defaults to ``True``."
    )
    separate_plots: Optional[bool] = Field(
        True, description="If ``True``, each residual component is plotted in a separate subplot."
    )
    xlim: Optional[Union[ManualLimit, Tuple[float, float]]] = Field(
        None,
        description="Limits for the *x*-axis. Can be a tuple ``(xmin, xmax)`` or a `ManualLimit`.",
    )
    section_title: Literal["Nonlinear residuals"] = Field("Nonlinear residuals", frozen=True)
    x: Literal["nonlinear_residuals/pseudo_step"] = Field(
        "nonlinear_residuals/pseudo_step", frozen=True
    )
    y_log: Literal[True] = Field(True, frozen=True)
    _requirements: List[str] = [_requirements_mapping["nonlinear_residuals"]]
    type_name: Literal["NonlinearResiduals"] = Field("NonlinearResiduals", frozen=True)

    def get_requirements(self):
        """
        Returns requirements for this item.
        """
        return self._requirements

    def _get_background_chart(self, _):
        return None

    def _handle_legend(self, cases, _, y_data):
        cols_exclude = cases[0].results.nonlinear_residuals.x_columns
        legend = []
        for case in cases:
            y_variables = [
                f"nonlinear_residuals/{res}"
                for res in case.results.nonlinear_residuals.as_dict().keys()
                if res not in cols_exclude
            ]
            legend += [
                (
                    f"{case.name} - {path_variable_name(y)}"
                    if len(cases) > 1
                    else f"{path_variable_name(y)}"
                )
                for y in y_variables
            ]

        return legend

    def _handle_secondary_x_axis(self, cases, x_data, x_lim, x_label):
        secondary_x_data, seondary_x_label = super()._handle_secondary_x_axis(
            cases, x_data, x_lim, x_label
        )
        if secondary_x_data is not None:
            return np.array(secondary_x_data)[:, 1:].tolist(), seondary_x_label
        return secondary_x_data, seondary_x_label

    def _load_data(self, cases):
        cols_exclude = cases[0].results.nonlinear_residuals.x_columns
        x_label = path_variable_name(self.x)
        y_label = "residual values"

        x_data = []
        y_data = []

        for case in cases:
            y_variables = [
                f"nonlinear_residuals/{res}"
                for res in case.results.nonlinear_residuals.as_dict().keys()
                if res not in cols_exclude
            ]
            for y in y_variables:
                x_data.append(data_from_path(case, self.x, cases)[1:])
                y_data.append(data_from_path(case, y, cases)[1:])

        return x_data, y_data, x_label, y_label


class Chart3D(Chart):
    """
    Represents a 3D chart within a report, displaying a specific surface field.
    """

    field: Optional[Union[SurfaceFieldNames, str]] = Field(
        None, description="The name of the field to display in the chart."
    )
    camera: Optional[
        Union[
            Camera,
            BottomCamera,
            FrontCamera,
            FrontLeftBottomCamera,
            FrontLeftTopCamera,
            LeftCamera,
            RearCamera,
            RearLeftTopCamera,
            RearRightBottomCamera,
            TopCamera,
        ]
    ] = pd.Field(
        default=Camera(), description="Specify how the view will be set up.", discriminator="type"
    )
    limits: Optional[Union[Tuple[float, float], Tuple[DimensionedTypes, DimensionedTypes]]] = Field(
        None, description="Limits for the field values, specified as a tuple (min, max)."
    )
    is_log_scale: bool = Field(
        False, description="Applies a logarithmic scale to the colormap. Defaults to ``False``."
    )
    show: ShutterObjectTypes = Field(
        description="Type of object to display in the 3D chart. Note: ``qcriterion`` refers to an iso-surface"
        + " that is created by default, whereas ``isosurface`` refers to iso-surfaces specified in simulation outputs."
    )
    iso_field: Optional[Union[IsoSurfaceFieldNames, str]] = Field(
        None,
        description="Iso-surface fields to be displayed when ``isosurface`` is selected in ``show``.",
    )
    mode: Optional[Literal["contour", "lic"]] = Field(
        "contour", description="Field display mode, lic stands for line integral convolution."
    )
    include: Optional[List[str]] = Field(
        None, description="Boundaries to be included in the chart."
    )
    exclude: Optional[List[str]] = Field(
        None, description="Boundaries to be excluded from the chart."
    )
    type_name: Literal["Chart3D"] = Field("Chart3D", frozen=True)

    # pylint: disable=unsubscriptable-object
    def _get_limits(self, case: Case):
        if self.limits is not None and not isinstance(self.limits[0], float):
            params: SimulationParams = case.params
            if is_flow360_unit(self.limits[0]):
                return (self.limits[0].value, self.limits[1].value)

            if isinstance(self.limits[0], unyt_quantity):
                _, unit_system = get_unit_for_field(self.field)
                target_system = "flow360"
                if unit_system is not None:
                    target_system = unit_system
                min_val = params.convert_unit(self.limits[0], target_system=target_system)
                max_val = params.convert_unit(self.limits[1], target_system=target_system)
                return (float(min_val.value), float(max_val.value))

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

    # pylint: disable=too-many-arguments
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
                payload=SetCameraPayload(**camera.model_dump(exclude_none=True, exclude=["type"])),
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
            script += [
                ActionPayload(
                    action="set-lic",
                    payload=SetLICPayload(object_id="boundaries", visibility=self.mode == "lic"),
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
            if self.include is not None:
                script += [
                    ActionPayload(
                        action="set-lic",
                        payload=SetLICPayload(object_id=slice, visibility=self.mode == "lic"),
                    )
                    for slice in self.include  # pylint: disable=not-an-iterable
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
        if self.camera.dimension is None:  # pylint: disable=no-member
            script = self._get_focus_camera(script)

        script = self._get_shutter_screenshot_script(script=script, screenshot_name=self.fig_name)

        scene = Scene(name="my-scene", script=script)
        path_prefix = case.get_cloud_path_prefix()
        if self.field is None and self.show == "boundaries":
            log.debug(
                "Not implemented: getting geometry resource for showing geometry. Currently using case resource."
            )
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

        img_files = Shutter(**make_shutter_context(context)).get_images(
            self.fig_name, shutter_requests, regenerate_if_not_found=False
        )
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

    def _handle_3d_caption(self, case: Case = None, case_number: int = None):
        """Handle captions for Chart3D."""

        if isinstance(self.caption, List):
            caption = self.caption[case_number]
        elif isinstance(self.caption, PatternCaption):
            caption = self.caption.resolve(case)
        else:
            caption = self.caption

        return caption

    # pylint: disable=too-many-arguments
    def get_doc_item(self, context: ReportContext, settings: Settings = None):
        """
        returns doc item for 3D chart
        """
        self._handle_new_page(context.doc)
        self._handle_grid_input(context.cases)
        self._handle_title(context.doc, context.section_func)
        cases = self._filter_input_cases(context.cases, context.case_by_case)
        self._check_caption_validity(cases)

        img_list = self._get_images(cases, context)
        legend_filename = self._get_legend(context)

        dpi = None
        if settings is not None:
            dpi = settings.dpi

        if self.items_in_row is not None:
            caption = self._handle_3d_caption()
            self._add_row_figure(
                context.doc,
                img_list,
                caption,
                sub_fig_captions=[case.name for case in cases],
                legend_filename=legend_filename,
                dpi=dpi,
            )
        else:
            for case_number, (case, filename) in enumerate(zip(cases, img_list)):
                caption = self._handle_3d_caption(case=case, case_number=case_number)
                self._add_figure(
                    context.doc, filename, caption, legend_filename=legend_filename, dpi=dpi
                )

        # Stops figures floating away from their sections
        context.doc.append(NoEscape(r"\FloatBarrier"))
        context.doc.append(NoEscape(r"\clearpage"))
