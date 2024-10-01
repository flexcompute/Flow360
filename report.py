import os
import random
import requests
import matplotlib.pyplot as plt

from pydantic import BaseModel, model_validator
from typing import List, Literal, Union, Any

from pylatex import Document, Section, Subsection, Tabular, Figure, NoEscape, Head, Foot, PageStyle, SubFigure, Package
from pylatex.utils import bold

from flow360 import Case

def get_case_from_id(id: str, cases: list[Case]) -> Case:
    # This can happen if Delta has no ref_case
    if len(cases) == 0:
        raise ValueError("No cases provided for `get_case_from_id`.")
    for case in cases:
        if case.id == id:
            return case

def data_from_path(case: Case, path:str, cases: list[Case]=[]) -> Any:
    # Handle Delta values
    if isinstance(path, Delta):
        ref_case = get_case_from_id(path.ref_case_id, cases)
        return path.calculate(case, ref_case)

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
            elif "value" in case:
                return case["value"]
            else:
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

        # Check if component is a key of a value
        try:
            return case.values[component]
        except KeyError:
            raise ValueError(f"Could not find path component: '{component}'")

    # Case variable is slightly misleading as this is only a case on the first iteration
    for component in path_components:
        case = _search_path(case, component)

    return case

class ReportItem(BaseModel):

    boundaries: Union[Literal['ALL'], List[str]] = 'ALL'

    def get_doc_item(self, cases: List[Case], doc: Document, section_func: Union[Section, Subsection]=Section, case_by_case=False) -> None:
        with doc.create(section_func(self.__class__.__name__)):
            doc.append(f"this is {self.__class__.__name__}")


class Report(BaseModel):
    items: List[ReportItem]
    include_case_by_case: bool = True

    def _create_header_footer(self) -> PageStyle:
        header = PageStyle("header")
        # Header title and logo
        with header.create(Head("C")):
            header.append("Flow 360 Report")

        with header.create(Head("R")):
            header.append(NoEscape(
                r"\includegraphics[width=2cm]{"
                f"{os.path.join(os.path.dirname(__file__), 'img', 'flow360.png')}"
                "}"
            ))

        # Footer date and page number
        with header.create(Foot("C")):
            header.append(NoEscape(r"\thepage"))

        with header.create(Foot("L")):
            header.append(NoEscape(r"\today"))

        return header

    def create_pdf(self, filename:str, cases: list[Case]) -> None:
        # Create a new LaTeX document
        doc = Document()
        # Package info
        doc.packages.append(Package("float"))
        doc.packages.append(Package("placeins")) # For FloatBarrier
        doc.packages.append(Package("geometry", options=["a4paper", "margin=1in"]))

        # Preamble info
        doc.preamble.append(self._create_header_footer())
        doc.preamble.append(NoEscape(r'\setlength{\headheight}{20pt}'))
        doc.preamble.append(NoEscape(r'\addtolength{\topmargin}{-5pt}'))
        doc.change_document_style("header")
    
        # Iterate through all cases together
        for item in self.items:
            item.get_doc_item(cases, doc, case_by_case=False)

        # Iterate each case one at a time
        if self.include_case_by_case is True:
            with doc.create(Section('Appendix', numbering=False)):
                for case in cases:
                    with doc.create(Section(f"Case: {case.id}")):
                        for item in self.items:
                            item.get_doc_item([case], doc, Subsection, self.include_case_by_case)


        # Generate the PDF
        doc.generate_pdf(filename, clean_tex=False)


class Summary(ReportItem):
    text: str

    def get_doc_item(self, cases: List[Case], doc: Document, section_func: Union[Section, Subsection]=Section, case_by_case=False) -> None:
        section = section_func("Summary")
        doc.append(section)
        doc.append(f"{self.text}\n")
        Table(
            data_path=["name"],
            section_title=None,
            custom_headings=["Case Name"]
        ).get_doc_item(cases, doc, section_func, case_by_case)


class Inputs(ReportItem):
    """
    Inputs is a wrapper for a specific Table setup that details key inputs from the simulation
    """
    def get_doc_item(self, cases: List[Case], doc: Document, section_func: Union[Section, Subsection]=Section, case_by_case=False) -> None:
        Table(data_path=[
                "params_as_dict/version",
                "params_as_dict/timeStepping/orderOfAccuracy",
                "params_as_dict/surfaceOutput/outputFormat",
                "params_as_dict/userDefinedFields/0/name",
                "params_as_dict/userDefinedFields/2/name",
            ],  
            section_title="Inputs",
            custom_headings=[
                "Version",
                "Order of Accuracy",
                "Output Format",
                "CustomField 1",
                "CustomField 3",
            ]
        ).get_doc_item(cases, doc, section_func, case_by_case)


class Delta(BaseModel):
    data_path: str
    ref_case_id: str

    def calculate(self, case: Case, ref: Case) -> float:
        # Used when trying to do a Delta in a case_by_case or if ref ID is bad
        if ref is None:
            return "Ref not found."
        case_result = data_from_path(case, self.data_path)
        ref_result = data_from_path(ref, self.data_path)
        return sum([c_val - r_val for c_val, r_val in zip(ref_result, case_result)]) / len(ref_result)
    
    __str__ = lambda self: f"Delta {self.data_path.split('/')[-1]}"


class Expression(BaseModel):
    title: str
    expression: str

    def calculate(self, case: Case):
        return random.uniform(0.0, 1.0)

    __str__ = lambda self: self.title

class Tabulary(Tabular):

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


class Table(ReportItem):
    data_path: list[Union[str, Delta]]
    section_title: Union[str, None]
    custom_headings: Union[list[str], None] = None

    @model_validator(mode="after")
    def check_custom_heading_count(self) -> None:
        if self.custom_headings is not None:
            if len(self.data_path) != len(self.custom_headings):
                raise ValueError(f"Suppled `custom_headings` must be the same length as `data_path`: " 
                                 f"{len(self.custom_headings)} instead of {len(self.data_path)}")


    def get_doc_item(self, cases: List[Case], doc: Document, section_func: Union[Section, Subsection]=Section, case_by_case=False) -> None:
        # Only create a title if specified
        if self.section_title is not None:
            section = section_func(self.section_title)
            doc.append(section)

        # Getting tables to wrap is a pain - Tabulary seems the best approach
        with doc.create(Tabulary('|C' * (len(self.data_path) + 1) + '|', width=len(self.data_path) + 1)) as table:
            table.add_hline()

            # Manage column headings
            field_titles = [bold("Case No.")]
            if self.custom_headings is None:
                for path in self.data_path:
                    if isinstance(path, Delta):
                        field = path.__str__()
                    else:
                        field = path.split("/")[-1]

                    field_titles.append(bold(str(field)))
            else:
                field_titles.extend([bold(heading) for heading in self.custom_headings])

            table.add_row(field_titles)
            table.add_hline()
            
            # Build data rows
            for idx, case in enumerate(cases):
                row_list = [data_from_path(case, path, cases) for path in self.data_path]
                row_list.insert(0, str(idx + 1)) # Case numbers
                table.add_row(row_list)
                table.add_hline()

def _assemble_fig_rows(img_list, items_in_row):
    minipage_size = 0.98 / items_in_row # Smaller than 1 to avoid overflowing
    main_fig = Figure(position="h!")
        
        # Build list of indices to combine into rows
    indices = list(range(len(img_list)))
    idx_list = [indices[i:i + items_in_row] for i in range(0, len(indices), items_in_row)]
    for row_idx in idx_list:
        for idx in row_idx:
            sub_fig = SubFigure(position="t", width=NoEscape(fr"{minipage_size}\textwidth"))
            sub_fig.add_image(filename=img_list[idx], width=NoEscape(fr"\textwidth"))

                # Stop caption for single subfigures - happens when include_case_by_case
            if items_in_row != 1:
                sub_fig.add_caption(idx)

            main_fig.append(sub_fig)
                
            main_fig.append(NoEscape(r'\hfill'))

        main_fig.append(NoEscape(r'\\'))
    return main_fig

class Chart(ReportItem):
    section_title: Union[str, None]
    fig_name: str
    fig_size: float = 0.8 # Relates to fraction of the textwidth
    items_in_row: Union[int, None] = None
    select_case_ids: list[str] = None

    @model_validator(mode="after")
    def check_chart_args(self) -> None:
        if self.items_in_row is not None:
            if self.items_in_row == -1:
                return
            if self.items_in_row <= 1:
                raise ValueError(f"`Items_in_row` should be greater than 1. Use -1 to include all cases on a single row. Use `None` to disable the argument.")
        if self.items_in_row is not None and self.single_plot:
            raise ValueError(f"`Items_in_row` and `single_plot` cannot be used together.")

class Chart2D(Chart):
    data_path: list[Union[str, Delta]]
    background: Union[Literal["geometry"], None] = None
    single_plot: bool = False

    def _create_fig(self, x_data: list, y_data: list, x_lab: str, y_lab: str, save_name: str) -> None:
        """Create a simple matplotlib figure"""
        plt.plot(x_data, y_data)
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        if self.single_plot:
            plt.legend([val + 1 for val in range(len(x_data))])

        plt.savefig(save_name)
        if not self.single_plot:
            plt.close()

    def get_doc_item(self, cases: List[Case], doc: Document, section_func: Union[Section, Subsection]=Section, case_by_case=False) -> None:
        # Change items in row to be the number of cases if higher number is supplied
        if self.items_in_row is not None:
            if self.items_in_row > len(cases) or self.items_in_row == -1:
                self.items_in_row = len(cases)
        
        # Only create a title if specified
        if self.section_title is not None:
            section = section_func(self.section_title)
            doc.append(section)

        x_lab = self.data_path[0].split("/")[-1]
        y_lab = self.data_path[1].split("/")[-1]

        figure_list = []
        for case in cases:
            # Skip Cases the user hasn't selected
            if self.select_case_ids is not None:
                if case.id not in self.select_case_ids:
                    continue
            
            # Extract data from the Case
            x_data = data_from_path(case, self.data_path[0], cases)
            y_data = data_from_path(case, self.data_path[1], cases)

            # Create the figure using basic matplotlib
            cbc_str = "_cbc_" if case_by_case else ""
            save_name = self.fig_name + cbc_str + case.name + ".png"
            self._create_fig(x_data, y_data, x_lab, y_lab, save_name)

            # Allow for handling the figures later inside a subfig
            if self.items_in_row is not None:
                figure_list.append(save_name)

            elif self.single_plot:
                continue

            else:
                # Fig is added to doc later to facilitate method of creating single_plot
                fig = Figure(position="h!")
                fig.add_image(save_name, width=NoEscape(fr'{self.fig_size}\textwidth'))
                fig.add_caption(f'{x_lab} against {y_lab} for {case.name}.')
                figure_list.append(fig)

        if self.items_in_row is not None:
            main_fig = _assemble_fig_rows(figure_list, self.items_in_row)

            doc.append(main_fig)
            main_fig.add_caption(f'{x_lab} against {y_lab} for all cases.')

        elif self.single_plot:
            # Takes advantage of plot cached by matplotlib and that the last save_name is the full plot
            fig = Figure(position="h!")
            fig.add_image(save_name, width=NoEscape(fr'{self.fig_size}\textwidth'))
            fig.add_caption(f'{x_lab} against {y_lab} for all cases.')
            doc.append(fig)
        else:
            for fig in figure_list:
                doc.append(fig)

        # Stops figures floating away from their sections
        doc.append(NoEscape(r'\FloatBarrier'))
        doc.append(NoEscape(r'\clearpage'))

        # Clear the matplotlib cache to be certain figure won't appear
        plt.close()

class Chart3D(Chart):
    # field: str
    # camera: List[float]
    # limits: List[float]

    def get_doc_item(self, cases: List[Case], doc: Document, section_func: Union[Section, Subsection]=Section, case_by_case: bool=False):
        for case in cases:
            # This will use case.id to create requests
            # case.id
            file_name = "cp.png"
            img = requests.get('https://simcloud-public-1.s3.amazonaws.com/temp/post-processing-demo/cp.png', allow_redirects=True)
            open(file_name, "wb").write(img.content)

        # Change items in row to be the number of cases if higher number is supplied
        if self.items_in_row is not None:
            if self.items_in_row > len(cases) or self.items_in_row == -1:
                self.items_in_row = len(cases)
        
        # Only create a title if specified
        if self.section_title is not None:
            section = section_func(self.section_title)
            doc.append(section)

        img_list = []
        for case in cases:
            # Skip Cases the user hasn't selected
            if self.select_case_ids is not None:
                if case.id not in self.select_case_ids:
                    continue
            
            # Will change this once request image is built
            cbc_str = "cbc_" if case_by_case else ""
            file_name = "cp.png"
            img = requests.get('https://simcloud-public-1.s3.amazonaws.com/temp/post-processing-demo/cp.png', allow_redirects=True)
            open(cbc_str + file_name, "wb").write(img.content)
            img_list.append(file_name)
                      
        if self.items_in_row is not None:
            main_fig = _assemble_fig_rows(img_list, self.items_in_row)

            doc.append(main_fig)
            main_fig.add_caption(f'A cool test picture in a row.')

        else:
            for filename in img_list:
                fig = Figure(position="h!")
                fig.add_image(filename, width=NoEscape(fr'{self.fig_size}\textwidth'))
                fig.add_caption(f'A cool test picture.')
                doc.append(fig)

        # Stops figures floating away from their sections
        doc.append(NoEscape(r'\FloatBarrier'))
        doc.append(NoEscape(r'\clearpage'))

if __name__ == "__main__":
    # Answered Questions:
    # Where to start putting in styling - font, colours, etc.

    # New Questions
    # Delta is currently just a mean - what other ways might people want to express a delta
    # Is there any guide to test design for Flow360?
    # Where should report sit in the repo?
    # Any other details that should be included in Summary?
    # Is Expression needed? Parsing a user defined expression safely is pretty challenging - lot of recent work on this for Tidy3D

    # To Do
    # Tweak Delta to have more control over output

    a2_case = Case("9d86fc07-3d43-4c72-b324-7acad033edde")
    b2_case = Case("bd63add6-4093-4fca-95e8-f1ff754cfcd9")
    other_a2_case = Case("706d5fad-39ef-4782-8df5-c020723259bf")
    
    report = Report(
        items=[
            Summary(text='Analysis of a new feature'),
            Inputs(),

            Table(
                data_path=[
                    "params_as_dict/geometry/refArea", 
                    Delta(data_path="total_forces/CD", ref_case_id=a2_case.id),
                ],
                section_title="My Favourite Quantities",
            ),
            # Chart3D(
            #     section_title="Chart3D Testing",
            #     fig_size=0.5,
            #     fig_name="Unused"
            # ),
            # Chart3D(
            #     section_title="Chart3D Rows Testing",
            #     items_in_row=-1,
            #     fig_name="Unused"
            # ),
            Chart2D(
                data_path=["total_forces/pseudo_step", "total_forces/pseudo_step"],
                section_title="Sanity Check Step against Step",
                fig_name="step_fig",
                background=None,
            ),
            Chart2D(
                data_path=["total_forces/pseudo_step", "total_forces/CL"],
                section_title="Global Coefficient of Lift (just first Case)",
                fig_name="cl_fig",
                background=None,
                select_case_ids=[a2_case.id]
            ),
            Chart2D(
                data_path=["total_forces/pseudo_step", "total_forces/CFy"],
                section_title="Global Coefficient of Force in Y (subfigure and combined)",
                fig_name="cd_fig",
                background=None,
                items_in_row=-1
            ),
            Chart2D(
                data_path=["total_forces/pseudo_step", "total_forces/CFy"],
                section_title=None,
                fig_name="cd_comb_fig",
                background=None,
                single_plot=True
            ),
        ],
        include_case_by_case=True
    )
    report.create_pdf("test_report", [a2_case, b2_case, other_a2_case])
