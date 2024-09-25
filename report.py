import random
import matplotlib.pyplot as plt

from pydantic import BaseModel, model_validator
from typing import List, Literal, Union

from pylatex import Document, Section, Subsection, Tabular, Figure, NoEscape, Head, PageStyle, MiniPage
from pylatex.utils import bold

from flow360 import Case

def get_case_from_id(id: str, cases: list[Case]):
    if len(cases) == 0:
        raise ValueError("No cases provided for `get_case_from_id`.")
    for case in cases:
        if case.id == id:
            return case

def data_from_path(case: Case, path:str, cases: list[Case]=[]):
    # Handle Delta values
    if isinstance(path, Delta):
        ref_case = get_case_from_id(path.ref_case_id, cases)
        return path.calculate(case, ref_case)

    # Split path into components
    path_components = path.split("/")

    def _search_path(case: Case, component: str):
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
            if isinstance(case, (int, str)):
                return case
            elif "value" in case:
                return case["value"]
            else:
                return case
        except TypeError: 
            pass

        # Check if case is a list and interpret component as an int index
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

    for component in path_components:
        case = _search_path(case, component)

    return case

class ReportItem(BaseModel):

    boundaries: Union[Literal['ALL'], List[str]] = 'ALL'

    def get_doc_node(self, cases: List[Case], doc: Document, section_func: Union[Section, Subsection]=Section):
        with doc.create(section_func(self.__class__.__name__)):
            doc.append(f"this is {self.__class__.__name__}")


class Report(BaseModel):
    items: List[ReportItem]
    include_case_by_case: bool = True

    def _create_header(self):
        header = PageStyle("header")           
        with header.create(Head("C")):
            header.append("Flow 360 Report")

        with header.create(Head("R")):
            header.append(NoEscape(r'\includegraphics[width=2cm]{/home/matt/Documents/Flexcompute/flow360/Flow360/flow360.png}'))

        return header

    def create_pdf(self, filename:str, cases: list[Case]):
        # Create a new LaTeX document
        doc = Document()
        doc.packages.append(NoEscape(r'\usepackage{float}'))
        doc.packages.append(NoEscape(r'\usepackage{graphicx}'))
        doc.packages.append(NoEscape(r'\usepackage{placeins}')) # For FloatBarrier
        doc.packages.append(NoEscape(r'\usepackage[a4paper, margin=1.5in]{geometry}'))

        doc.preamble.append(self._create_header())
        doc.preamble.append(NoEscape(r'\setlength{\headheight}{20pt}'))
        doc.preamble.append(NoEscape(r'\addtolength{\topmargin}{-5pt}'))
        doc.change_document_style("header")
    
        for item in self.items:
            item.get_doc_node(cases, doc)

        if self.include_case_by_case is True:

            with doc.create(Section('Appendix', numbering=False)):
                for case in cases:
                    with doc.create(Section(f"Case: {case.id}")):
                        for item in self.items:
                            item.get_doc_node([case], doc, Subsection)

        # Generate the PDF
        doc.generate_pdf(filename, clean_tex=False)


class Summary(ReportItem):
    text: str

    def get_doc_node(self, cases: List[Case], doc: Document, section_func: Union[Section, Subsection]=Section):
        with doc.create(section_func("Summary")):
            doc.append(f"{self.text}\n")
            for case in cases:
                doc.append(f"id={case.id}, name: {case.name}\n")


class Inputs(ReportItem):
    """
    Inputs is a wrapper for a specific Table setup that details key inputs from the simulation
    """
    def get_doc_node(self, cases: List[Case], doc: Document, section_func: Union[Section, Subsection]=Section):
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
        ).get_doc_node(cases, doc, section_func)


class Delta(BaseModel):
    data_path: str
    ref_case_id: str

    def calculate(self, case: Case, ref: Case):
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


class Table(ReportItem):
    data_path: list[Union[str, Delta]]
    section_title: str
    custom_headings: list[str] = None

    @model_validator(mode="after")
    def check_custom_heading_count(self):
        if self.custom_headings is not None:
            if len(self.data_path) != len(self.custom_headings):
                raise ValueError(f"Suppled `custom_headings` must be the same length as `data_path`: " 
                                 f"{len(self.custom_headings)} instead of {len(self.data_path)}")


    def get_doc_node(self, cases: List[Case], doc: Document, section_func: Union[Section, Subsection]=Section):
        
        with doc.create(section_func(self.section_title)):
            with doc.create(Tabular('|c' * (len(self.data_path) + 1) + '|')) as table:
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
                    field_titles.extend(self.custom_headings)

                table.add_row(field_titles)
                table.add_hline()
                for idx, case in enumerate(cases):
                    row_list = [data_from_path(case, path, cases) for path in self.data_path]
                    # row_list = [getattr(case.results, loc).values[field] for loc, field in zip(self.data_loc, self.field)]
                    row_list.insert(0, str(idx + 1))
                    table.add_row(row_list)
                    table.add_hline()


class Chart2D(ReportItem):
    data_path: list[Union[str, Delta]]
    section_title: str
    fig_name: str
    background: Union[Literal["geometry"], None]
    select_case_ids: list[str] = None
    fig_size: float = 0.8 # Relates to fraction of the textwidth
    fit_horizontal: bool = False
    single_plot: bool = False

    def _create_fig(self, x_data: list, y_data: list, x_lab: str, y_lab: str, save_name: str):
        plt.plot(x_data, y_data)
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        plt.savefig(save_name)
        if not self.single_plot:
            plt.close()

    def get_doc_node(self, cases: List[Case], doc: Document, section_func: Union[Section, Subsection]=Section):
        with doc.create(section_func(self.section_title)):
            x_lab = self.data_path[0].split("/")[-1]
            y_lab = self.data_path[1].split("/")[-1]

            if self.fit_horizontal:
                main_fig = Figure(position="h!")
                sub_fig_width = 0.95 / len(cases) # Smaller than 1 to avoid overflowing

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
                save_name = self.fig_name + case.name + ".png"
                self._create_fig(x_data, y_data, x_lab, y_lab, save_name)

                # Combine smaller figures inside MiniPages into a single figure
                if self.fit_horizontal:
                    mini_page = MiniPage(width=NoEscape(fr"{sub_fig_width}\textwidth"))
                    mini_page.append(NoEscape(r"\includegraphics[width=\textwidth]{" + save_name + "}"))
                    main_fig.append(mini_page)
                
                elif self.single_plot:
                    continue

                else:
                    fig = Figure(position="h!")
                    fig.add_image(save_name, width=NoEscape(fr'{self.fig_size}\textwidth'))
                    fig.add_caption(f'{x_lab} against {y_lab} for {case.name}.')
                    figure_list.append(fig)

            if self.fit_horizontal:
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

        # Clear the matplotlib cache to be certain figure won't appear
        plt.close()

class Chart3D(ReportItem):
    field: str
    camera: List[float]
    limits: List[float]

    def _get_doc_node(self, cases: List[Case], doc: Document, section_func: Union[Section, Subsection]=Section):
        with doc.create(Figure(position='H')) as fig:
            for case in cases:
                fig.add_image(case.results.get_chart3D(self.field, self.camera, self.limits), width=NoEscape(r'0.5\textwidth'))
                fig.add_caption(f'Image case: {case.name}, for boundaries: {self.boundaries}')

    def get_doc_node(self, cases: List[Case], doc: Document, section_func: Union[Section, Subsection]=Section, create_title: bool = True):
        if create_title:
            with doc.create(section_func(self.__class__.__name__)):
                self._get_doc_node(cases, doc, section_func)
        else:
            self._get_doc_node(cases, doc, section_func)


class Group(ReportItem):
    title: str
    layout: Literal["vertical", "horizontal"]
    item: Union[Chart2D, Chart3D]

    def get_doc_node(self, cases: List[Case], doc: Document):
        with doc.create(Section(f"{self.__class__.__name__}: {self.title}")):
            for case in cases:
                self.item.get_doc_node([case], doc, create_title=False)

if __name__ == "__main__":
    # Answered Questions:
    # Where to start putting in styling - font, colours, etc.
    # Lost as to what Group is doing. What about changing Chart2D to control fig size and a TitledSummary for other text?
    # # How does include_case_by_case operate with Group?

    # New Questions
    # Delta is currently just a mean - what other ways might people want to express a delta
    # Is there any guide to test design for Flow360?
    # Any other details that should be included in Summary? 
    # # Need to think how to format case id and name as these are often long strings
    # What's the usecase for Chart3D? Is it not just displaying a png?
    # Discuss Expression: Parsing an expression safely is pretty challenging - lot of recent work on this for Tidy3D
    # Group may be redundant. What if Chart2D can tesselate any X by Y arrangement with something like a draw_rows=2 arg?

    # To Do
    # Need to improve data_loc/field system. Need something that makes it easy to get into nested data
    # Expressions and Deltas in "field" will need to have some name attached to them
    # # This will then need to be pulled out, instead of assuming the input is a str
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
                    Delta(data_path="total_forces/CD", ref_case_id=a2_case.id)
                ],
                section_title="My Favourite Quantities",
            ),

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
                data_path=["total_forces/pseudo_step", "total_forces/CD"],
                section_title="Global Coefficient of Drag (horizontal figure)",
                fig_name="cd_fig",
                background=None,
                fit_horizontal=True
            ),
            Chart2D(
                data_path=["total_forces/pseudo_step", "total_forces/CFy"],
                section_title="Coefficient of Force in Y Plane (combined figure)",
                fig_name="cfy_fig",
                background=None,
                single_plot=True
            ),
        ],
        include_case_by_case=True
    )
    report.create_pdf("test_report", [a2_case, b2_case, other_a2_case])
