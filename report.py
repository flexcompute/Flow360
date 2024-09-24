import random
import matplotlib.pyplot as plt

from pydantic.v1 import BaseModel, Field
from typing import List, Literal, Union

from pylatex import Document, Section, Subsection, Tabular, Figure, NoEscape, Head, PageStyle, MiniPage
from pylatex.utils import bold

import flow360 as fl
from flow360 import Case

class ReportItem(BaseModel):

    boundaries: Union[Literal['ALL'], List[str]] = 'ALL'

    def get_doc_node(self, cases: List[Case], doc: Document):
        with doc.create(Section(self.__class__.__name__)):
            doc.append(f"this is {self.__class__.__name__}")


class Report(BaseModel):
    source: List[Case]
    items: List[ReportItem]
    include_case_by_case: bool = True

    # Would like to remove this
    class Config:
        arbitrary_types_allowed = True

    def to_file(self, filename: str):
        # print('this is a report')
        # print(self.model_dump_json(indent=2))

        self.create_pdf(filename)

    def _create_header(self):
        header = PageStyle("header")           
        with header.create(Head("C")):
            header.append("Flow 360 Report")

        with header.create(Head("R")):
            header.append(NoEscape(r'\includegraphics[width=2cm]{/home/matt/Documents/Flexcompute/flow360/Flow360/flow360.png}'))

        return header

    def create_pdf(self, filename):
        # Create a new LaTeX document
        doc = Document()
        doc.packages.append(NoEscape(r'\usepackage{float}'))
        doc.packages.append(NoEscape(r'\usepackage{placeins}')) # For FloatBarrier
        doc.packages.append(NoEscape(r'\usepackage[a4paper, margin=1.5in]{geometry}'))

        doc.preamble.append(self._create_header())
        doc.preamble.append(NoEscape(r'\setlength{\headheight}{20pt}'))
        doc.preamble.append(NoEscape(r'\addtolength{\topmargin}{-5pt}'))
        doc.change_document_style("header")
    
        for item in self.items:
            item.get_doc_node(self.source, doc)


        if self.include_case_by_case is True:

            with doc.create(Section('Appendix')):
                for case in self.source:
                    with doc.create(Subsection(f"Case: {case.id}")):
                        for item in self.items:
                            item.get_doc_node([case], doc)

        # Generate the PDF
        doc.generate_pdf(filename, clean_tex=False)




class Summary(ReportItem):
    text: str

    def get_doc_node(self, cases: List[Case], doc: Document):
        with doc.create(Section(self.__class__.__name__)):
            doc.append(f"{self.text}\n")
            for case in cases:
                doc.append(f"id={case.id}, name: {case.name}\n")



class Inputs(ReportItem):
    def get_doc_node(self, cases: List[Case], doc: Document):
        with doc.create(Section(self.__class__.__name__)):
            doc.append(f"a table displaying all solver inputs including: models, operating conditions, and references\n")



class Delta(BaseModel):
    field: str

    def calculate(self, case: Case, ref: Case):
        return ref.get_result(self.field, ref) - case.get_result(self.field, ref)

    __str__ = lambda self: f"Delta {self.field}"

class Expression(BaseModel):
    title: str
    expression: str

    def calculate(self, case: Case):
        return random.uniform(0.0, 1.0)


    __str__ = lambda self: self.title

class Table(ReportItem):
    data_loc: list[str, str]
    field: list[str, str]
    section_title: str

    # columns: List[Union[str, Delta, Expression]]


    def get_doc_node(self, cases: List[Case], doc: Document):
        with doc.create(Section(self.__class__.__name__)):
            with doc.create(Tabular('|c' * len(self.columns) + '|')) as table:
                table.add_hline()
                table.add_row([bold(str(title)) for title in self.columns])
                table.add_hline()
                for case in cases:
                    table.add_row([case.get_result(item, cases[0]) for item in self.columns])
                    table.add_hline()


class Chart2D(ReportItem):
    data_loc: list[str, str]
    field: list[str, str]
    section_title: str
    fig_name: str
    background: Union[Literal["geometry"], None]
    select_cases: list[Case] = None
    fig_size: float = 0.8 # Relates to fraction of the textwidth
    fit_horizontal: bool = False
    single_plot: bool = False

    # Would like to remove this
    class Config:
        arbitrary_types_allowed = True

    def _create_fig(self, x_data, y_data, x_lab, y_lab, save_name):
        plt.plot(x_data, y_data)
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        plt.savefig(save_name)
        if not self.single_plot:
            plt.close()

    def get_doc_node(self, cases: List[Case], doc: Document):
        with doc.create(Section(self.section_title)):
            x_lab = self.field[0]
            y_lab = self.field[1]

            if self.fit_horizontal:
                main_fig = Figure(position="h!")
                sub_fig_width = 0.95 / len(cases) # Smaller than 1 to avoid overflowing

            figure_list = []
            for case in cases:
                # Skip Cases the user hasn't selected
                if self.select_cases is not None:
                    if case not in self.select_cases:
                        continue
                
                # Extract data from the Case
                x_data = getattr(case.results, self.data_loc[0]).values[x_lab]
                y_data = getattr(case.results, self.data_loc[1]).values[y_lab]

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


class Chart3D(ReportItem):
    field: str
    camera: List[float]
    limits: List[float]

    def _get_doc_node(self, cases: List[Case], doc: Document):
        with doc.create(Figure(position='H')) as fig:
            for case in cases:
                fig.add_image(case.results.get_chart3D(self.field, self.camera, self.limits), width=NoEscape(r'0.5\textwidth'))
                fig.add_caption(f'Image case: {case.name}, for boundaries: {self.boundaries}')

    def get_doc_node(self, cases: List[Case], doc: Document, create_title: bool = True):
        if create_title:
            with doc.create(Section(self.__class__.__name__)):
                self._get_doc_node(cases, doc)
        else:
            self._get_doc_node(cases, doc)



class Group(ReportItem):
    title: str
    layout: Literal["vertical", "horizontal"]
    item: Union[Chart2D, Chart3D]

    def get_doc_node(self, cases: List[Case], doc: Document):
        with doc.create(Section(f"{self.__class__.__name__}: {self.title}")):
            for case in cases:
                self.item.get_doc_node([case], doc, create_title=False)

if __name__ == "__main__":
    # To Learn:
    # What are the boundaries refering to? How can I retreave this info from a case
    # Pydantic in Flow360 - how to remove Config update for cases
    # Anyway to keep csvs downloaded?
    # Where to start putting in styling - font, colours, etc.
    # Lost as to what Group is doing. What about changing Chart2D to control fig size and a TitledSummary for other text?
    # # Is there interest in plotting everything on a single plot?
    # # How does include_case_by_case operate with Group?
    # How to get geometry slice and values at different distances?
    # What needs to be included in Inputs? Though about tabulating params_as_dict but it's probably too much

    a2_case = fl.Case("9d86fc07-3d43-4c72-b324-7acad033edde")
    b2_case = fl.Case("bd63add6-4093-4fca-95e8-f1ff754cfcd9")
    
    report = Report(
        source=[a2_case, b2_case],
        items=[
            Summary(text='Analysis of a new feature'),
            Inputs(),
            Chart2D(
               data_loc=["total_forces", "total_forces"], 
               field=["pseudo_step", "pseudo_step"],
               section_title="Sanity Check Step against Step",
               fig_name="step_fig",
               background=None,
            ),
            Chart2D(
               data_loc=["total_forces", "total_forces"], 
               field=["pseudo_step", "CL"],
               section_title="Global Coefficient of Lift (just first Case)",
               fig_name="cl_fig",
               background=None,
               select_cases=[a2_case]
            ),
            Chart2D(
               data_loc=["total_forces", "total_forces"], 
               field=["pseudo_step", "CD"],
               section_title="Global Coefficient of Drag (horizontal figure)",
               fig_name="cd_fig",
               background=None,
               fit_horizontal=True
            ),
            Chart2D(
               data_loc=["total_forces", "total_forces"], 
               field=["pseudo_step", "CFy"],
               section_title="Coefficient of Force in Y Plane (combined figure)",
               fig_name="cfy_fig",
               background=None,
               single_plot=True
            ),
        ],
        include_case_by_case=False
    )
    report.to_file("test_report")
