"""
Report DOC representation
"""

import os
import posixpath

# this plugin is optional, thus pylatex is not required: TODO add handling of installation of pylatex
# pylint: disable=import-error
from pylatex import (
    Center,
    Document,
    Foot,
    HugeText,
    LargeText,
    MediumText,
    MiniPage,
    NewPage,
    NoEscape,
    Package,
    PageStyle,
    StandAloneGraphic,
)
from pylatex.utils import bold

from flow360.log import log
from flow360.plugins.report.utils import detect_latex_compiler, font_definition


class ReportDoc:
    """
    ReportDoc
    """

    def __init__(self, title, landscape=True) -> None:
        self.compiler, self.compiler_args = detect_latex_compiler()

        self.use_xelatex = self.compiler == "xelatex"
        if self.use_xelatex:
            log.info("Using 'xelatex' as the LaTeX compiler.")  # preferred for styling
        else:
            log.warning(f"Using '{self.compiler}' as the LaTeX compiler.")
            log.warning(
                "Warning: 'xelatex' is not available. Some font-related features may be disabled."
            )

        self._doc = Document(document_options=["10pt"])
        self._define_preamble(self._doc, landscape)
        self._create_custom_page_style(self._doc)
        self._title_page_style(self._doc)
        self._make_title(self._doc, title)
        self.doc.change_document_style("customstyle")

    @property
    def doc(self) -> Document:
        """
        Get current pylatex document

        Returns
        -------
        Document
            Current pylatex document
        """
        return self._doc

    def _define_preamble(self, doc, landscape):
        # Package info
        geometry_options = ["a4paper", "margin=0.5in", "bottom=0.7in", "includefoot"]
        if landscape:
            geometry_options.append("landscape")

        packages = [
            Package("float"),
            Package("caption"),
            Package("graphicx"),
            Package("placeins"),
            Package("xcolor", options="table"),
            Package("geometry", options=geometry_options),
            Package("tikz"),
            Package("colortbl"),
            Package("array"),
            NoEscape(r"\usepackage{eso-pic}"),
            Package("fancyhdr"),
        ]
        if self.use_xelatex:
            packages.append(Package("fontspec"))

        for package in packages:
            doc.packages.append(package)

        doc.preamble.append(NoEscape(r"\definecolor{customgray}{HTML}{E0E2E7}"))
        doc.preamble.append(NoEscape(r"\definecolor{labelgray}{HTML}{F4F5F7}"))

        doc.preamble.append(
            NoEscape(r"\DeclareCaptionLabelFormat{graybox}{\colorbox{labelgray}{}}")
        )
        doc.preamble.append(
            NoEscape(
                r"\captionsetup[figure]{position=bottom, font=large, labelformat=graybox, "
                r"labelsep=none, justification=raggedright, singlelinecheck=false}"
            )
        )
        doc.preamble.append(
            NoEscape(r"\captionsetup[subfigure]{labelformat=empty, justification=centering}")
        )

        if self.use_xelatex:
            doc.preamble.append(NoEscape(font_definition))

        self._table_settings(doc)
        self._background(doc)

    def _table_settings(self, doc):
        doc.preamble.append(NoEscape(r"\setlength{\tabcolsep}{12pt}"))
        doc.preamble.append(NoEscape(r"\renewcommand{\arraystretch}{2}"))
        doc.append(NoEscape(r"\arrayrulecolor{customgray}"))
        doc.preamble.append(
            NoEscape(
                r"""
            \newcolumntype{C}{>{\columncolor{white}}c}
        """
            )
        )

    def _background(self, doc):
        background_file = posixpath.join(
            os.path.dirname(__file__), "img", "background.pdf"
        ).replace("\\", "/")
        if not os.path.isfile(background_file):
            raise FileNotFoundError(f"Background image not found at path: {background_file}")

        background_latex = (
            r"""
        \AddToShipoutPictureBG{
            \begin{tikzpicture}[remember picture, overlay]
                \node[anchor=north east] at (current page.north east) {
                    \includegraphics[width=\textwidth]{"""
            + background_file
            + r"""}
                };
            \end{tikzpicture}
        }
        """
        )
        doc.preamble.append(NoEscape(background_latex))

    def _create_custom_page_style(self, doc) -> PageStyle:
        page_style = PageStyle("customstyle")
        padding = r"\vspace{10pt}"

        doc.preamble.append(NoEscape(r"\setlength{\footskip}{40pt}"))

        with page_style.create(Foot("C")) as footer:
            footer.append(NoEscape(r"\textcolor{customgray}{\rule{\textwidth}{0.5pt}}"))
            footer.append(NoEscape(r"\\"))
            footer.append(NoEscape(padding))
            with footer.create(
                MiniPage(width=NoEscape(r"\textwidth"), align="c")
            ) as footer_content:
                with footer_content.create(
                    MiniPage(width=NoEscape(r"0.33\textwidth"), align="l")
                ) as url:
                    url.append(NoEscape(r"\hspace{1em}"))
                    url.append(NoEscape(r"FLEXCOMPUTE.COM"))
                    url.append(NoEscape(r"\hspace{1em}"))

                with footer_content.create(
                    MiniPage(width=NoEscape(r"0.33\textwidth"), align="c")
                ) as logo1:
                    logo1.append(
                        StandAloneGraphic(
                            image_options="height=18pt",
                            filename=posixpath.join(
                                os.path.dirname(__file__), "img", "flow360_logo_grey.pdf"
                            ).replace("\\", "/"),
                        )
                    )

                with footer_content.create(
                    MiniPage(width=NoEscape(r"0.33\textwidth"), align="r")
                ) as page_num:
                    page_num.append(NoEscape(r"\thepage"))
            footer.append(NoEscape(padding))
        doc.preamble.append(page_style)

    def _title_page_style(self, doc) -> PageStyle:
        page_style = PageStyle("titlestyle")
        with page_style.create(Foot("C")) as footer:
            with footer.create(
                MiniPage(width=NoEscape(r"\textwidth"), align="c")
            ) as footer_content:
                footer_content.append(
                    StandAloneGraphic(
                        image_options="height=25pt",
                        filename=posixpath.join(
                            os.path.dirname(__file__), "img", "cover_logo.pdf"
                        ).replace("\\", "/"),
                    )
                )
        doc.preamble.append(page_style)

    def _make_title(self, doc, title: str = None):
        # pylint: disable=invalid-name
        NewLine = NoEscape(r"\\")  # pylatex NewLine() is causing problems with centering
        doc.append(NoEscape(r"\thispagestyle{titlestyle}"))

        doc.append(NoEscape(r"\vspace*{\fill}"))

        with doc.create(Center()):
            doc.append(HugeText(bold("Flow360 Report")))
            doc.append(NewLine)
            doc.append(NoEscape(r"\vspace{1cm}"))

            if title is not None:
                doc.append(LargeText(NoEscape(r"\textcolor{gray}{" + title + "}")))
                doc.append(NewLine)
            doc.append(NoEscape(r"\vspace{2cm}"))

            doc.append(MediumText(NoEscape(r"\textcolor{gray}{\today}")))
            doc.append(NewLine)

        doc.append(NoEscape(r"\vspace*{\fill}"))
        doc.append(NewPage())

    def generate_pdf(self, filename: str):
        """
        Generates PDF from ReportTemplate and Cases

        Parameters
        ----------
        filename : str
            Filename for PDF
        """
        pdf_ext = ".pdf"
        name_without_ext, file_ext = os.path.splitext(filename)
        if file_ext.lower() != pdf_ext:
            name_without_ext = filename
            filename = name_without_ext + pdf_ext

        self.doc.generate_pdf(
            name_without_ext,
            compiler=self.compiler,
            compiler_args=self.compiler_args,
            clean_tex=False,
        )

        log.info(f"PDF '{filename}' generated successfully using '{self.compiler}'.")
