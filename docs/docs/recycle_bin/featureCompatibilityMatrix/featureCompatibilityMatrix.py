import os

import pandas as pd

here = os.path.abspath(os.path.dirname(__file__))

CSVfile = os.path.join(here, "Compatibility_Matrix.csv")
rstFilename = "featureCompatibilityMatrix.rst"

LEGEND = "Legend"
FOOTNOTE = "Footnote list"

htmlStyle = """
    <style>
        table {
            border: 1px solid #ccc;
            border-collapse: collapse;
        }
        .compatibilitytable {
            font-size:10px;
        }
        th, td {
            border: 1px solid #ccc;
            font-weight: normal;
        }

        .textcenter {
            text-align: center;
        }

        /*vertical text*/
        .v span {
            /*writing-mode: tb-rl;*/
            writing-mode: vertical-rl;
            -webkit-writing-mode: vertical-rl;
            /*-ms-writing-mode: vertical-rl;*/
            transform: rotate(-180deg);
            width: 0.7rem;
            height: 15rem;
            text-align: left;
        }
        .vline span{
            height: 15rem;
        }
        /*highlight*/
        .hl {
            background: #e16173;
            color: #fff;
            font-weight: bold;
            border: none;
        }
    </style>
"""

categories = ["Models", "Turbulence Models", "Meshing", "Boundary conditions"]


def headerEntry(content, highlight=False):
    if highlight:
        return f'<th class="v hl"><span>{content}</span></th>'
    else:
        return f'<th class="v"><span>{content}</span></th>'


def getHeader(df):
    legend = """
        <td>
            <div>Legend</div>
            <div>&radic; = Supported and Well Tested</div>
            <div>O = Supported</div>
            <div>X = Not Supported</div>
        </td>
"""

    header = f"""
    <table class="compatibilitytable">
    <tbody>             
        <tr>{legend}"""

    for key in df.keys()[1:]:
        header += f"""
        {headerEntry(key, key in categories)}"""
    header += """
    </tr>"""

    return header


def getFooter():
    return """
    </tbody>
    </table>

"""


def getBottomHeader(row):
    bottomHeader = """
    <tr>"""

    for key, value in row.items():
        bottomHeader += f"""
        {headerEntry(value, key in categories)}"""
    bottomHeader += """
    </tr>"""
    return bottomHeader


def getRowItem(content, center=False, highlight=False):
    if highlight:
        return f'<td class="hl"><span>{content}</span></td>'
    if center:
        return f'<td class="textcenter">{content}</td>'
    else:
        return f"<td>{content}</td>"


def getRow(dfrow):

    if any([dfrow[LEGEND].startswith(cat) for cat in categories]):
        print(dfrow[LEGEND], "is category")
        Ncolumns = len(dfrow.keys())
        return f"""
    <tr>
        <td class="hl" colspan="{Ncolumns}">{dfrow[LEGEND]}</td>
    </tr>"""

    row = """
    <tr>"""
    for key, value in dfrow.items():
        entry = getRowItem(
            value, center=(key != LEGEND and value != ""), highlight=(key in categories)
        )
        row += f"""
        {entry}"""
    row += """
    </tr>"""
    return row


def getTable(csvFile, skipEmptyRows=True):
    df = pd.read_csv(csvFile, dtype=str, na_filter=False, skip_blank_lines=True)
    df = df.drop(df.columns[0], axis=1)
    df = df.rename(columns={df.columns[0]: LEGEND})
    df = df.drop([key for key in df.keys() if key.startswith("Unnamed:")], axis=1)

    if skipEmptyRows:
        df = df.drop(
            [i for i, row in df.iterrows() if not any(row.astype(bool).values)]
        )

    print(df)
    print(df.keys())

    with open(os.path.join(here, rstFilename), "w", encoding="utf-8") as fh:
        fh.write(
            """
..
   _THIS FILE IS GENERATED AUTOMATICALLY

Feature Compatibility Matrix
****************************        

"""
        )
        fh.write(".. raw:: html\n")
        fh.write(htmlStyle)
        fh.write(getHeader(df))
        for i, row in df.iterrows():
            if row[LEGEND] == FOOTNOTE:
                break
            if row[LEGEND] == "":
                fh.write(getBottomHeader(row))
            else:
                fh.write(getRow(row))
        fh.write(getFooter())

        # Footnote list:
        skip = True
        fh.write("|\n")
        for i, row in df.iterrows():
            if row[LEGEND] != FOOTNOTE and skip:
                continue
            if row[LEGEND] == FOOTNOTE:
                fh.write(f"\n**{row[LEGEND]}**\n\n")
            else:
                fh.write(f"| {row[LEGEND]}\n")
            skip = False


getTable(CSVfile)
