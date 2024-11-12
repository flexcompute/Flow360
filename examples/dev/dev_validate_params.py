import os
import re

from flow360.component.v1.flow360_params import Flow360ParamsLegacy

rootdir = "../../tests/data/cases/"
regex = re.compile(r"(case_.*\.json$)")

for root, dirs, files in os.walk(rootdir):
    for file in files:
        if regex.match(file):
            try:
                print(f"Now validating {file}")
                validated = Flow360ParamsLegacy(os.path.join(rootdir, file))
            except Exception as error:
                print(error)
