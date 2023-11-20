import os
import re

import flow360 as fl

rootdir = "../../tests/data/cases/"
regex = re.compile("(case_.*\.json$)")


for root, dirs, files in os.walk(rootdir):
    for file in files:
        if regex.match(file):
            try:
                print(f"Now validating {file}")
                validated = fl.Flow360Params(os.path.join(rootdir, file))
            except Exception as error:
                print(error)
