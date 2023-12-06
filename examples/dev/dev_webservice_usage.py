import os
import re
import json
from pprint import pprint

from flow360.services import get_default_fork

# Webservice examples

rootdir = "../../tests/data/cases/web/"
regex = re.compile("(.*\.json$)")

for root, dirs, files in os.walk(rootdir):
    for file in files:
        if regex.match(file):
            f = open(os.path.join(rootdir, file))
            legacy = json.load(f)
            get_default_fork(legacy, legacy=True)
            print(f"Forked {file}")
