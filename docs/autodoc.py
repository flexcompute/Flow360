
import sphinx.ext.autodoc
from sphinx.cmd.build import main

import os


def abspath(fname):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), fname))


rst = []


def add_line(self, line, source, *lineno):
    """Append one line of generated reST to the output."""
    rst.append(self.indent + line)
    self.directive.result.append(self.indent + line, source, *lineno)


sphinx.ext.autodoc.Documenter.add_line = add_line


output = abspath('_build/html/')
sourceDir = abspath('.')
doctrees = abspath('_build/doctrees')
main(['-a', sourceDir, output, '-b', 'html'])

with open('flow360.rst', 'w') as f:
    print("flow360 python client 2 module", file=f)
    print("********************", file=f)
    print("", file=f)

    for line in rst:
        print(line, file=f)
