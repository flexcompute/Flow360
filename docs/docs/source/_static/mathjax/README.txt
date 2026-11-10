Vendored MathJax
================

Version: 3.2.2
Source:  https://registry.npmjs.org/mathjax/-/mathjax-3.2.2.tgz
License: Apache-2.0 (see the LICENSE field of the npm package)

Why this is here
----------------

Sphinx's mathjax extension defaults to loading MathJax from
cdn.jsdelivr.net. Any reader whose network or browser cannot reach that
host sees every equation on the site as raw LaTeX markup, for example
\(q_\infty \, S_{ref}\) instead of a typeset formula. Serving MathJax
from our own site removes that dependency.

conf.py points at this copy with:

    mathjax_path = "mathjax/tex-mml-chtml.js"

A relative mathjax_path is resolved against _static, so the published
URL is _static/mathjax/tex-mml-chtml.js.

What is included
----------------

Only the two pieces the HTML build needs, not the whole es5 tree:

    tex-mml-chtml.js                     the combined TeX input plus
                                         CommonHTML output bundle
    output/chtml/fonts/woff-v2/*.woff    the 23 web fonts that the
                                         CommonHTML output loads

MathJax works out the location of the font directory from the URL of the
script that loaded it, so the output/chtml/fonts/woff-v2 path below this
directory has to be preserved. Moving or flattening it leaves the
equations typeset in fallback fonts.

How to update
-------------

    curl -sL -o mathjax.tgz https://registry.npmjs.org/mathjax/-/mathjax-<version>.tgz
    tar xzf mathjax.tgz --wildcards \
        'package/es5/tex-mml-chtml.js' \
        'package/es5/output/chtml/fonts/woff-v2/*'

Copy the two paths over this directory, update the version recorded
above, and rebuild. Check a page carrying equations in a browser
afterwards, because a missing font directory degrades quietly rather
than failing the build.
