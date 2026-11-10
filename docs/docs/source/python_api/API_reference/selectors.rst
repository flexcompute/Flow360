.. _python_api_selectors:

*********
Selectors
*********
.. currentmodule:: flow360

A selector picks entities by rule instead of by listing them. It names the kind of entity it
targets and carries predicates on the entity name; the rule is stored with the simulation and
re-applied on the cloud, so it keeps working when a geometry is re-uploaded with the same naming.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   SurfaceSelector
   EdgeSelector
   VolumeSelector
   BodyGroupSelector

.. code-block:: python

    wheels = fl.SurfaceSelector(name="wheels").match("*rim*")
    body = fl.SurfaceSelector(name="body").match("*body*").not_match("*_mirrored")
    fixed = fl.SurfaceSelector(name="fixed").any_of(["fuselage", "tail"])

``match`` and ``not_match`` take a pattern; ``any_of`` and ``not_any_of`` take exact names.
Predicates chained on one selector are intersected by default; pass ``logic="OR"`` to take
their union instead.

.. _python_api_glob_syntax:

Pattern syntax
==============

A pattern must match the **whole** entity name, **case-sensitively**. Four constructs have a
special meaning; every other character stands for itself.

.. list-table::
   :header-rows: 1
   :widths: 12 40 48

   * - Pattern
     - Matches
     - Example
   * - ``*``
     - any run of characters, including none
     - ``wing*`` matches ``wing``, ``wing_1`` and ``wing/flap``
   * - ``?``
     - exactly one character
     - ``face_?`` matches ``face_1`` but not ``face_12``
   * - ``[seq]``
     - one character out of ``seq``; ranges such as ``[a-z]`` are allowed
     - ``face_[0-9]`` matches ``face_7``; ``[wW]ing*`` matches ``wing_1`` and ``Wing_1``
   * - ``[!seq]``
     - one character not in ``seq``
     - ``face_[!0-9]`` matches ``face_a`` but not ``face_1``

Consequences worth knowing:

- ``*`` also matches the empty string: ``wing*`` matches ``wing`` itself, and ``*`` matches
  every entity.
- ``/`` is an ordinary character. ``*`` crosses it, and ``zone/bc*`` is written literally.
- ``{``, ``}``, ``\``, ``(``, ``)``, ``^`` and an unclosed ``[`` are ordinary characters. A
  backslash does not escape anything, and ``^`` inside brackets is a member of the set, not a
  negation -- negate with ``!``.
- To match a literal ``*``, ``?`` or ``[``, put it in brackets: ``literal[*]star``,
  ``face[?]``, ``face[[]1]``. A ``]`` outside brackets needs nothing.

The dialect is exactly the Python standard library's :func:`fnmatch.fnmatchcase`, whose
documentation is the normative reference (not :func:`fnmatch.fnmatch`, which folds case on
some platforms). The same rules apply wherever Flow360 accepts a name pattern, in the Python
client and in the web interface.

Case-insensitive matching
-------------------------

There is no flag. Either spell the pattern with bracket sets, ``[wW]ing*``, or give one
selector a predicate per spelling and take the union:

.. code-block:: python

    wing = fl.SurfaceSelector(name="wing", logic="OR").match("wing*").match("Wing*").match("WING*")

Rejected spellings
------------------

Some glob engines give a meaning to brace alternation, extended-glob groups and POSIX
character classes. Flow360 does not. Rather than let such a pattern silently degrade to literal
characters and match nothing, validation rejects it with a message naming the replacement:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Rejected
     - Write instead
   * - ``{wing,flap}*``
     - one predicate per alternative under ``logic="OR"``: ``.match("wing*").match("flap*")``
   * - ``!(internal)*``
     - ``.not_match("internal*")``
   * - ``@(wing|flap)_1``
     - one predicate per alternative, as for braces
   * - ``+(ab)``
     - no equivalent; repetition is not part of the dialect
   * - ``[[:digit:]]*``, ``[[:alpha:]]*``
     - ``[0-9]*``, ``[a-zA-Z]*``

``*(`` and ``?(`` are not rejected: they read as a wildcard followed by a literal parenthesis,
so ``Body*(1)`` matches ``Body_(1)``.
