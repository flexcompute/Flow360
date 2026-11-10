"""The one glob dialect for every name pattern in flow360-schema.

The dialect is the standard library's ``fnmatch`` as evaluated by ``fnmatch.fnmatchcase``::

    *       any run of characters, including none
    ?       exactly one character
    [seq]   one character out of seq; ranges such as [a-z] are allowed
    [!seq]  one character not in seq

Matching is case-sensitive and covers the whole string. ``/`` is an ordinary character, and so
are ``{``, ``}``, ``\\``, ``(``, ``)``, ``^`` and an unclosed ``[``. A literal ``*``, ``?`` or ``[``
is written inside brackets: ``[*]``, ``[?]``, ``[[]``.

Normative reference: https://docs.python.org/3/library/fnmatch.html -- specifically the
``fnmatchcase`` variant; plain ``fnmatch.fnmatch`` folds case on some platforms.

Five spellings that other glob engines give a meaning are rejected instead of silently degrading
to literal characters: brace alternation ``{a,b}``, the extended-glob groups ``!(...)``, ``@(...)``
and ``+(...)``, and POSIX character classes ``[[:alpha:]]``.
"""

import fnmatch
import re
from functools import lru_cache

DIALECT_SUMMARY = (
    "Supported glob syntax: '*' any run of characters (including none), '?' exactly one character, "
    "'[seq]' one character out of seq, '[!seq]' one character not in seq; "
    "case-sensitive, whole-string match; every other character is literal."
)

_EXTENDED_GLOB_HINT = (
    "Extended-glob groups such as '!(a)', '@(a|b)' and '+(a)' are not supported; "
    "combine plain globs with the selector's not/or operators instead."
)

_REJECTED_SPELLINGS: tuple[tuple[str, str], ...] = (
    ("{", "Brace alternation such as '{a,b}' is not supported; write one pattern per alternative."),
    ("!(", _EXTENDED_GLOB_HINT),
    ("@(", _EXTENDED_GLOB_HINT),
    ("+(", _EXTENDED_GLOB_HINT),
    (
        "[[:",
        "POSIX character classes such as '[[:alpha:]]' are not supported; "
        "write the range out instead, e.g. '[a-zA-Z]' or '[0-9]'.",
    ),
)


@lru_cache(maxsize=2048)
def compile_glob(pattern: str) -> re.Pattern[str]:
    """Compile ``pattern`` to a regex whose ``match`` is a whole-string, case-sensitive glob test.

    Raises ``ValueError`` for the rejected spellings listed in the module docstring.
    """
    for spelling, hint in _REJECTED_SPELLINGS:
        if spelling in pattern:
            raise ValueError(f"Glob pattern {pattern!r} contains {spelling!r}. {hint} {DIALECT_SUMMARY}")
    return re.compile(fnmatch.translate(pattern))


def glob_match(value: str, pattern: str) -> bool:
    """Whether ``value`` as a whole matches the glob ``pattern``."""
    return compile_glob(pattern).match(value) is not None
