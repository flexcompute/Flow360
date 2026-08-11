"""Unified Sphinx ClassDocumenter for all Flow360 classes.

Produces structured Attributes / Properties / Additional Constructors / Methods
sections for both pydantic models (Flow360BaseModel subclasses) and non-pydantic
classes (Flow360Resource, AssetBase subclasses, etc.).

Uses ``.. automethod::`` directives so that Napoleon, source links, and
signatures are handled automatically by Sphinx.
"""

from __future__ import annotations

import re
from typing import Any

from flow360_doc_utils import (
    collect_attributes,
    collect_methods,
    collect_properties,
    format_default_value,
    process_docstring_napoleon,
)
from flow360_type_resolver import format_field_annotation, format_type_annotation
from pydantic_core import PydanticUndefined
from sphinx.ext.autodoc import ClassDocumenter, Documenter

# A line consisting solely of repeated section/transition punctuation, e.g.
# ``====`` or ``----``. When such a line trails a docstring (preceded by a
# blank line rather than a title) docutils treats it as a transition.
_TRANSITION_RE = re.compile(r"^[=\-`:.'\"~^_*+#]{4,}$")


class Flow360Documenter(ClassDocumenter):
    """Unified documenter for ``.. autoflow360class::`` directives.

    Handles both Flow360BaseModel (pydantic) subclasses and non-pydantic
    Flow360 classes, producing identical documentation structure:
    Attributes → Properties → Additional Constructors → Methods.
    """

    objtype = "flow360class"
    directivetype = ClassDocumenter.objtype
    priority = 10 + ClassDocumenter.priority
    option_spec = dict(ClassDocumenter.option_spec)

    @classmethod
    def can_document_member(
        cls, member: Any, membername: str, isattr: bool, parent: Documenter
    ) -> bool:
        return isinstance(member, type)

    def __init__(self, *args) -> None:
        super().__init__(*args)
        if hasattr(self.object, "model_fields"):
            exclude = self.options.setdefault("exclude-members", set())
            exclude.update({"model_fields", "model_config", "model_computed_fields"})

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)

    def get_doc(self, *args: Any, **kwargs: Any) -> Any:
        """Strip a trailing transition rule from the class docstring.

        Several rc 25.10 schema docstrings end with a bare ``====`` rule. For
        a class with documented members that rule is harmless (members render
        after it), but for a member-less class it becomes the document's last
        node, and docutils rejects a document that ends with a transition
        ("Document may not end with a transition"). Drop the trailing rule
        before parsing so member-less classes build cleanly.
        """
        docstrings = super().get_doc(*args, **kwargs)
        if docstrings:
            block = docstrings[-1]
            while block and not block[-1].strip():
                block.pop()
            if block and _TRANSITION_RE.match(block[-1].strip()):
                preceded_by_title = len(block) >= 2 and block[-2].strip() != ""
                if not preceded_by_title:
                    block.pop()
                    while block and not block[-1].strip():
                        block.pop()
        return docstrings

    def document_members(self, all_members=False):
        pass

    def add_content(self, more_content: Any) -> None:
        super().add_content(more_content)
        source = self.get_sourcename()
        cls_name = self.objpath[-1] if self.objpath else self.object.__name__
        cls_path = f"{self.modname}.{'.'.join(self.objpath)}"

        self._emit_attributes(source, cls_name)
        self._emit_properties(source, cls_name)
        constructors, methods = collect_methods(self.object)
        self._emit_method_section(source, "Additional Constructors", constructors, cls_path)
        self._emit_method_section(source, "Methods", methods, cls_path)

    # ---- Section emitters ------------------------------------------------

    def _emit_attributes(self, source: str, cls_name: str) -> None:
        entries = collect_attributes(self.object)
        if not entries:
            return

        self.add_line("", source)
        self.add_line(".. rubric:: Attributes", source)
        self.add_line("", source)

        for name, field_info in entries:
            try:
                type_str = format_field_annotation(field_info)
            except Exception:
                type_str = str(field_info.annotation)

            self.add_line(f".. attribute:: {cls_name}.{name}", source)
            self.add_line(f"   :type: {type_str}", source)
            self.add_line("", source)

            if field_info.description:
                self.add_line(f"   {field_info.description}", source)
                self.add_line("", source)

            default = field_info.default
            factory = field_info.default_factory
            if default is not PydanticUndefined:
                self.add_line(f"   :Default: ``{format_default_value(default)}``", source)
                self.add_line("", source)
            elif factory is not None:
                try:
                    val = factory()
                    self.add_line(f"   :Default: ``{format_default_value(val)}``", source)
                except Exception:
                    self.add_line("   :Default: *factory*", source)
                self.add_line("", source)

    def _emit_properties(self, source: str, cls_name: str) -> None:
        props = collect_properties(self.object)
        if not props:
            return

        self.add_line("", source)
        self.add_line(".. rubric:: Properties", source)
        self.add_line("", source)

        from functools import cached_property

        for name, prop in props:
            getter = prop.fget if isinstance(prop, property) else prop
            if isinstance(prop, cached_property):
                getter = prop.func

            type_str = ""
            if getter is not None:
                ret = getattr(getter, "__annotations__", {}).get("return")
                if ret is not None:
                    try:
                        type_str = format_type_annotation(ret)
                    except Exception:
                        type_str = str(ret)

            self.add_line(f".. py:attribute:: {cls_name}.{name}", source)
            if type_str:
                self.add_line(f"   :type: {type_str}", source)
            self.add_line("", source)

            docstring = getattr(getter, "__doc__", None)
            if docstring:
                for line in process_docstring_napoleon(docstring):
                    self.add_line(line, source)
                self.add_line("", source)

    def _emit_method_section(
        self, source: str, title: str, names: list[str], cls_path: str
    ) -> None:
        if not names:
            return
        self.add_line("", source)
        self.add_line(f".. rubric:: {title}", source)
        self.add_line("", source)
        for name in names:
            self.add_line(f".. automethod:: {cls_path}.{name}", source)
            self.add_line("", source)
