"""
This module implements a dependency graph for user variables.
"""

import ast
import copy
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set

from pydantic import ValidationError
from pydantic_core import InitErrorDetails


class DependencyGraph:
    """
    A dependency graph for variables.
    """

    __slots__ = ("_graph", "_deps")

    def __init__(self):
        # adjacency list: key = variable u, value = set of variables v that depend on u
        self._graph: Dict[str, Set[str]] = defaultdict(set)
        # reverse dependency map: key = variable v, value = set of variables u that v depends on
        self._deps: Dict[str, Set[str]] = defaultdict(set)

    @staticmethod
    def _extract_deps(expression: str, all_names: Set[str]) -> Set[str]:
        """
        Parse the expression into an AST and collect all Name nodes,
        then filter them against the set of known variable names.
        """
        # trailing semicolon breaks the AST parser
        expression = expression.rstrip("; \n\t")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError:
            return set()
        found: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                found.add(node.id)
        return found & all_names

    def _check_for_cycle(self) -> None:
        """
        Use Kahn's algorithm to detect cycles; if any remain unprocessed,
        raise ValidationError with details on the cycle.
        """
        indegree = {name: len(self._deps[name]) for name in self._deps}
        for name in self._graph:
            indegree.setdefault(name, 0)

        queue = deque([n for n, deg in indegree.items() if deg == 0])
        processed = set(queue)

        while queue:
            u = queue.popleft()
            for v in self._graph.get(u, ()):
                indegree[v] -= 1
                if indegree[v] == 0:
                    processed.add(v)
                    queue.append(v)

        if len(processed) != len(indegree):
            cycle_nodes = set(indegree) - processed
            details = InitErrorDetails(
                type="value_error",
                ctx={"error": f"Circular dependency detected among: {sorted(cycle_nodes)}"},
            )
            raise ValidationError.from_exception_data("Variable value error", [details])

    def load_from_list(self, vars_list: List[Dict[str, Any]]) -> None:
        """
        Load variables from a list of dicts, clear existing graph, build dependencies,
        and restore on error.

        Expected schema of the variable list:
        ``` JSON
        [
            {
                "name": str,
                "value": str
            }
        ]
        ```
        """
        # backup state
        old_graph = copy.deepcopy(self._graph)
        old_deps = copy.deepcopy(self._deps)
        try:
            self._graph.clear()
            self._deps.clear()
            names = {item["name"] for item in vars_list}
            for name in names:
                # pylint: disable=pointless-statement
                self._graph[name]
                self._deps[name]

            for item in vars_list:
                name = item["name"]
                expr = item["value"]
                deps = self._extract_deps(expr, names)
                for dep in deps:
                    if dep not in names:
                        raise ValueError(
                            f"Expression for {name!r} references unknown variable {dep!r}"
                        )
                    self._graph[dep].add(name)
                    self._deps[name].add(dep)

            self._check_for_cycle()
        except Exception:
            # restore old state on any error
            self._graph = old_graph
            self._deps = old_deps
            raise

    def add_variable(self, name: str, expression: Optional[str] = None) -> None:
        """
        Add or overwrite a variable. Restores previous state on cycle.
        """
        # backup state
        old_graph = copy.deepcopy(self._graph)
        old_deps = copy.deepcopy(self._deps)
        try:
            # clear existing edges if overwrite
            if name in self._graph:
                for dep in self._deps[name]:
                    self._graph[dep].discard(name)
                self._deps[name].clear()
                for dependents in self._graph.values():
                    dependents.discard(name)
            else:
                self._graph[name] = set()
                self._deps[name] = set()

            if expression:
                deps = self._extract_deps(expression, set(self._graph.keys()))
                for dep in deps:
                    if dep not in self._graph:
                        raise ValueError(
                            f"Expression for {name!r} references unknown variable {dep!r}"
                        )
                    self._graph[dep].add(name)
                    self._deps[name].add(dep)

            self._check_for_cycle()
        except Exception:
            # restore old state
            self._graph = old_graph
            self._deps = old_deps
            raise

    def remove_variable(self, name: str) -> None:
        """
        Remove a variable and all its edges.
        """
        if name not in self._graph:
            raise KeyError(f"Variable {name!r} does not exist")

        for dep in self._deps[name]:
            self._graph[dep].discard(name)
        del self._deps[name]

        del self._graph[name]
        for dependents in self._graph.values():
            dependents.discard(name)

    def update_expression(self, name: str, expression: Optional[str]) -> None:
        """
        Update expression for an existing variable; restores previous state on cycle.
        """
        if name not in self._graph:
            raise KeyError(f"Variable {name!r} does not exist")

        # backup state
        old_graph = copy.deepcopy(self._graph)
        old_deps = copy.deepcopy(self._deps)
        try:
            # clear old deps
            for dep in self._deps[name]:
                self._graph[dep].discard(name)
            self._deps[name].clear()

            if expression:
                deps = self._extract_deps(expression, set(self._graph.keys()))
                for dep in deps:
                    if dep not in self._graph:
                        raise ValueError(
                            f"Expression for {name!r} references unknown variable {dep!r}"
                        )
                    self._graph[dep].add(name)
                    self._deps[name].add(dep)

            self._check_for_cycle()
        except Exception:
            # restore on error
            self._graph = old_graph
            self._deps = old_deps
            raise

    def topology_sort(self) -> List[str]:
        """
        Return names in topological order; raises ValidationError on cycle.
        """
        indegree = {n: len(self._deps[n]) for n in self._deps}
        for n in self._graph:
            indegree.setdefault(n, 0)

        queue = deque(n for n, d in indegree.items() if d == 0)
        order: List[str] = []

        while queue:
            u = queue.popleft()
            order.append(u)
            for v in self._graph.get(u, ()):  # type: ignore
                indegree[v] -= 1
                if indegree[v] == 0:
                    queue.append(v)

        if len(order) != len(indegree):
            cycle_nodes = set(indegree) - set(order)
            details = InitErrorDetails(
                type="value_error",
                ctx={"error": f"Circular dependency detected among: {sorted(cycle_nodes)}"},
            )
            raise ValidationError.from_exception_data("Variable value error", [details])
        return order
