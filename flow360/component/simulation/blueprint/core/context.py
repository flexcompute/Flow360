"""Evaluation context that contains references to known symbols"""

import collections
from copy import deepcopy
from typing import Any, List, Optional, Tuple

import pydantic as pd

from flow360.component.simulation.blueprint.core.dependency_graph import DependencyGraph
from flow360.component.simulation.blueprint.core.resolver import CallableResolver
from flow360.log import log


class ReturnValue(Exception):
    """
    Custom exception to signal a 'return' during the evaluation
    of a function model.
    """

    def __init__(self, value: Any):
        super().__init__("Function returned.")
        self.value = value


def _levenshtein_distance(a: str, b: str) -> int:
    """Compute the Levenshtein distance between two strings."""
    if len(a) < len(b):
        return _levenshtein_distance(a=b, b=a)
    if not b:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        row = [i]
        for j, cb in enumerate(b, 1):
            insert = prev_row[j] + 1
            delete = row[j - 1] + 1
            replace = prev_row[j - 1] + (ca != cb)
            row.append(min(insert, delete, replace))
        prev_row = row
    return prev_row[-1]


def _find_closest_levenshtein(target: str, candidates: List[str]) -> Optional[Tuple[str, int]]:
    """
    Return (best_match, distance) where `best_match` is the candidate
    with the smallest edit distance to `target`. If `candidates` is empty,
    returns None.
    """
    if not candidates:
        return None

    best = (None, float("inf"))
    for candidate in candidates:
        d = _levenshtein_distance(target, candidate)
        if d < best[1]:
            best = (candidate, d)
    return best  # type: ignore


class EvaluationContext:
    """
    Manages variable scope and access during function evaluation.

    This class stores named values and optionally resolves names through a
    `CallableResolver` when not already defined in the context.
    """

    __slots__ = (
        "_values",
        "_data_models",
        "_metadata",  # Contains description of the variable
        "_resolver",
        "_aliases",
        "_dependency_graph",
    )

    def __init__(
        self, resolver: CallableResolver, initial_values: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Initialize the evaluation context.

        Args:
            resolver (CallableResolver): A resolver used to look up callable names
                and constants if not explicitly defined.
            initial_values (Optional[dict[str, Any]]): Initial variable values to populate
                the context with.
        """
        self._values = initial_values or {}
        self._data_models = {}
        self._metadata = {}
        self._resolver = resolver
        self._aliases: dict[str, str] = {}
        self._dependency_graph = DependencyGraph()

    def get(self, name: str, resolve: bool = True) -> Any:
        """
        Retrieve a value by name from the context.

        If the name is not explicitly defined and `resolve` is True,
        attempt to resolve it using the resolver.

        Args:
            name (str): The variable or callable name to retrieve.
            resolve (bool): Whether to attempt to resolve the name if it's undefined.

        Returns:
            Any: The corresponding value.

        Raises:
            NameError: If the name is not found and cannot be resolved.
            ValueError: If resolution is disabled and the name is undefined.
        """
        if name not in self._values:
            # Try loading from builtin callables/constants if possible
            try:
                if not resolve:
                    raise ValueError(f"{name} was not defined explicitly in the context")
                val = self.resolve(name)
                # If successful, store it so we don't need to import again
                self._values[name] = val
            except ValueError as err:
                closest_name, distance = _find_closest_levenshtein(
                    name, list(self._values.keys()) + ["math.pi"]
                )
                error_message = f"Name '{name}' is not defined."
                if distance < 4:
                    # Else the recommended name might not make sense.
                    error_message += f" Did you mean: `{closest_name}`?"
                raise NameError(error_message) from err
        return self._values[name]

    def _get_all_variables_to_remove(self, start_name: str) -> set[str]:
        # pylint: disable=protected-access
        to_remove = {start_name}
        queue = collections.deque([start_name])

        while queue:
            current_name = queue.popleft()
            dependents_of_current = self._dependency_graph._graph.get(current_name, set())

            for dep in dependents_of_current:
                if dep not in to_remove:
                    to_remove.add(dep)
                    queue.append(dep)
        return to_remove

    def remove(self, name: str, _is_recursive_call: bool = False) -> None:
        """
        Remove the variable with the input name in the context.
        The variables that depends on this variable are also removed recursively.
        """
        # pylint: disable=protected-access
        if name not in self._values:
            raise NameError(f"There is no variable named '{name}'.")
        if not _is_recursive_call:
            all_affected_vars = self._get_all_variables_to_remove(name)

            log.info("--- Confirmation Required ---")
            log.info("The following variables will be removed:")
            for var in sorted(list(all_affected_vars)):  # Sort for consistent display
                log.info(f"  - {var}")

            if len(all_affected_vars) > 1:
                confirmation = (
                    input(
                        f"Are you sure you want to remove '{name}' and "
                        f"its {len(all_affected_vars) - 1} dependent variable(s)? (yes/no): "
                    )
                    .lower()
                    .strip()
                )
            else:
                confirmation = (
                    input(f"Are you sure you want to remove '{name}'? (yes/no): ").lower().strip()
                )

            if confirmation not in ["yes", "y"]:
                log.info("Operation cancelled. No variables were removed.")
                return
            log.info("--- Proceeding with removal ---")

        current_dependents = self._dependency_graph._graph.get(name, set()).copy()

        for dep in current_dependents:
            self.remove(name=dep, _is_recursive_call=True)  # Pass the flag

        self._dependency_graph.remove_variable(name=name)
        self._values.pop(name)
        self._metadata.pop(name, None)
        log.info(f"Removed '{name}' from values.")

    def get_data_model(self, name: str) -> Optional[pd.BaseModel]:
        """Get the Validation model for the given name."""
        if name not in self._data_models:
            return None
        return self._data_models[name]

    def set_alias(self, name, alias) -> None:
        """
        Set alias used for code generation.
        This is meant for non-user variables.
        """
        self._aliases[name] = alias

    def get_alias(self, name) -> Optional[str]:
        """
        Get alias used for code generation.
        This is meant for non-user variables.
        """
        return self._aliases.get(name)

    def set(self, name: str, value: Any, data_model: pd.BaseModel = None) -> None:
        """
        Assign a value to a name in the context.

        Args:
            name (str): The variable name to set.
            value (Any): The value to assign.
            data_model (BaseModel, optional): The type of the associate with this entry (for non-user variables)
        """
        if name in self._values:
            self._dependency_graph.update_expression(name, str(value))
        else:
            self._dependency_graph.add_variable(name, str(value))

        self._values[name] = value

        if data_model:
            self._data_models[name] = data_model

    def resolve(self, name):
        """
        Resolve a name using the provided resolver.

        Args:
            name (str): The name to resolve.

        Returns:
            Any: The resolved callable or constant.

        Raises:
            ValueError: If the name cannot be resolved by the resolver.
        """
        return self._resolver.get_allowed_callable(name)

    def can_evaluate(self, name) -> bool:
        """
        Check if the name can be evaluated via the resolver.

        Args:
            name (str): The name to check.

        Returns:
            bool: True if the name is allowed and resolvable, False otherwise.
        """
        return self._resolver.can_evaluate(name)

    def copy(self) -> "EvaluationContext":
        """
        Create a copy of the current context.

        Returns:
            EvaluationContext: A new context instance with the same resolver and a copy
            of the current variable values.
        """
        return deepcopy(self)

    def set_metadata(self, name: str, key: str, value: Any) -> None:
        """
        Set the metadata for a variable.
        """
        if name not in self._metadata:
            self._metadata[name] = {}

        self._metadata[name][key] = value

    def get_metadata(self, name: str, key: str) -> Any:
        """
        Get the metadata for a variable.

        Returns:
            Any: The metadata value if it exists, otherwise None.
        """
        return self._metadata.get(name, {}).get(key, None)

    @property
    def user_variable_names(self):
        """Get the set of user variables in the context."""
        return {name for name in self._values.keys() if "." not in name}

    @property
    def registered_names(self):
        """Show the registered names in the context."""
        return list(self._values.keys())
