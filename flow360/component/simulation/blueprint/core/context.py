"""Evaluation context that contains references to known symbols"""

from typing import Any, Optional

from flow360.component.simulation.blueprint.core.resolver import CallableResolver


class ReturnValue(Exception):
    """
    Custom exception to signal a 'return' during the evaluation
    of a function model.
    """

    def __init__(self, value: Any):
        super().__init__("Function returned.")
        self.value = value


class EvaluationContext:
    """
    Manages variable scope and access during function evaluation.

    This class stores named values and optionally resolves names through a
    `CallableResolver` when not already defined in the context.
    """

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
        self._resolver = resolver

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
                raise NameError(f"Name '{name}' is not defined") from err
        return self._values[name]

    def set(self, name: str, value: Any) -> None:
        """
        Assign a value to a name in the context.

        Args:
            name (str): The variable name to set.
            value (Any): The value to assign.
        """
        self._values[name] = value

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
        return EvaluationContext(self._resolver, dict(self._values))
