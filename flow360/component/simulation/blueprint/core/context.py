from typing import Any

from .resolver import CallableResolver


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
    """

    def __init__(self, resolver: CallableResolver, initial_values: dict[str, Any] | None = None) -> None:
        self._values = initial_values or {}
        self._resolver = resolver

    def get(self, name: str) -> Any:
        if name not in self._values:
            # Try loading from builtin callables/constants if possible
            try:
                val = self.resolve(name)
                # If successful, store it so we don't need to import again
                self._values[name] = val
            except ValueError as err:
                raise NameError(f"Name '{name}' is not defined") from err
        return self._values[name]

    def set(self, name: str, value: Any) -> None:
        self._values[name] = value

    def resolve(self, name):
        return self._resolver.get_allowed_callable(name)

    def copy(self) -> "EvaluationContext":
        """Create a copy of the current context."""
        return EvaluationContext(self._resolver, dict(self._values))
