from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated

from flow360.component.utils import process_expressions

# TODO: Add units to expression?
# TODO: Add variable existence check?
StringExpression = Annotated[str, AfterValidator(process_expressions)]