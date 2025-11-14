"""
Sharing the config for base models to reduce unnecessary inheritance from Flow360BaseModel.
"""

import pydantic as pd


def snake_to_camel(string: str) -> str:
    """
    Convert a snake_case string to camelCase.

    This function takes a snake_case string as input and converts it to camelCase.
    It splits the input string by underscores, capitalizes the first letter of
    each subsequent component (after the first one), and joins them together.

    Parameters:
    string (str): The input string in snake_case format.

    Returns:
    str: The converted string in camelCase format.

    Example:
    >>> snake_to_camel("example_snake_case")
    'exampleSnakeCase'
    """
    components = string.split("_")

    camel_case_string = components[0]

    for component in components[1:]:
        camel_case_string += component[0].upper() + component[1:]

    return camel_case_string


base_model_config = pd.ConfigDict(
    ##:: Pydantic kwargs
    arbitrary_types_allowed=True,  # ?
    extra="forbid",
    frozen=False,
    populate_by_name=True,
    validate_assignment=True,
    validate_default=True,
    ##:: Custom keys
    require_one_of=[],
    allow_but_remove=[],
    conflicting_fields=[],
    include_hash=False,
    include_defaults_in_schema=True,
    # pylint: disable=fixme
    # TODO: Remove alias_generator since it is only for translator
    alias_generator=pd.AliasGenerator(
        serialization_alias=snake_to_camel,
    ),
)
