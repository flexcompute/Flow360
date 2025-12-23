import inspect

import flow360.component.simulation.draft_context.mirror as mirror
import flow360.component.simulation.outputs.output_entities as output_entities
import flow360.component.simulation.primitives as primitives
from flow360.component.simulation.framework.entity_materializer import ENTITY_TYPE_MAP


def test_entity_type_map_completeness():
    """
    Ensure all classes in primitives and output_entities that define
    'private_attribute_entity_type_name' are registered in ENTITY_TYPE_MAP.
    """

    modules_to_check = [primitives, output_entities, mirror]

    missing_entities = []

    for module in modules_to_check:
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                # Check if the class has 'private_attribute_entity_type_name'
                # We need to check if it's a pydantic model field or a class attribute

                # Skip private classes and EntityBase
                if name.startswith("_") or name == "EntityBase":
                    continue

                if inspect.isabstract(obj):
                    continue

                has_entity_type = False

                # Check Pydantic fields
                if (
                    hasattr(obj, "model_fields")
                    and "private_attribute_entity_type_name" in obj.model_fields
                ):
                    has_entity_type = True

                # Check class attributes (if defined as a simple attribute, though unlikely for Pydantic models)
                elif hasattr(obj, "private_attribute_entity_type_name"):
                    has_entity_type = True

                if has_entity_type:
                    # Get the expected type name.
                    # Usually it's the class name or the default value of the field.
                    # We check if the class name itself is in the map or if the value is in the map.

                    # For safety, we check if the class itself is in ENTITY_TYPE_MAP.values()
                    if obj not in ENTITY_TYPE_MAP.values():
                        missing_entities.append(name)

    assert (
        not missing_entities
    ), f"The following entities are missing from ENTITY_TYPE_MAP: {missing_entities}"
