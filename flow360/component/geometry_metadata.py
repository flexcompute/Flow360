"""Geometry metadata pipeline output models"""

from collections import defaultdict
from typing import DefaultDict, List, Literal, Optional, Set, Union

import pydantic as pd

from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.primitives import Edge, Surface
from flow360.component.simulation.web.asset_metadata import MetadataBaseModel
from flow360.log import log


class _KeyValuePair(MetadataBaseModel):
    key: str = pd.Field()
    value: Union[List, str]  = pd.Field()


class _FaceEdgeMeta(MetadataBaseModel):
    id: str = pd.Field()
    attributes: Optional[List[_KeyValuePair]] = pd.Field(None)


class _FaceInfo(MetadataBaseModel):
    type: Literal["face"] = pd.Field("face", frozen=True)
    metadata: _FaceEdgeMeta = pd.Field()


class _EdgeInfo(MetadataBaseModel):
    type: Literal["edge"] = pd.Field("edge", frozen=True)
    metadata: _FaceEdgeMeta = pd.Field()


class _BodyMeta(MetadataBaseModel):
    faces: List[_FaceInfo] = pd.Field()
    edges: List[_EdgeInfo] = pd.Field()


class _BodyInfo(MetadataBaseModel):
    type: Literal["body"] = pd.Field("body", frozen=True)
    metadata: _BodyMeta = pd.Field()


class _GeometryMetadataModel(MetadataBaseModel):
    """
    A model to handle geometry metadata including bodies and processing of entity types like faces or edges.
    """

    bodies: list[_BodyInfo] = pd.Field()
    version: str = pd.Field()

    def process_metadata_for_given_type(self, entity_type_name: Literal["faces", "edges"]):
        """
        Process the metadata for the given entity type (faces or edges) and return grouped information.

        Args:
            entity_type_name (Literal["faces", "edges"]): The type of entity to process ('faces' or 'edges').

        Returns:
            DefaultDict[str, DefaultDict[str, Set[str]]]: Grouped information based on entity attributes.

        Metadata JSON example:
            {
                "faceName": {
                    "MyFace1": {"body01_face001"},
                    "MyFace2": {"body01_face002"},
                },
                "boundaryName": {
                    "Wing1": ["body01_face002"],
                    "Wing2": ["body01_face003"],
                },
                "__all__": {"body01_face001", "body01_face002", "body01_face003"},
            }
        """
        # Create the nested defaultdict with sets as the default value
        group_name_to_items: DefaultDict[str, DefaultDict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        group_name_to_items["__all__"] = set()

        # pylint: disable=not-an-iterable
        for body in self.bodies:
            for item in getattr(body.metadata, entity_type_name):
                item_id = item.metadata.id
                if item.metadata.attributes is not None:
                    for item_attribute in item.metadata.attributes:
                        item_tag_key = item_attribute.key
                        item_tag_value = item_attribute.value
                        group_name_to_items[item_tag_key][item_tag_value].add(item_id)
                group_name_to_items["__all__"].add(item_id)
        return group_name_to_items

    def group_items_with_given_tag(
        self,
        entity_class: Union[type[Surface], type[Edge]],
        attribute_tag: str,
        registry: EntityRegistry,
        warn_ungrouped: bool = True,
    ) -> EntityRegistry:
        """
        Group items with the given tag key and return an EntityRegistry.
        """

        entity_type_name = "faces" if entity_class is Surface else "edges"

        grouped_items_collection = self.process_metadata_for_given_type(entity_type_name)
        if attribute_tag not in grouped_items_collection:
            raise ValueError(
                f"None of the {entity_type_name} have the attribute key '{attribute_tag}'."
            )

        group_with_tag = grouped_items_collection[attribute_tag]
        included_item = set()
        for group_name, grouped_items in group_with_tag.items():
            registry.register(
                entity_class(
                    name=group_name,
                    private_attribute_sub_components=list(grouped_items),
                    private_attribute_tag_key=attribute_tag,
                )
            )
            included_item.update(grouped_items)

        all_items = grouped_items_collection["__all__"]
        for item in all_items.difference(included_item):
            registry.register(
                entity_class(
                    name=item,
                    private_attribute_sub_components=[item],
                    private_attribute_tag_key="__standalone__",
                ),
            )
            if warn_ungrouped:
                log.warning(
                    f"'{item}' does not contain the given attribute_tag: '{attribute_tag}'. "
                    "It will have to be refenreced by its id. "
                    "Please make sure this is intended."
                )
        # For other assets (surface mesh, volume mesh), we need to merge above registered entities with brith
        # SimulationParams used entities. This is not relevant for geometry metadata.
        return registry
