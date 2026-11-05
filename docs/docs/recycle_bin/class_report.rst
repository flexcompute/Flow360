:html_theme.sidebar_secondary.remove:
{{ fullname | escape | underline}}
{% set excluded_members = "entity_bucket, \
                           entity_type, \
                           full_name, \
                           used_entity_registry, \
                           model_constructor, \
                           model_computed_fields, \
                           model_config, \
                           type_name, \
                           model_fields" %}

{% set inherited_members = "Flow360BaseModel, \
                           _ParamModelBase, \
                           _VolumeEntityBase, \
                           MultiConstructorBaseModel" %}

.. currentmodule:: flow360

.. autopydantic_model:: {{ fullname }}
   :members:
   :show-inheritance:
   :undoc-members:
   :member-order: groupwise
   :inherited-members: {{ inherited_members }}
   :exclude-members: {{ excluded_members }}

.. ..  rubric:: Inherited Common Usage

.. ..   include:: ../_custom_autosummary/{{ fullname }}.rst