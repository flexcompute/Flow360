.. _python_API_migration_guide:

.. currentmodule:: flow360

***************
Migration Guide
***************

These guides help you transition from legacy Flow360 API patterns to the modern Pydantic-based interface, ensuring a smooth upgrade path while maintaining simulation capabilities.

.. grid:: 1

    .. grid-item-card:: 🔄 BET Model Conversion
        :link: bet_migration
        :link-type: doc
        
        Learn how to convert legacy Blade Element Theory model definitions to the modern Pydantic model format.

    .. grid-item-card:: 📊 Monitor Conversion
        :link: monitor_conversion
        :link-type: doc
        
        Guide for transitioning simulation monitors from legacy dictionary-based format to structured Pydantic models.

Key Benefits of Migration
=========================

* Better type safety and validation with Pydantic models
* Improved IDE autocompletion support
* Clearer error messages with specific validation feedback
* Consistent API patterns across all Flow360 components
* Access to new features available only in modern API

When to Migrate
===============

* When updating existing scripts to use newer Flow360 versions
* If you need to leverage new simulation capabilities
* When standardizing workflows across team members
* For improved maintainability of long-term production code

.. toctree::
   :hidden:

   bet_migration
   monitor_conversion