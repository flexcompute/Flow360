"""
utilities module
"""


# pylint: disable=invalid-name
class classproperty(property):
    """classproperty decorator"""

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)
