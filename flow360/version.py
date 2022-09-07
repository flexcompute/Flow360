"""
Version control module
"""
__version__ = "22.2.3.0"

import re


class Flow360Version:
    """
    Flow360 version
    """

    def __init__(self, version):
        self.full = version
        # pylint: disable=anomalous-backslash-in-string,consider-using-f-string
        ret = re.findall("^([a-zA-Z0-9\-]+)-([0-9\.]+)$", version)
        if len(ret) != 1:
            print("matched version = " + str(ret))
            raise RuntimeError("solver version is not valid: {}".format(version))
        self.head = ret[0][0]
        self.tail = [int(i) for i in ret[0][1].strip().split(".")]
        if self.head == "master":
            self.tail = [i * 100 for i in self.tail]

    def __lt__(self, other):
        return self.tail < other.tail

    def __le__(self, other):
        return self.tail <= other.tail

    def __gt__(self, other):
        return self.tail > other.tail

    def __ge__(self, other):
        return self.tail >= other.tail

    def __eq__(self, other):
        return self.tail == other.tail and self.head == other.head

    def __ne__(self, other):
        return self.tail != other.tail or self.head != other.head
