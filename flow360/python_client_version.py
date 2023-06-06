"""
Version control module
"""
__version__ = "v0.2.0"

import re


class Flow360ClientVersion:
    """
    Flow360 python client version
    """

    def __init__(self, version):
        self.full = version
        # format v<major>.<minor>.<fix>
        match = re.match(r"^v(\d+)\.(\d+)\.(\d+)(?:b.*)?$", version)
        if match:
            major, minor, fix = map(int, match.groups())

            fix = str(fix).split('b')[0]
            self.tail = int(f"{major}{minor}{fix}")
        else:
            print("matched version = " + str(match))
            raise RuntimeError(
                "solver version is not valid: {}".format(version))

    def __lt__(self, other):
        return self.tail < other.tail

    def __le__(self, other):
        return self.tail <= other.tail

    def __gt__(self, other):
        return self.tail > other.tail

    def __ge__(self, other):
        return self.tail >= other.tail

    def __eq__(self, other):
        return self.tail == other.tail

    def __ne__(self, other):
        return self.tail != other.tail
