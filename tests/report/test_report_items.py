import pytest
from flow360.plugins.report.report_items import human_readable_formatter

@pytest.mark.parametrize("value,expected", [
    # Large values (millions)
    (225422268, "225M"),   # Large number well into millions
    (1000000, "1M"),       # Exactly 1 million
    (9999999, "10M"),      # Just under 10 million, rounds to 10M
    (25400000, "25M"),     # Between 10 and 100 million

    # Thousands
    (22542, "23k"),      # Between 10k and 100k => one decimal
    (225422, "225k"),      # Over 100k => no decimals
    (2254, "2.3k"),       # Under 10k => one decimal
    (1000, "1k"),          # Exactly 1k

    # Less than 1000
    (225.4, "225.4"),      # No suffix, up to two decimals
    (2.345, "2.345"),      # No change
    (2, "2"),              # Whole number <1000
    (0.5, "0.5"),          # Decimal less than 1
    (0.123456, "0.123456"),  # no change

    # Negative values
    (-225422268, "-225M"), # Negative large number
    (-22542, "-23k"),
    (-2254, "-2.3k"),
    (-225.4, "-225.4"),
    (-2.345, "-2.345"),

    # Non-numeric
    ("abc", "abc"),
    (None, "None"),
])
def test_human_readable_formatter(value, expected):
    assert human_readable_formatter(value) == expected
