"""Smoke test verifying the species library is reachable through the public ``flow360`` package.

The schema-side library and Species.from_library classmethod are covered exhaustively
in flex/share/flow360-schema/tests/simulation/params/test_species_library.py. This file
verifies only the re-export plumbing -- that a user typing ``import flow360 as fl``
can reach ``fl.Species.from_library`` and ``fl.SpeciesTransportModel``.
"""

import flow360 as fl


def test_species_from_library_reachable_via_fl():
    """fl.Species.from_library works through the public re-export."""
    with fl.SI_unit_system:
        n2 = fl.Species.from_library("N2")
    assert n2.name == "N2"
    assert n2.molecular_weight.to("g/mol").v.item() == 28.0134


def test_species_transport_model_reachable_via_fl():
    """fl.SpeciesTransportModel is exposed and accepts library-built species."""
    with fl.SI_unit_system:
        n2 = fl.Species.from_library("N2")
        n2.mass_fraction = 0.767
        o2 = fl.Species.from_library("O2")
        o2.mass_fraction = 0.233
        o2.schmidt_number = 0.72
        model = fl.SpeciesTransportModel(species=[n2, o2])
    assert [s.name for s in model.species] == ["N2", "O2"]
    assert [s.schmidt_number for s in model.species] == [0.7, 0.72]


def test_library_discovery_via_unknown_name_listing():
    """An unknown species name raises ValueError listing all available species."""
    import pytest

    with fl.SI_unit_system, pytest.raises(ValueError, match="Available species"):
        fl.Species.from_library("XYZ")
