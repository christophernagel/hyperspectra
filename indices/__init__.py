"""
Spectral indices for mineral, hydrocarbon, and vegetation mapping.

This module implements peer-reviewed spectral indices with full citations.
"""

from .vegetation import (
    ndvi, ndwi, ndmi, evi, savi,
    pssr, mcari, tcari, osavi
)
from .minerals import (
    clay_index, carbonate_index, alunite_index,
    kaolinite_index, smectite_index, illite_index,
    ferric_iron_index, ferrous_iron_index,
    calcite_index, dolomite_index
)
from .hydrocarbons import (
    hydrocarbon_index, oil_index,
    methane_ratio, hydrocarbon_absorption_depth
)
from .nitrogen import (
    ndni, nri, canopy_chlorophyll_index,
    protein_index
)

__all__ = [
    # Vegetation
    'ndvi', 'ndwi', 'ndmi', 'evi', 'savi',
    'pssr', 'mcari', 'tcari', 'osavi',
    # Minerals
    'clay_index', 'carbonate_index', 'alunite_index',
    'kaolinite_index', 'smectite_index', 'illite_index',
    'ferric_iron_index', 'ferrous_iron_index',
    'calcite_index', 'dolomite_index',
    # Hydrocarbons
    'hydrocarbon_index', 'oil_index',
    'methane_ratio', 'hydrocarbon_absorption_depth',
    # Nitrogen/Agriculture
    'ndni', 'nri', 'canopy_chlorophyll_index',
    'protein_index',
]
