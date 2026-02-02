"""
Forward Model Module

Integrates atmospheric and sensor models into a complete
simulation of the hyperspectral imaging chain.

The forward model answers:
    Given a surface reflectance, what does the sensor measure?

The inverse problem (atmospheric correction) answers:
    Given a measurement, what is the surface reflectance?
"""

from .forward_model import ForwardModel, SceneParameters
from .radiative_transfer import RadiativeTransfer, RTParameters

__all__ = ['ForwardModel', 'SceneParameters', 'RadiativeTransfer', 'RTParameters']
