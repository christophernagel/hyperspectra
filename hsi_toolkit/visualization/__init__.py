"""
Visualization Module

Interactive visualizations for exploring hyperspectral imaging concepts.

Features:
    - Interactive parameter exploration
    - Real-time simulation updates
    - Educational animations
    - Comparative analysis tools
"""

from .hsi_visualizer import HSIVisualizer
from .dashboard import launch_dashboard, create_dashboard_app

__all__ = ['HSIVisualizer', 'launch_dashboard', 'create_dashboard_app']
