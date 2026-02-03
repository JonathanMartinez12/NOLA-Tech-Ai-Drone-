"""
Configuration module for AI Grand Prix Drone Racing.

This module provides centralized configuration management for all
tunable parameters, drone specifications, and system settings.
"""

from config.settings import Settings, get_settings
from config.drone_specs import DroneSpecs, get_drone_specs

__all__ = ["Settings", "get_settings", "DroneSpecs", "get_drone_specs"]
