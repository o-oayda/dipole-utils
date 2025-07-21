"""
dipole-utils: Utilities for fitting dipoles and higher order harmonics to healpy maps.
"""

from dipoleutils.utils.data_loader import _config_manager, CatalogueValidationError

def configure_data_directory(path: str) -> None:
    """
    Configure the global data directory for dipole-utils.
    This setting will persist across sessions.
    
    Args:
        path: Absolute path to the directory containing catalogue files
        
    Raises:
        ValueError: If the directory doesn't exist
    """
    _config_manager.set_data_directory(path)
    print(f"Global data directory configured: {path}")

def get_data_directory() -> str | None:
    """
    Get the configured global data directory.
    
    Returns:
        The configured data directory path, or None if not configured
    """
    return _config_manager.get_data_directory()

def clear_data_directory_config() -> None:
    """Clear the persistent data directory configuration."""
    _config_manager.clear_data_directory()
    print("Global data directory configuration cleared")

__all__ = [
    'configure_data_directory',
    'get_data_directory', 
    'clear_data_directory_config',
    'CatalogueValidationError'
]