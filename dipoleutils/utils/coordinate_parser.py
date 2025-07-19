from astropy.table import Table
import re
from typing import Dict, List, Optional, Tuple


class CoordinateSystemParser:
    """
    A parser for identifying coordinate systems in astronomical catalogues.
    
    This class handles the detection of equatorial, galactic, and ecliptic
    coordinate systems based on column names in astropy Tables.
    """
    
    def __init__(self):
        self.coordinate_patterns = {
            'equatorial': {
                'azimuthal': [  # RA-like columns
                    r'ra\b', r'right[\s_-]*ascension', r'alpha', r'α',
                    r'r\.?a\.?', r'ra_deg', r'ra_rad', r'ra_hours'
                ],
                'polar': [  # Dec-like columns  
                    r'dec\b', r'declination', r'delta', r'δ',
                    r'dec_deg', r'dec_rad', r'de\b'
                ]
            },
            'galactic': {
                'azimuthal': [  # Galactic longitude-like columns
                    r'l\b', r'glon', r'galactic[\s_-]*longitude',
                    r'longitude', r'lon', r'gl\b', r'l_deg', r'l_rad'
                ],
                'polar': [  # Galactic latitude-like columns
                    r'b\b', r'glat', r'galactic[\s_-]*latitude',
                    r'latitude', r'lat', r'gb\b', r'b_deg', r'b_rad'
                ]
            },
            'ecliptic': {
                'azimuthal': [  # Ecliptic longitude-like columns
                    r'elon', r'ecliptic[\s_-]*longitude', r'lambda', r'λ',
                    r'ecl_lon', r'elam'
                ],
                'polar': [  # Ecliptic latitude-like columns
                    r'elat', r'ecliptic[\s_-]*latitude', r'beta', r'β',
                    r'ecl_lat', r'ebet'
                ]
            }
        }
    
    def parse_coordinate_systems(self, catalogue: Table) -> Dict[str, Dict[str, str]]:
        """
        Parse the catalogue and identify all coordinate systems present.
        
        Args:
            catalogue: Astropy Table containing the data
            
        Returns:
            Dictionary mapping coordinate system names to their column mappings
            e.g., {'equatorial': {'azimuthal': 'RA', 'polar': 'DEC'}}
        """
        column_names = [col.lower() for col in catalogue.colnames]
        coordinate_systems = {}
        
        for system_name, patterns in self.coordinate_patterns.items():
            azimuthal_matches = self._find_column_matches(
                column_names, patterns['azimuthal'], catalogue.colnames
            )
            polar_matches = self._find_column_matches(
                column_names, patterns['polar'], catalogue.colnames
            )
            
            if azimuthal_matches and polar_matches:
                # Select the best match for each coordinate type
                best_azimuthal = self._select_best_match(
                    azimuthal_matches, patterns['azimuthal']
                )
                best_polar = self._select_best_match(
                    polar_matches, patterns['polar']
                )
                
                coordinate_systems[system_name] = {
                    'azimuthal': best_azimuthal,
                    'polar': best_polar
                }
        
        return coordinate_systems
    
    def _find_column_matches(self, column_names_lower: List[str], 
                           patterns: List[str], original_colnames: List[str]) -> List[str]:
        """Find column names matching any of the given patterns."""
        matches = []
        for i, col_name in enumerate(column_names_lower):
            for pattern in patterns:
                if re.search(pattern, col_name, re.IGNORECASE):
                    # Get original column name (not lowercased)
                    original_name = original_colnames[i]
                    if original_name not in matches:
                        matches.append(original_name)
                    break
        return matches
    
    def _select_best_match(self, matches: List[str], patterns: List[str]) -> str:
        """Select the best match from a list of potential matches."""
        # Prefer exact matches
        for match in matches:
            if self._is_exact_match(match, patterns):
                return match
        # Otherwise return the first match
        return matches[0]
    
    def _is_exact_match(self, column_name: str, patterns: List[str]) -> bool:
        """Check if column name is an exact match to any pattern."""
        col_lower = column_name.lower()
        for pattern in patterns:
            # Remove regex markers for exact comparison
            clean_pattern = pattern.replace(r'\b', '').replace('.*', '')\
                .replace(r'[\s_-]*', '')
            if col_lower == clean_pattern\
                or col_lower.replace('_', '') == clean_pattern.replace('_', ''):
                return True
        return False
    
    def get_supported_systems(self) -> List[str]:
        """Return list of supported coordinate systems."""
        return list(self.coordinate_patterns.keys())
    
    def add_coordinate_system(self, name: str, azimuthal_patterns: List[str], 
                            polar_patterns: List[str]) -> None:
        """
        Add a custom coordinate system pattern.
        
        Args:
            name: Name of the coordinate system
            azimuthal_patterns: List of regex patterns for azimuthal coordinate columns
            polar_patterns: List of regex patterns for polar coordinate columns
        """
        self.coordinate_patterns[name] = {
            'azimuthal': azimuthal_patterns,
            'polar': polar_patterns
        }
