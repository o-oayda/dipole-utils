"""
Test to demonstrate that DataLoader tests work without external drive dependency.

This test demonstrates that all external drive operations are properly mocked 
and don't require an actual external drive to be connected.
"""
import pytest
import os
import tempfile
from unittest.mock import patch
from dipoleutils.utils.data_loader import DataLoader


def test_external_drive_independence():
    """
    Demonstrate that DataLoader tests don't depend on external drive being connected.
    This test shows that external drive operations are properly mocked.
    """
    
    # This should work without any external drive being connected
    loader = DataLoader('milliquas')
    
    # Test that external drive path generation works (but doesn't require actual drive)
    with patch('platform.system', return_value='Darwin'):
        external_path = loader._get_cat_path_in_external_drive()
        expected_path = '/Volumes/NICE-DRIVE/research_data/surveys/milliquas/milliquas.fits'
        assert external_path == expected_path
    
    with patch('platform.system', return_value='Linux'):
        external_path = loader._get_cat_path_in_external_drive()
        expected_path = '/media/oliver/NICE-DRIVE/research_data/surveys/milliquas/milliquas.fits'
        assert external_path == expected_path
    
    # Test multi-file catalogue external paths
    quaia_loader = DataLoader('quaia', 'high')
    
    with patch('platform.system', return_value='Darwin'):
        external_paths = quaia_loader._get_external_paths_for_multi_file_catalogue()
        expected_paths = {
            'catalogue': '/Volumes/NICE-DRIVE/research_data/surveys/quaia/high/quaia_G20.5.fits',
            'selection_function': '/Volumes/NICE-DRIVE/research_data/surveys/quaia/high/selection_function_NSIDE64_G20.5.fits'
        }
        assert external_paths == expected_paths
    
    # Test that copy operations can be tested without external drive
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, 'data')
        external_dir = os.path.join(temp_dir, 'external')
        os.makedirs(data_dir)
        os.makedirs(external_dir)
        
        # Mock config manager
        with patch('dipoleutils.utils.data_loader._config_manager') as mock_config:
            mock_config.get_data_directory.return_value = data_dir
            
            # Mock external path to point to our test directory
            with patch.object(loader, '_get_cat_path_in_external_drive') as mock_external:
                fake_external_path = os.path.join(external_dir, 'fake_file.fits')
                mock_external.return_value = fake_external_path
                
                # This should handle the case where external file doesn't exist
                with patch('builtins.print'):
                    with pytest.raises(FileNotFoundError, match="Source file does not exist"):
                        loader._copy_single_file_catalogue()


if __name__ == "__main__":
    test_external_drive_independence()
    print("✅ All external drive independence tests passed!")
    print("Tests can run successfully without any external drive connected.")
