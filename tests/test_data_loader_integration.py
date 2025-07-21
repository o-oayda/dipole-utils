import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from astropy.table import Table
import numpy as np

from dipoleutils.utils.data_loader import (
    DataLoader, 
    ConfigManager, 
    CatalogueValidationError,
    _config_manager
)


class TestDataLoaderIntegration:
    """Integration tests for the DataLoader class using real files and directories."""
    
    def setup_method(self):
        """Set up test fixtures with temporary directories and files."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, 'data')
        self.external_dir = os.path.join(self.temp_dir, 'external')
        
        # Create directory structure
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.external_dir, exist_ok=True)
        
        # Create test FITS files
        self._create_test_fits_files()
        
        # Mock the config manager to use our temp directory
        self.original_config_manager = _config_manager
        self.test_config_manager = ConfigManager()
        self.test_config_manager.config_dir = Path(self.temp_dir) / '.dipole-utils'
        self.test_config_manager.config_file = self.test_config_manager.config_dir / 'config.json'
        
    def teardown_method(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def _create_test_fits_files(self):
        """Create test FITS files for testing."""
        # Create single-file catalogue (milliquas)
        # From file_names.py: 'milliquas': 'milliquas.fits'
        milliquas_dir = os.path.join(self.data_dir, 'milliquas')
        os.makedirs(milliquas_dir, exist_ok=True)
        
        # Create a simple test table
        test_data = Table({
            'ra': np.random.uniform(0, 360, 100),
            'dec': np.random.uniform(-90, 90, 100),
            'mag': np.random.uniform(10, 20, 100)
        })
        
        milliquas_file = os.path.join(milliquas_dir, 'milliquas.fits')
        test_data.write(milliquas_file, overwrite=True)
        
        # Create multi-file catalogue (quaia)
        quaia_dir = os.path.join(self.data_dir, 'quaia', 'high')
        os.makedirs(quaia_dir, exist_ok=True)
        
        catalogue_data = Table({
            'source_id': np.arange(50),
            'ra': np.random.uniform(0, 360, 50),
            'dec': np.random.uniform(-90, 90, 50),
            'G_mag': np.random.uniform(10, 21, 50)
        })
        
        selection_data = Table({
            'healpix_id': np.arange(100),
            'selection_function': np.random.uniform(0, 1, 100)
        })
        
        catalogue_file = os.path.join(quaia_dir, 'quaia_G20.5.fits')
        selection_file = os.path.join(quaia_dir, 'selection_function_NSIDE64_G20.5.fits')
        
        catalogue_data.write(catalogue_file, overwrite=True)
        selection_data.write(selection_file, overwrite=True)
        
        # Create external drive structure for testing copy functionality
        self._create_external_drive_structure()
        
    def _create_external_drive_structure(self):
        """Create external drive structure with test files."""
        # Create milliquas on external drive
        external_milliquas_dir = os.path.join(self.external_dir, 'milliquas')
        os.makedirs(external_milliquas_dir, exist_ok=True)
        
        external_test_data = Table({
            'ra': np.random.uniform(0, 360, 200),
            'dec': np.random.uniform(-90, 90, 200),
            'mag': np.random.uniform(10, 20, 200)
        })
        
        external_milliquas_file = os.path.join(external_milliquas_dir, 'milliquas.fits')
        external_test_data.write(external_milliquas_file, overwrite=True)
        
        # Create quaia files on external drive
        external_quaia_dir = os.path.join(self.external_dir, 'quaia', 'low')
        os.makedirs(external_quaia_dir, exist_ok=True)
        
        external_catalogue_data = Table({
            'source_id': np.arange(75),
            'ra': np.random.uniform(0, 360, 75),
            'dec': np.random.uniform(-90, 90, 75),
            'G_mag': np.random.uniform(10, 20, 75)
        })
        
        external_selection_data = Table({
            'healpix_id': np.arange(150),
            'selection_function': np.random.uniform(0, 1, 150)
        })
        
        external_catalogue_file = os.path.join(external_quaia_dir, 'quaia_G20.0.fits')
        external_selection_file = os.path.join(external_quaia_dir, 'selection_function_NSIDE64_G20.0.fits')
        
        external_catalogue_data.write(external_catalogue_file, overwrite=True)
        external_selection_data.write(external_selection_file, overwrite=True)
        
    @patch('dipoleutils.utils.data_loader._config_manager')
    def test_configure_and_use_data_directory(self, mock_config_manager):
        """Test configuring data directory and using it to load files."""
        mock_config_manager.set_data_directory = self.test_config_manager.set_data_directory
        mock_config_manager.get_data_directory = self.test_config_manager.get_data_directory
        
        # Ensure config directory exists
        self.test_config_manager._ensure_config_dir()
        
        # Configure the data directory
        loader = DataLoader('milliquas')
        loader.configure_directory(self.data_dir)
        
        # Verify configuration was saved
        assert self.test_config_manager.get_data_directory() == os.path.abspath(self.data_dir)
        
        # Load the file
        result = loader.load()
        
        # Verify we got a Table back with the expected structure
        assert isinstance(result, Table)
        assert len(result) == 100
        assert 'ra' in result.colnames
        assert 'dec' in result.colnames
        assert 'mag' in result.colnames
        
    @patch('dipoleutils.utils.data_loader._config_manager')
    @patch('builtins.input')  # Mock input to prevent stdin issues
    def test_load_single_file_catalogue_integration(self, mock_input, mock_config_manager):
        """Test loading a single-file catalogue end-to-end."""
        mock_config_manager.get_data_directory.return_value = self.data_dir
        mock_input.return_value = 'n'  # Decline to copy if file is missing
        
        loader = DataLoader('milliquas')
        result = loader.load()
        
        assert isinstance(result, Table)
        assert len(result) == 100
        assert set(result.colnames) == {'ra', 'dec', 'mag'}
        
    @patch('dipoleutils.utils.data_loader._config_manager')
    def test_load_multi_file_catalogue_integration(self, mock_config_manager):
        """Test loading a multi-file catalogue end-to-end."""
        mock_config_manager.get_data_directory.return_value = self.data_dir
        
        loader = DataLoader('quaia', 'high')
        result = loader.load()
        
        assert isinstance(result, dict)
        assert 'catalogue' in result
        assert 'selection_function' in result
        
        catalogue = result['catalogue']
        selection_function = result['selection_function']
        
        assert isinstance(catalogue, Table)
        assert isinstance(selection_function, Table)
        
        assert len(catalogue) == 50
        assert len(selection_function) == 100
        
        assert set(catalogue.colnames) == {'source_id', 'ra', 'dec', 'G_mag'}
        assert set(selection_function.colnames) == {'healpix_id', 'selection_function'}
        
    @patch('dipoleutils.utils.data_loader._config_manager')
    def test_copy_from_external_drive_single_file(self, mock_config_manager):
        """Test copying single file from external drive."""
        mock_config_manager.get_data_directory.return_value = self.data_dir
        
        # Remove the milliquas file to simulate it being missing
        milliquas_file = os.path.join(self.data_dir, 'milliquas', 'milliquas.fits')
        os.remove(milliquas_file)
        
        loader = DataLoader('milliquas')
        
        # Mock the external drive path to point to our test structure
        with patch.object(loader, '_get_cat_path_in_external_drive') as mock_external_path:
            mock_external_path.return_value = os.path.join(self.external_dir, 'milliquas', 'milliquas.fits')
            
            # Mock user input to confirm copy
            with patch('builtins.input', side_effect=['y', 'y']) as mock_input:
                with patch('builtins.print'):  # Suppress print output
                    loader.copy_from_external_drive()
            
            # Verify file was copied
            assert os.path.exists(milliquas_file)
            
            # Verify copied file has correct data
            copied_table = Table.read(milliquas_file)
            assert len(copied_table) == 200  # External version has 200 rows
            
    @patch('dipoleutils.utils.data_loader._config_manager')
    def test_copy_from_external_drive_multi_file(self, mock_config_manager):
        """Test copying multi-file catalogue from external drive."""
        mock_config_manager.get_data_directory.return_value = self.data_dir
        
        loader = DataLoader('quaia', 'low')  # This variant doesn't exist in our data_dir
        
        # Mock the external drive paths
        with patch.object(loader, '_get_external_paths_for_multi_file_catalogue') as mock_external_paths:
            mock_external_paths.return_value = {
                'catalogue': os.path.join(self.external_dir, 'quaia', 'low', 'quaia_G20.0.fits'),
                'selection_function': os.path.join(self.external_dir, 'quaia', 'low', 'selection_function_NSIDE64_G20.0.fits')
            }
            
            # Mock user input to confirm copy
            with patch('builtins.input', return_value='y') as mock_input:
                with patch('builtins.print'):  # Suppress print output
                    loader.copy_from_external_drive()
            
            # Verify files were copied
            quaia_low_dir = os.path.join(self.data_dir, 'quaia', 'low')
            catalogue_file = os.path.join(quaia_low_dir, 'quaia_G20.0.fits')
            selection_file = os.path.join(quaia_low_dir, 'selection_function_NSIDE64_G20.0.fits')
            
            assert os.path.exists(catalogue_file)
            assert os.path.exists(selection_file)
            
            # Verify copied files have correct data
            copied_catalogue = Table.read(catalogue_file)
            copied_selection = Table.read(selection_file)
            
            assert len(copied_catalogue) == 75
            assert len(copied_selection) == 150
            
    @patch('dipoleutils.utils.data_loader._config_manager')
    def test_copy_operation_cancelled(self, mock_config_manager):
        """Test that copy operation can be cancelled."""
        mock_config_manager.get_data_directory.return_value = self.data_dir
        
        # Remove the milliquas file
        milliquas_file = os.path.join(self.data_dir, 'milliquas', 'milliquas.fits')
        os.remove(milliquas_file)
        
        loader = DataLoader('milliquas')
        
        with patch.object(loader, '_get_cat_path_in_external_drive') as mock_external_path:
            mock_external_path.return_value = os.path.join(self.external_dir, 'milliquas', 'milliquas.fits')
            
            # Mock user input to cancel copy (only one 'n' needed)
            with patch('builtins.input', return_value='n') as mock_input:
                with patch('builtins.print'):
                    loader.copy_from_external_drive()
            
            # Verify file was not copied
            assert not os.path.exists(milliquas_file)
            
    @patch('dipoleutils.utils.data_loader._config_manager')
    def test_file_not_found_on_external_drive(self, mock_config_manager):
        """Test handling when file is not found on external drive."""
        mock_config_manager.get_data_directory.return_value = self.data_dir
        
        # First remove the existing file
        milliquas_file = os.path.join(self.data_dir, 'milliquas', 'milliquas.fits')
        os.remove(milliquas_file)
        
        loader = DataLoader('milliquas')
        
        # Mock external path to non-existent file
        with patch.object(loader, '_get_cat_path_in_external_drive') as mock_external_path:
            mock_external_path.return_value = '/nonexistent/path/milliquas.fits'
            
            with patch('builtins.print'):
                with pytest.raises(FileNotFoundError, match="Source file does not exist"):
                    loader._copy_single_file_catalogue()
                    
    @patch('dipoleutils.utils.data_loader._config_manager')
    def test_load_with_missing_file_decline_copy(self, mock_config_manager):
        """Test loading when file is missing and user declines to copy."""
        mock_config_manager.get_data_directory.return_value = self.data_dir
        
        # Remove the milliquas file (correct path without Data subdirectory)
        milliquas_file = os.path.join(self.data_dir, 'milliquas', 'milliquas.fits')
        os.remove(milliquas_file)
        
        loader = DataLoader('milliquas')
        
        # Mock user input to decline copy
        with patch('builtins.input', return_value='n'):
            with patch('builtins.print'):
                with pytest.raises(FileNotFoundError, match="Catalogue file not found"):
                    loader.load()
                    
    @patch('dipoleutils.utils.data_loader._config_manager')
    def test_load_multi_file_with_partial_missing_files(self, mock_config_manager):
        """Test loading multi-file catalogue with some files missing."""
        mock_config_manager.get_data_directory.return_value = self.data_dir
        
        # Remove one of the quaia files
        selection_file = os.path.join(self.data_dir, 'quaia', 'high', 'selection_function_NSIDE64_G20.5.fits')
        os.remove(selection_file)
        
        loader = DataLoader('quaia', 'high')
        
        # Mock external paths
        with patch.object(loader, '_get_external_paths_for_multi_file_catalogue') as mock_external_paths:
            mock_external_paths.return_value = {
                'catalogue': os.path.join(self.external_dir, 'quaia', 'high', 'quaia_G20.5.fits'),
                'selection_function': os.path.join(self.external_dir, 'quaia', 'high', 'selection_function_NSIDE64_G20.5.fits')
            }
            
            # Create the missing external files for this test
            external_quaia_high_dir = os.path.join(self.external_dir, 'quaia', 'high')
            os.makedirs(external_quaia_high_dir, exist_ok=True)
            
            # Create external selection function file
            external_selection_data = Table({
                'healpix_id': np.arange(100),
                'selection_function': np.random.uniform(0, 1, 100)
            })
            external_selection_file = os.path.join(external_quaia_high_dir, 'selection_function_NSIDE64_G20.5.fits')
            external_selection_data.write(external_selection_file, overwrite=True)
            
            # Mock user input to decline copy
            with patch('builtins.input', return_value='n'):
                with patch('builtins.print'):
                    with pytest.raises(FileNotFoundError, match="Missing files for quaia"):
                        loader.load()
                        
    def test_different_file_formats(self):
        """Test loading different file formats (CSV, DAT, FITS)."""
        # Create test CSV file
        csv_dir = os.path.join(self.data_dir, 'test_csv')
        os.makedirs(csv_dir, exist_ok=True)
        csv_file = os.path.join(csv_dir, 'test.csv')
        
        test_data = Table({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        test_data.write(csv_file, format='csv', overwrite=True)
        
        # Create test DAT file
        dat_dir = os.path.join(self.data_dir, 'test_dat')
        os.makedirs(dat_dir, exist_ok=True)
        dat_file = os.path.join(dat_dir, 'test.dat')
        test_data.write(dat_file, format='ascii', overwrite=True)
        
        # Test each format
        with patch('dipoleutils.utils.data_loader._config_manager') as mock_config_manager:
            mock_config_manager.get_data_directory.return_value = self.data_dir
            
            loader = DataLoader('milliquas')  # Using milliquas as base, we'll override the file loading
            
            # Test CSV
            csv_result = loader._load_file(csv_file)
            assert isinstance(csv_result, Table)
            assert len(csv_result) == 5
            
            # Test DAT
            dat_result = loader._load_file(dat_file)
            assert isinstance(dat_result, Table)
            assert len(dat_result) == 5
            
            # Test FITS (already tested above, but for completeness)
            fits_file = os.path.join(self.data_dir, 'milliquas', 'milliquas.fits')
            fits_result = loader._load_file(fits_file)
            assert isinstance(fits_result, Table)
            assert len(fits_result) == 100


class TestConfigManagerIntegration:
    """Integration tests for ConfigManager with real file system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager()
        # Override config directory to use temp directory
        self.config_manager.config_dir = Path(self.temp_dir) / '.dipole-utils'
        self.config_manager.config_file = self.config_manager.config_dir / 'config.json'
        
    def teardown_method(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def test_full_config_workflow(self):
        """Test complete configuration workflow with real files."""
        # Create a test data directory
        test_data_dir = os.path.join(self.temp_dir, 'test_data')
        os.makedirs(test_data_dir)
        
        # Initially no config should exist
        assert self.config_manager.get_data_directory() is None
        
        # Ensure config directory exists
        self.config_manager._ensure_config_dir()
        
        # Set data directory
        self.config_manager.set_data_directory(test_data_dir)
        
        # Verify config file was created and contains correct data
        assert self.config_manager.config_file.exists()
        
        with open(self.config_manager.config_file, 'r') as f:
            config_data = json.load(f)
        
        assert config_data['data_directory'] == os.path.abspath(test_data_dir)
        
        # Verify we can retrieve the directory
        retrieved_dir = self.config_manager.get_data_directory()
        assert retrieved_dir == os.path.abspath(test_data_dir)
        
        # Clear the configuration
        self.config_manager.clear_data_directory()
        assert self.config_manager.get_data_directory() is None
        
        # Verify config file still exists but without data_directory
        with open(self.config_manager.config_file, 'r') as f:
            config_data = json.load(f)
        
        assert 'data_directory' not in config_data
        
    def test_config_persistence_across_instances(self):
        """Test that configuration persists across ConfigManager instances."""
        # Create a test data directory
        test_data_dir = os.path.join(self.temp_dir, 'persistent_data')
        os.makedirs(test_data_dir)
        
        # Ensure config directory exists
        self.config_manager._ensure_config_dir()
        
        # Set config with first instance
        self.config_manager.set_data_directory(test_data_dir)
        
        # Create new instance with same config directory
        new_config_manager = ConfigManager()
        new_config_manager.config_dir = Path(self.temp_dir) / '.dipole-utils'
        new_config_manager.config_file = new_config_manager.config_dir / 'config.json'
        
        # Verify new instance can read the configuration
        retrieved_dir = new_config_manager.get_data_directory()
        assert retrieved_dir == os.path.abspath(test_data_dir)
        
    def test_corrupted_config_handling(self):
        """Test handling of corrupted configuration files."""
        # Create config directory and write invalid JSON
        self.config_manager._ensure_config_dir()
        
        with open(self.config_manager.config_file, 'w') as f:
            f.write('invalid json content')
        
        # Should handle corrupted config gracefully
        assert self.config_manager.get_data_directory() is None
        
        # Should be able to set new configuration
        test_data_dir = os.path.join(self.temp_dir, 'recovery_data')
        os.makedirs(test_data_dir)
        
        self.config_manager.set_data_directory(test_data_dir)
        assert self.config_manager.get_data_directory() == os.path.abspath(test_data_dir)
