import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open, call
from astropy.table import Table
import numpy as np

from dipoleutils.utils.data_loader import (
    DataLoader, 
    ConfigManager, 
    CatalogueValidationError, 
    validate_catalogue_and_variant,
    _config_manager
)


class TestCatalogueValidation:
    """Unit tests for catalogue and variant validation."""
    
    def test_validate_existing_catalogue_no_variant(self):
        """Test validation with existing catalogue that has no variants."""
        # milliquas is a string in fn_dict, so no variants
        validate_catalogue_and_variant('milliquas', None)  # Should not raise
        
    def test_validate_existing_catalogue_with_variant(self):
        """Test validation with existing catalogue that has variants."""
        # quaia has variants, so should require one
        validate_catalogue_and_variant('quaia', 'high')  # Should not raise
        
    def test_validate_nonexistent_catalogue(self):
        """Test validation with nonexistent catalogue."""
        with pytest.raises(CatalogueValidationError, match="Catalogue 'nonexistent' is not recognised"):
            validate_catalogue_and_variant('nonexistent')
            
    def test_validate_catalogue_with_invalid_variant(self):
        """Test validation with invalid variant."""
        with pytest.raises(CatalogueValidationError, match="Variant 'invalid' is not recognised"):
            validate_catalogue_and_variant('quaia', 'invalid')
            
    def test_validate_catalogue_no_variants_but_variant_provided(self):
        """Test validation when variant is provided for catalogue that has no variants."""
        with pytest.raises(CatalogueValidationError, match="does not have variants"):
            validate_catalogue_and_variant('milliquas', 'some_variant')
            
    def test_validate_catalogue_requires_variant_but_none_provided(self):
        """Test validation when catalogue requires variant but none provided."""
        with pytest.raises(CatalogueValidationError, match="requires a variant"):
            validate_catalogue_and_variant('quaia', None)


class TestConfigManager:
    """Unit tests for the ConfigManager class."""
    
    def setup_method(self):
        """Set up test fixtures with temporary directories."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager()
        # Override config directory to use temp directory
        self.config_manager.config_dir = Path(self.temp_dir) / '.dipole-utils'
        self.config_manager.config_file = self.config_manager.config_dir / 'config.json'
        
    def teardown_method(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def test_ensure_config_dir_creates_directory(self):
        """Test that config directory is created if it doesn't exist."""
        assert not self.config_manager.config_dir.exists()
        self.config_manager._ensure_config_dir()
        assert self.config_manager.config_dir.exists()
        
    def test_get_data_directory_no_config_file(self):
        """Test getting data directory when no config file exists."""
        result = self.config_manager.get_data_directory()
        assert result is None
        
    def test_get_data_directory_empty_config(self):
        """Test getting data directory with empty config."""
        self.config_manager._ensure_config_dir()
        with open(self.config_manager.config_file, 'w') as f:
            json.dump({}, f)
            
        result = self.config_manager.get_data_directory()
        assert result is None
        
    def test_set_and_get_data_directory(self):
        """Test setting and getting data directory."""
        test_dir = tempfile.mkdtemp()
        try:
            self.config_manager._ensure_config_dir()  # Ensure config directory exists
            self.config_manager.set_data_directory(test_dir)
            result = self.config_manager.get_data_directory()
            assert result == os.path.abspath(test_dir)
        finally:
            shutil.rmtree(test_dir)
            
    def test_set_data_directory_nonexistent_path(self):
        """Test setting data directory with nonexistent path."""
        with pytest.raises(ValueError, match="Directory does not exist"):
            self.config_manager.set_data_directory("/nonexistent/path")
            
    def test_clear_data_directory(self):
        """Test clearing data directory configuration."""
        test_dir = tempfile.mkdtemp()
        try:
            # Set a directory first
            self.config_manager._ensure_config_dir()  # Ensure config directory exists
            self.config_manager.set_data_directory(test_dir)
            assert self.config_manager.get_data_directory() == os.path.abspath(test_dir)
            
            # Clear it
            self.config_manager.clear_data_directory()
            assert self.config_manager.get_data_directory() is None
        finally:
            shutil.rmtree(test_dir)
            
    def test_clear_data_directory_no_config_file(self):
        """Test clearing data directory when no config file exists."""
        # Should not raise an error
        self.config_manager.clear_data_directory()


class TestDataLoaderUnit:
    """Unit tests for the DataLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def test_initialization_valid_catalogue_no_variant(self):
        """Test DataLoader initialization with valid catalogue that has no variants."""
        loader = DataLoader('milliquas')
        assert loader.catalogue_name == 'milliquas'
        assert loader.catalogue_variant is None
        
    def test_initialization_valid_catalogue_with_variant(self):
        """Test DataLoader initialization with valid catalogue and variant."""
        loader = DataLoader('quaia', 'high')
        assert loader.catalogue_name == 'quaia'
        assert loader.catalogue_variant == 'high'
        
    def test_initialization_invalid_catalogue(self):
        """Test DataLoader initialization with invalid catalogue."""
        with pytest.raises(CatalogueValidationError):
            DataLoader('nonexistent')
            
    def test_initialization_invalid_variant(self):
        """Test DataLoader initialization with invalid variant."""
        with pytest.raises(CatalogueValidationError):
            DataLoader('quaia', 'invalid')
            
    @patch.object(_config_manager, 'set_data_directory')
    def test_configure_directory(self, mock_set_data_directory):
        """Test configuring data directory."""
        loader = DataLoader('milliquas')
        test_path = '/test/path'
        
        loader.configure_directory(test_path)
        mock_set_data_directory.assert_called_once_with(test_path)
        
    @patch.object(_config_manager, 'get_data_directory')
    @patch('os.path.exists')
    def test_get_data_directory_configured(self, mock_exists, mock_get_data_directory):
        """Test getting configured data directory."""
        mock_get_data_directory.return_value = '/configured/path'
        mock_exists.return_value = True
        
        loader = DataLoader('milliquas')
        result = loader.get_data_directory()
        
        assert result == '/configured/path'
        
    @patch.object(_config_manager, 'get_data_directory')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('os.path.expanduser')
    def test_get_data_directory_fallback(self, mock_expanduser, mock_makedirs, mock_exists, mock_get_data_directory):
        """Test getting data directory when none configured (fallback to default)."""
        mock_get_data_directory.return_value = None
        mock_exists.return_value = False
        mock_expanduser.return_value = '/home/user/catalogue_data'
        
        loader = DataLoader('milliquas')
        result = loader.get_data_directory()
        
        assert result == '/home/user/catalogue_data'
        mock_makedirs.assert_called_once_with('/home/user/catalogue_data', exist_ok=True)
        
    def test_get_file_name_single_file_catalogue(self):
        """Test getting file name for single-file catalogue."""
        loader = DataLoader('milliquas')
        result = loader.get_file_name()
        assert result == 'milliquas.fits'
        
    def test_get_file_name_multi_file_catalogue(self):
        """Test getting file name for multi-file catalogue."""
        loader = DataLoader('quaia', 'high')
        result = loader.get_file_name()
        expected = {
            'catalogue': 'quaia_G20.5.fits',
            'selection_function': 'selection_function_NSIDE64_G20.5.fits'
        }
        assert result == expected
        
    def test_is_multi_file_catalogue_single_file(self):
        """Test checking if catalogue is multi-file for single-file catalogue."""
        loader = DataLoader('milliquas')
        assert not loader._is_multi_file_catalogue()
        
    def test_is_multi_file_catalogue_multi_file(self):
        """Test checking if catalogue is multi-file for multi-file catalogue."""
        loader = DataLoader('quaia', 'high')
        assert loader._is_multi_file_catalogue()
        
    @patch('os.path.exists')
    def test_load_file_fits(self, mock_exists):
        """Test loading FITS file."""
        loader = DataLoader('milliquas')
        test_file = '/test/path/file.fits'
        
        with patch.object(Table, 'read') as mock_read:
            mock_table = Mock(spec=Table)
            mock_read.return_value = mock_table
            
            result = loader._load_file(test_file)
            
            mock_read.assert_called_once_with(test_file)
            assert result == mock_table
            
    @patch('os.path.exists')
    def test_load_file_dat(self, mock_exists):
        """Test loading DAT file."""
        loader = DataLoader('milliquas')
        test_file = '/test/path/file.dat'
        
        with patch.object(Table, 'read') as mock_read:
            mock_table = Mock(spec=Table)
            mock_read.return_value = mock_table
            
            result = loader._load_file(test_file)
            
            mock_read.assert_called_once_with(test_file, format='ascii')
            assert result == mock_table
            
    @patch('os.path.exists')
    def test_load_file_csv(self, mock_exists):
        """Test loading CSV file."""
        loader = DataLoader('milliquas')
        test_file = '/test/path/file.csv'
        
        with patch.object(Table, 'read') as mock_read:
            mock_table = Mock(spec=Table)
            mock_read.return_value = mock_table
            
            result = loader._load_file(test_file)
            
            mock_read.assert_called_once_with(test_file, format='csv')
            assert result == mock_table
            
    def test_load_file_unsupported_format(self):
        """Test loading unsupported file format."""
        loader = DataLoader('milliquas')
        test_file = '/test/path/file.xyz'
        
        with pytest.raises(NotImplementedError, match="File type \\(.xyz\\) not implemented"):
            loader._load_file(test_file)
            
    def test_get_catalogue_dir_no_variant(self):
        """Test getting catalogue directory for catalogue without variant."""
        loader = DataLoader('milliquas')
        result = loader._get_catalogue_dir()
        assert result == 'milliquas'
        
    def test_get_catalogue_dir_with_variant(self):
        """Test getting catalogue directory for catalogue with variant."""
        loader = DataLoader('quaia', 'high')
        result = loader._get_catalogue_dir()
        assert result == 'quaia/high'
        
    @patch.object(DataLoader, 'get_data_directory')
    @patch.object(DataLoader, 'get_file_name')
    def test_get_cat_path_in_config_dir(self, mock_get_file_name, mock_get_data_directory):
        """Test getting catalogue path in config directory."""
        mock_get_data_directory.return_value = '/data'
        mock_get_file_name.return_value = 'test_file.fits'
        
        loader = DataLoader('milliquas')
        result = loader._get_cat_path_in_config_dir()
        
        assert result == '/data/milliquas/test_file.fits'
        
    @patch('platform.system')
    @patch.object(DataLoader, 'get_file_name')
    def test_get_cat_path_in_external_drive_linux(self, mock_get_file_name, mock_platform):
        """Test getting catalogue path on external drive for Linux."""
        mock_platform.return_value = 'Linux'
        mock_get_file_name.return_value = 'test_file.fits'
        
        loader = DataLoader('milliquas')
        result = loader._get_cat_path_in_external_drive()
        
        expected = '/media/oliver/NICE-DRIVE/research_data/surveys/milliquas/test_file.fits'
        assert result == expected
        
    @patch('platform.system')
    @patch.object(DataLoader, 'get_file_name')
    def test_get_cat_path_in_external_drive_macos(self, mock_get_file_name, mock_platform):
        """Test getting catalogue path on external drive for macOS."""
        mock_platform.return_value = 'Darwin'
        mock_get_file_name.return_value = 'test_file.fits'
        
        loader = DataLoader('milliquas')
        result = loader._get_cat_path_in_external_drive()
        
        expected = '/Volumes/NICE-DRIVE/research_data/surveys/milliquas/test_file.fits'
        assert result == expected
        
    def test_get_file_paths_for_multi_file_catalogue_single_file_raises(self):
        """Test that getting file paths for single-file catalogue raises error."""
        loader = DataLoader('milliquas')
        with pytest.raises(ValueError, match="This method should only be called for multi-file catalogues"):
            loader._get_file_paths_for_multi_file_catalogue()
            
    @patch.object(DataLoader, 'get_data_directory')
    def test_get_file_paths_for_multi_file_catalogue(self, mock_get_data_directory):
        """Test getting file paths for multi-file catalogue."""
        mock_get_data_directory.return_value = '/data'
        
        loader = DataLoader('quaia', 'high')
        result = loader._get_file_paths_for_multi_file_catalogue()
        
        expected = {
            'catalogue': '/data/quaia/high/quaia_G20.5.fits',
            'selection_function': '/data/quaia/high/selection_function_NSIDE64_G20.5.fits'
        }
        assert result == expected
        
    def test_get_external_paths_for_multi_file_catalogue_single_file_raises(self):
        """Test that getting external paths for single-file catalogue raises error."""
        loader = DataLoader('milliquas')
        with pytest.raises(ValueError, match="This method should only be called for multi-file catalogues"):
            loader._get_external_paths_for_multi_file_catalogue()
            
    @patch('platform.system')
    def test_get_external_paths_for_multi_file_catalogue(self, mock_platform):
        """Test getting external paths for multi-file catalogue."""
        mock_platform.return_value = 'Darwin'  # macOS
        
        loader = DataLoader('quaia', 'high')
        result = loader._get_external_paths_for_multi_file_catalogue()
        
        expected = {
            'catalogue': '/Volumes/NICE-DRIVE/research_data/surveys/quaia/high/quaia_G20.5.fits',
            'selection_function': '/Volumes/NICE-DRIVE/research_data/surveys/quaia/high/selection_function_NSIDE64_G20.5.fits'
        }
        assert result == expected
        
    def test_download_not_implemented(self):
        """Test that download method raises NotImplementedError."""
        loader = DataLoader('milliquas')
        with pytest.raises(NotImplementedError):
            loader.download()


class TestDataLoaderLoadMethods:
    """Unit tests for DataLoader load methods with mocked file operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_table = Mock(spec=Table)
        
    @patch.object(DataLoader, '_is_multi_file_catalogue')
    @patch.object(DataLoader, '_load_single_file_catalogue')
    @patch.object(DataLoader, '_load_multi_file_catalogue')
    def test_load_single_file(self, mock_load_multi, mock_load_single, mock_is_multi):
        """Test load method for single-file catalogue."""
        mock_is_multi.return_value = False
        mock_load_single.return_value = self.mock_table
        
        loader = DataLoader('milliquas')
        result = loader.load()
        
        mock_load_single.assert_called_once()
        mock_load_multi.assert_not_called()
        assert result == self.mock_table
        
    @patch.object(DataLoader, '_is_multi_file_catalogue')
    @patch.object(DataLoader, '_load_single_file_catalogue')
    @patch.object(DataLoader, '_load_multi_file_catalogue')
    def test_load_multi_file(self, mock_load_multi, mock_load_single, mock_is_multi):
        """Test load method for multi-file catalogue."""
        mock_is_multi.return_value = True
        mock_result = {'catalogue': self.mock_table, 'selection_function': Mock(spec=Table)}
        mock_load_multi.return_value = mock_result
        
        loader = DataLoader('quaia', 'high')
        result = loader.load()
        
        mock_load_multi.assert_called_once()
        mock_load_single.assert_not_called()
        assert result == mock_result
        
    @patch('os.path.exists')
    @patch.object(DataLoader, '_get_cat_path_in_config_dir')
    @patch.object(DataLoader, '_load_file')
    def test_load_single_file_catalogue_exists(self, mock_load_file, mock_get_path, mock_exists):
        """Test loading single-file catalogue when file exists."""
        mock_get_path.return_value = '/data/milliquas/milliquas.fits'
        mock_exists.return_value = True
        mock_load_file.return_value = self.mock_table
        
        loader = DataLoader('milliquas')
        
        with patch('builtins.print') as mock_print:
            result = loader._load_single_file_catalogue()
            
        mock_load_file.assert_called_once_with('/data/milliquas/milliquas.fits')
        mock_print.assert_called_with('Loading file: /data/milliquas/milliquas.fits')
        assert result == self.mock_table
        
    @patch('os.path.exists')
    @patch.object(DataLoader, '_get_cat_path_in_config_dir')
    @patch.object(DataLoader, 'copy_from_external_drive')
    @patch.object(DataLoader, '_load_file')  # Mock the file loading
    @patch('builtins.input')
    def test_load_single_file_catalogue_missing_copy_yes(self, mock_input, mock_load_file, mock_copy, mock_get_path, mock_exists):
        """Test loading single-file catalogue when file missing and user chooses to copy."""
        mock_get_path.return_value = '/data/milliquas/milliquas.fits'
        mock_exists.side_effect = [False, True]  # First call returns False (missing), second True (copied)
        mock_input.return_value = 'y'
        mock_load_file.return_value = self.mock_table

        loader = DataLoader('milliquas')

        with patch('builtins.print'):
            result = loader._load_single_file_catalogue()
            
        mock_copy.assert_called_once()
        mock_load_file.assert_called_once()
        assert result == self.mock_table
        
    @patch('os.path.exists')
    @patch.object(DataLoader, '_get_cat_path_in_config_dir')
    @patch('builtins.input')
    def test_load_single_file_catalogue_missing_copy_no(self, mock_input, mock_get_path, mock_exists):
        """Test loading single-file catalogue when file missing and user chooses not to copy."""
        mock_get_path.return_value = '/data/milliquas/milliquas.fits'
        mock_exists.return_value = False
        mock_input.return_value = 'n'
        
        loader = DataLoader('milliquas')
        
        with patch('builtins.print'):
            with pytest.raises(FileNotFoundError, match="Catalogue file not found"):
                loader._load_single_file_catalogue()
