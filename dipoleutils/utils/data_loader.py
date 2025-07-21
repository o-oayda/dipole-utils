from typing import Optional, Union
import os
import json
from pathlib import Path
from .file_names import fn_dict
from astropy.table import Table
import platform
import shutil
import sys

class CatalogueValidationError(ValueError):
    """Raised when catalogue name or variant validation fails."""
    pass

def validate_catalogue_and_variant(
        catalogue_name: str,
        catalogue_variant: Optional[str] = None
    ) -> None:
    """
    Validate that the catalogue name and variant exist in the file names dictionary.
    
    Args:
        catalogue_name: Name of the catalogue
        catalogue_variant: Variant of the catalogue (optional)
        
    Raises:
        CatalogueValidationError: If catalogue name or variant is invalid
    """
    # Check if catalogue exists
    if catalogue_name not in fn_dict:
        available_catalogues = list(fn_dict.keys())
        raise CatalogueValidationError(
            f"Catalogue '{catalogue_name}' is not recognised. "
            f"Available catalogues: {available_catalogues}"
        )
    
    catalogue_entry = fn_dict[catalogue_name]
    
    # If catalogue entry is a string, it has no variants
    if isinstance(catalogue_entry, str):
        if catalogue_variant is not None:
            raise CatalogueValidationError(
                f"Catalogue '{catalogue_name}' does not have variants, "
                f"but variant '{catalogue_variant}' was specified. "
                f"Use catalogue_variant=None or omit the parameter."
            )
    
    # If catalogue entry is a dict, it has variants
    elif isinstance(catalogue_entry, dict):
        if catalogue_variant is None:
            available_variants = list(catalogue_entry.keys())
            raise CatalogueValidationError(
                f"Catalogue '{catalogue_name}' requires a variant. "
                f"Available variants: {available_variants}"
            )
        
        if catalogue_variant not in catalogue_entry:
            available_variants = list(catalogue_entry.keys())
            raise CatalogueValidationError(
                f"Variant '{catalogue_variant}' is not recognised for catalogue '{catalogue_name}'. "
                f"Available variants: {available_variants}"
            )
    
    else:
        # This shouldn't happen with proper fn_dict structure, but handle it gracefully
        raise CatalogueValidationError(
            f"Invalid catalogue entry structure for '{catalogue_name}'"
        )

class ConfigManager:
    """Manages persistent configuration for dipole-utils package."""
    
    def __init__(self):
        self.config_dir = Path.home() / '.dipole-utils'
        self.config_file = self.config_dir / 'config.json'
        self._ensure_config_dir()
    
    def _ensure_config_dir(self) -> None:
        """Create configuration directory if it doesn't exist."""
        self.config_dir.mkdir(exist_ok=True)
    
    def get_data_directory(self) -> Optional[str]:
        """Get the configured data directory path."""
        if not self.config_file.exists():
            return None
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                return config.get('data_directory')
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    def set_data_directory(self, path: str) -> None:
        """Set and persist the data directory path."""
        # Validate that the path exists
        if not os.path.exists(path):
            raise ValueError(f"Directory does not exist: {path}")
        
        # Convert to absolute path
        abs_path = os.path.abspath(path)
        
        # Load existing config or create new one
        config = {}
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                config = {}
        
        # Update config
        config['data_directory'] = abs_path
        
        # Save config
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def clear_data_directory(self) -> None:
        """Clear the configured data directory."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                if 'data_directory' in config:
                    del config['data_directory']
                
                with open(self.config_file, 'w') as f:
                    json.dump(config, f, indent=2)
            except (json.JSONDecodeError, FileNotFoundError):
                pass

# Global configuration manager instance
_config_manager = ConfigManager()

class DataLoader:
    '''
    Object for loading in and downloading catalogues.
    
    The data directory can be configured using configure_directory() and will
    persist across sessions. If no directory is configured, methods will use
    the current working directory.
    
    Supports both single-file catalogues (most catalogues) and multi-file catalogues
    (like quaia which has both catalogue and selection function files).
    
    Args:
        catalogue_name: Name of the catalogue (must exist in fn_dict)
        catalogue_variant: Variant of the catalogue (required for catalogues with variants)
        
    Raises:
        CatalogueValidationError: If catalogue name or variant is invalid
    '''
    def __init__(self, catalogue_name: str, catalogue_variant: Optional[str] = None) -> None:
        # Validate catalogue and variant before setting attributes
        validate_catalogue_and_variant(catalogue_name, catalogue_variant)
        
        self.catalogue_name = catalogue_name
        self.catalogue_variant = catalogue_variant
        self._config_manager = _config_manager
    
    def configure_directory(self, absolute_path: str) -> None:
        """
        Configure the directory where catalogue files are stored.
        This setting will persist across sessions.
        
        Args:
            absolute_path: Absolute path to the directory containing catalogue files
            
        Raises:
            ValueError: If the directory doesn't exist
        """
        self._config_manager.set_data_directory(absolute_path)
        print(f"Data directory configured: {absolute_path}")
    
    def get_data_directory(self) -> str:
        """
        Get the configured data directory.
        
        Returns:
            The configured data directory path, or ~/catalogue_data if none configured
        """
        configured_dir = self._config_manager.get_data_directory()
        if configured_dir and os.path.exists(configured_dir):
            return configured_dir
        else:
            # Fall back to ~/catalogue_data
            default_dir = os.path.expanduser("~/catalogue_data")
            # Create the directory if it doesn't exist
            os.makedirs(default_dir, exist_ok=True)
            return default_dir
    
    def clear_directory_config(self) -> None:
        """Clear the persistent directory configuration."""
        self._config_manager.clear_data_directory()
        print("Data directory configuration cleared")

    def _load_file(self, full_path: str) -> Table:
        file_extension = full_path.split('.')[-1]

        if file_extension == 'fits':
            return Table.read(full_path)
        else:
            return self._to_fits_table(full_path, file_extension)

    def _to_fits_table(self, full_path: str, extension: str) -> Table:
        if extension == 'dat':
            return Table.read(full_path, format='ascii')
        elif extension == 'csv':
            return Table.read(full_path, format='csv')
        else:
            raise NotImplementedError(
                f'File type (.{extension}) not implemented.'
            )

    def load(self) -> Union[Table, dict[str, Table]]:
        """
        Load catalogue data from the configured directory.
        
        Returns:
            For single-file catalogues: Table object
            For multi-file catalogues (like quaia): dict with keys 'catalogue' and 'selection_function' containing respective Tables
        """
        if self._is_multi_file_catalogue():
            return self._load_multi_file_catalogue()
        else:
            return self._load_single_file_catalogue()
    
    def _load_single_file_catalogue(self) -> Table:
        """Load a single-file catalogue."""
        full_path = self._get_cat_path_in_config_dir()
        
        if not os.path.exists(full_path):
            print(f"File not found: {full_path}")
            confirm = input(
                f"Copy {self.catalogue_name} from the external drive (y/n)? "
            ).strip().lower()
            if confirm == 'y':
                self.copy_from_external_drive()
            else:
                raise FileNotFoundError(
                    f"Catalogue file not found at {full_path}"
                )

        print(f"Loading file: {full_path}")
        return self._load_file(full_path)
    
    def _load_multi_file_catalogue(self) -> dict[str, Table]:
        """Load a multi-file catalogue (like quaia with cat and sel files)."""
        file_paths = self._get_file_paths_for_multi_file_catalogue()
        
        # Check which files exist and which are missing
        missing_files = []
        existing_files = []
        
        for key, path in file_paths.items():
            if os.path.exists(path):
                existing_files.append((key, path))
            else:
                missing_files.append((key, path))
        
        # Handle missing files
        if missing_files:
            print("Missing files:")
            for key, path in missing_files:
                print(f"  {key}: {path}")
            
            if existing_files:
                print("\nExisting files:")
                for key, path in existing_files:
                    print(f"  {key}: {path}")
            
            copy_missing = input(
                f"Copy missing {self.catalogue_name} files from external drive (y/n)? "
            ).strip().lower()
            
            if copy_missing == 'y':
                self.copy_from_external_drive()
            else:
                raise FileNotFoundError(
                    f"Missing files for {self.catalogue_name}: {[key for key, _ in missing_files]}"
                )
        
        # Load all files
        tables = {}
        print(f"Loading {self.catalogue_name} files:")
        for key, path in file_paths.items():
            print(f"  Loading {key}: {path}")
            tables[key] = self._load_file(path)
        
        return tables

    def _get_cat_path_in_config_dir(self) -> str:
        data_dir = self.get_data_directory()
        catalogue_dir = self._get_catalogue_dir()
        path_to_containing_dir = f'{data_dir}/{catalogue_dir}'

        file_name = self.get_file_name()
        full_path = path_to_containing_dir + f'/{file_name}'
        
        return full_path
    
    def _get_catalogue_dir(self):
        catalogue_dir = f'{self.catalogue_name}'
        if self.catalogue_variant:
            catalogue_dir += f'/{self.catalogue_variant}'
        return catalogue_dir

    def _get_cat_path_in_external_drive(self):
        if platform.system() == 'Linux':
            path_header = '/media/oliver'
        else:
            # assume mac os
            path_header = '/Volumes'
        
        file_name = self.get_file_name()
        catalogue_dir = self._get_catalogue_dir()
        drive_path = 'NICE-DRIVE/research_data/surveys'
        cat_path = f'{path_header}/{drive_path}/{catalogue_dir}/{file_name}'

        return cat_path
        
    def copy_from_external_drive(self):
        """
        Copy catalogue files from external drive to configured directory with a progress bar.
        Handles both single-file and multi-file catalogues.
        """
        if self._is_multi_file_catalogue():
            self._copy_multi_file_catalogue()
        else:
            self._copy_single_file_catalogue()
    
    def _copy_single_file_catalogue(self):
        """Copy a single-file catalogue."""
        intended_full_path = self._get_cat_path_in_config_dir()
        if os.path.exists(intended_full_path):
            raise FileExistsError(
                f"File already exists at {intended_full_path}"
            )

        external_path_to_file = self._get_cat_path_in_external_drive()
        if not os.path.exists(external_path_to_file):
            raise FileNotFoundError(
                f"Source file does not exist: {external_path_to_file}"
            )

        # Show file size information
        file_size_mb = os.path.getsize(external_path_to_file) / (1024 * 1024)
        print(f"Source: {external_path_to_file}")
        print(f"Destination: {intended_full_path}")
        print(f"File size: {file_size_mb:.1f} MB")
        
        confirm = input('Confirm (y/n)? ').strip().lower()
        if confirm != 'y':
            print("Copy operation cancelled.")
            return

        os.makedirs(os.path.dirname(intended_full_path), exist_ok=True)
        self._progress_bar_copy(external_path_to_file, intended_full_path)
    
    def _copy_multi_file_catalogue(self):
        """Copy a multi-file catalogue (like quaia with cat and sel files)."""
        internal_paths = self._get_file_paths_for_multi_file_catalogue()
        external_paths = self._get_external_paths_for_multi_file_catalogue()
        
        # Check which files exist and which need to be copied
        files_to_copy = []
        files_already_exist = []
        missing_external_files = []
        
        for key in internal_paths.keys():
            internal_path = internal_paths[key]
            external_path = external_paths[key]
            
            if os.path.exists(internal_path):
                files_already_exist.append((key, internal_path))
            elif not os.path.exists(external_path):
                missing_external_files.append((key, external_path))
            else:
                files_to_copy.append((key, external_path, internal_path))
        
        # Handle missing external files
        if missing_external_files:
            print("Cannot find source files on external drive:")
            for key, path in missing_external_files:
                print(f"  {key}: {path}")
            raise FileNotFoundError(
                f"Missing source files for {self.catalogue_name}: {[key for key, _ in missing_external_files]}"
            )
        
        # Show status
        if files_already_exist:
            print("Files already exist (will skip):")
            for key, path in files_already_exist:
                print(f"  {key}: {path}")
        
        if not files_to_copy:
            print("All files already exist. Nothing to copy.")
            return
        
        # Show files to be copied with sizes
        print(f"\nFiles to copy for {self.catalogue_name}:")
        total_size_mb = 0
        for key, external_path, internal_path in files_to_copy:
            file_size_mb = os.path.getsize(external_path) / (1024 * 1024)
            total_size_mb += file_size_mb
            print(f"  {key}: {file_size_mb:.1f} MB")
            print(f"    From: {external_path}")
            print(f"    To:   {internal_path}")
        
        print(f"\nTotal size: {total_size_mb:.1f} MB")
        
        confirm = input('Confirm copying these files (y/n)? ').strip().lower()
        if confirm != 'y':
            print("Copy operation cancelled.")
            return
        
        # Copy each file
        for i, (key, external_path, internal_path) in enumerate(files_to_copy, 1):
            print(f"\n[{i}/{len(files_to_copy)}] Copying {key} file...")
            os.makedirs(os.path.dirname(internal_path), exist_ok=True)
            self._progress_bar_copy(external_path, internal_path)
        
        print(f"\n✅ Successfully copied all {self.catalogue_name} files!")

    def _progress_bar_copy(self,
            external_path: str,
            internal_path: str,
        ) -> None:
        """
        Alternative high-performance copy implementation using shutil.copyfile
        with a progress callback. This can be faster for very large files.
        """
        import time
        
        total_size = os.path.getsize(external_path)
        copied = [0]  # Use list to allow modification in nested function
        start_time = time.time()
        last_update = [0]
        
        def progress_callback(chunk):
            copied[0] += len(chunk)
            # Update every 10MB or 1% progress
            if copied[0] - last_update[0] >= min(10 * 1024 * 1024, total_size // 100):
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    speed_mbps = (copied[0] / (1024 * 1024)) / elapsed_time
                    done = int(50 * copied[0] / total_size)
                    percent = 100 * copied[0] / total_size
                    mb_copied = copied[0] / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    bar = (
                        f"[{'=' * done}{' ' * (50 - done)}] {percent:5.1f}%"
                        f"({mb_copied:.1f}/{total_mb:.1f} MB) {speed_mbps:.1f} MB/s"
                    )
                    print(f"\rCopying: {bar}", end='', flush=True)
                    last_update[0] = copied[0]
        
        # This uses the OS-optimized copy functions when available
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(internal_path), exist_ok=True)
            
            # For Python 3.8+ we can use a more efficient approach
            if sys.version_info >= (3, 8):
                # Use shutil.copyfile which is optimized at the OS level
                with open(external_path, 'rb') as src, open(internal_path, 'wb') as dst:
                    while True:
                        chunk = src.read(16 * 1024 * 1024)  # 16MB chunks
                        if not chunk:
                            break
                        dst.write(chunk)
                        progress_callback(chunk)
            else:
                # Fallback for older Python versions
                shutil.copyfile(external_path, internal_path)
                progress_callback(b'x' * total_size)  # Fake progress for completion
            
            # Show completion
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                avg_speed = (total_size / (1024 * 1024)) / elapsed_time
                total_mb = total_size / (1024 * 1024)
                bar = (
                    f"[{'=' * 50}] 100.0% ({total_mb:.1f}/{total_mb:.1f} MB)"
                    f"Avg: {avg_speed:.1f} MB/s"
                )
                print(f"\rCopying: {bar}")
            print()
            
            # Copy metadata
            shutil.copystat(external_path, internal_path)
            print(f"Successfully copied file from {external_path} to {internal_path}")
            
        except Exception as e:
            print(f"\nCopy failed: {e}")
            # Clean up partial file
            if os.path.exists(internal_path):
                os.remove(internal_path)
            raise

    def download(self) -> NotImplementedError:
        """Download catalogue files to the configured directory."""
        raise NotImplementedError

    def get_file_name(self) -> str | dict:
        """
        Get the file name or file dictionary for this catalogue.
        
        Returns:
            For catalogues without variants: the filename string
            For catalogues with variants: the variant's file entry (could be string or dict)
        """
        catalogue_entry = fn_dict[self.catalogue_name]

        if isinstance(catalogue_entry, str):
            # Catalogue has no variants, return the filename directly
            return catalogue_entry
        else:    
            # Catalogue has variants, return the specific variant entry
            # self.catalogue_variant is guaranteed to be valid due to validation in __init__
            return catalogue_entry[self.catalogue_variant]
    
    def _is_multi_file_catalogue(self) -> bool:
        """Check if this catalogue has multiple files (like quaia with catalogue and selection_function)."""
        file_entry = self.get_file_name()
        return isinstance(file_entry, dict) and 'catalogue' in file_entry and 'selection_function' in file_entry
    
    def _get_file_paths_for_multi_file_catalogue(self) -> dict[str, str]:
        """Get paths for all files in a multi-file catalogue."""
        if not self._is_multi_file_catalogue():
            raise ValueError("This method should only be called for multi-file catalogues")
        
        file_entry = self.get_file_name()
        if not isinstance(file_entry, dict):
            raise ValueError("Expected dict for multi-file catalogue")
            
        data_dir = self.get_data_directory()
        catalogue_dir = self._get_catalogue_dir()
        path_to_containing_dir = f'{data_dir}/{catalogue_dir}'
        
        paths = {}
        for key, filename in file_entry.items():
            paths[key] = f'{path_to_containing_dir}/{filename}'
        
        return paths
    
    def _get_external_paths_for_multi_file_catalogue(self) -> dict[str, str]:
        """Get external drive paths for all files in a multi-file catalogue."""
        if not self._is_multi_file_catalogue():
            raise ValueError("This method should only be called for multi-file catalogues")
        
        file_entry = self.get_file_name()
        if not isinstance(file_entry, dict):
            raise ValueError("Expected dict for multi-file catalogue")
        
        if platform.system() == 'Linux':
            path_header = '/media/oliver'
        else:
            # assume mac os
            path_header = '/Volumes'
        
        catalogue_dir = self._get_catalogue_dir()
        drive_path = 'NICE-DRIVE/research_data/surveys'
        
        paths = {}
        for key, filename in file_entry.items():
            paths[key] = f'{path_header}/{drive_path}/{catalogue_dir}/{filename}'
        
        return paths