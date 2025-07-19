#!/usr/bin/env python3
"""
Test runner for dipole-utils tests.

This script provides a simple way to run the different test suites.
"""
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"❌ {description} failed with return code {result.returncode}")
        return False
    else:
        print(f"✅ {description} passed!")
        return True

def main():
    parser = argparse.ArgumentParser(description="Run dipole-utils tests")
    parser.add_argument(
        '--type', 
        choices=['unit', 'integration', 'all'], 
        default='all',
        help='Type of tests to run'
    )
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--coverage', 
        action='store_true',
        help='Run with coverage reporting'
    )
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path('tests').exists():
        print("❌ Error: 'tests' directory not found. Please run from the dipole-utils root directory.")
        sys.exit(1)
    
    # Base pytest command
    base_cmd = ['python', '-m', 'pytest']
    
    if args.verbose:
        base_cmd.append('-v')
    
    if args.coverage:
        base_cmd.extend(['--cov=dipoleutils', '--cov-report=html', '--cov-report=term'])
    
    success = True
    
    if args.type in ['unit', 'all']:
        cmd = base_cmd + ['tests/test_dipole_unit.py']
        success &= run_command(cmd, "Unit Tests")
    
    if args.type in ['integration', 'all']:
        cmd = base_cmd + ['tests/test_dipole_integration.py']
        success &= run_command(cmd, "Integration Tests")
    
    if args.type == 'all':
        print(f"\n{'='*50}")
        if success:
            print("🎉 All tests passed!")
        else:
            print("❌ Some tests failed. Check the output above.")
        print(f"{'='*50}")
    
    if args.coverage:
        print("\n📊 Coverage report generated in htmlcov/index.html")
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
