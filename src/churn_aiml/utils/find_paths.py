# -*- coding: utf-8 -*-
"""Finding paths in a project directory structure"""
# %% --------------------------------------------------------------------------
# * Author: Evgeni Nikolaev
# * Email: evgeni.v.nikolaev.ai@gmail.com
# %%
# UPDATED ON: 2025-07-05
# CREATED ON: 2025-06-16
# -----------------------------------------------------------------------------
# COPYRIGHT @ 2025 Ricoh. All rights reserved.
# The information contained herein is copyright and proprietary to
# Ricoh and may not be reproduced, disclosed, or used in
# any manner without prior written permission from Ricoh.
# -----------------------------------------------------------------------------
# %% --------------------------------------------------------------------------
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union
# %% --------------------------------------------------------------------------
class PathFinder(ABC):
    """ Abstract base class for path finding strategies."""

    @abstractmethod
    def find_path(
        self,
        filename: str,
        search_dir: Optional[Union[str, Path]] = None,
        max_depth: Optional[int] = None
    ) -> Optional[Path]:
        """ Find a file with the specified filename by searching recursively in the search directory.

            Args:
                filename: The name of the file to find (e.g., 'requirements.txt')
                search_dir: The directory to start searching from. If None, uses the current directory.
                max_depth: Maximum directory depth to search. None means no limit.
            Returns:
                Path object to the found file, or None if not found
        """
        pass

# %% --------------------------------------------------------------------------
class ProjectRootFinder(PathFinder):
    """PathFinder implementation for finding the project root directory."""

    def find_path(self, current_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """Find the project root directory by looking for setup.py or pyproject.toml.

            Args:
                current_path: The directory to start searching from bottom up.
                If None, uses the current directory.
            Returns:
                Path object to the ProjectRoot, or print a message if not found.
        """
        if current_path is None:
            current_path = Path(__file__).resolve().parent

        # Start from the directory containing this file and go up
        for directory in [current_path, *current_path.parents]:
            # Check for common project root indicators
            if (directory / "setup.py").exists() or (directory / "pyproject.toml").exists():
                return directory

        # If no project root found
        raise FileNotFoundError("Could not find project root (no setup.py or pyproject.toml found)")

# %% --------------------------------------------------------------------------
class FileFinder(PathFinder):
    """PathFinder implementation for finding files in a project directory."""

    def find_path(
            self,
            filename: str,
            search_dir: Optional[Union[str, Path]] = None,
            max_depth: Optional[int] = None
    ) -> Optional[Path]:
        """ Find a file with the specified filename by searching recursively in the search directory.

            Args:
                filename: The name of the file to find (e.g., 'requirements.txt')
                search_dir: The directory to start searching from top down. If None, uses the current directory.
                max_depth: Maximum directory depth to search. None means no limit.

            Returns:
                Path object to the found file, or None if not found
        """
        # Set default search directory to current directory if not specified
        if search_dir is None:
            search_dir = Path.cwd()
        else:
            search_dir = Path(search_dir)

        # Make sure the search directory exists
        if not search_dir.exists() or not search_dir.is_dir():
            raise ValueError(f"Search directory {search_dir} does not exist or is not a directory")

        # Use Path.rglob for simple cases when max_depth is None
        if max_depth is None:
            for file_path in search_dir.rglob(filename):
                if file_path.is_file(): # Make sure it's a file, not a directory
                    return file_path
            return None

        # For cases with max_depth, we need a custom search
        found_files = []

        # Define a recursive function to search with depth limit
        def search_with_depth(current_dir, current_depth):
            if current_depth > max_depth:
                return

            # Check for the file in the current directory
            for item in current_dir.iterdir():
                if item.is_file() and item.name == filename:
                    found_files.append(item)
                    return # Found the file, no need to search further

                # If it's a directory, search it recursively
                if item.is_dir():
                    search_with_depth(item, current_depth + 1)

                if found_files: # Stop if we've found the file
                    return

        # Start the search
        search_with_depth(search_dir, 0)

        # Return the first found file, or None if none were found
        return found_files[0] if found_files else None

# %% --------------------------------------------------------------------------
class DirectoryFinderTopDown(PathFinder):
    """PathFinder implementation for finding directories in a project directory."""

    def find_path(
            self,
            directory_name: str,
            search_dir: Optional[Union[str, Path]] = None,
            max_depth: Optional[int] = None
    ) -> Optional[Path]:
        """ Find a directory with the specified directory_name by searching recursively in the search directory.

            Args:
                directory_name: The name of the directory to find (e.g., 'src', 'tests')
                search_dir: The directory to start searching from top down. If None, uses the current directory.
                max_depth: Maximum directory depth to search. None means no limit.

            Returns:
                Path object to the found directory, or None if not found
        """
        # Set default search directory to current directory if not specified
        if search_dir is None:
            search_dir = Path.cwd()
        else:
            search_dir = Path(search_dir)

        # Make sure the search directory exists
        if not search_dir.exists() or not search_dir.is_dir():
            raise ValueError(f"Search directory {search_dir} does not exist or is not a directory")

        # Use Path.rglob for simple cases when max_depth is None
        if max_depth is None:
            for dir_path in search_dir.rglob(directory_name):
                if dir_path.is_dir(): # Make sure it's a directory, not a file
                    return dir_path
            return None

        # For cases with max_depth, we need a custom search
        found_directories = []

        # Define a recursive function to search with depth limit
        def search_with_depth(current_dir, current_depth):
            if current_depth > max_depth:
                return

            # Check for the directory in the current directory
            for item in current_dir.iterdir():
                if item.is_dir() and item.name == directory_name:
                    found_directories.append(item)
                    return # Found the directory, no need to search further

                # If it's a directory, search it recursively
                if item.is_dir():
                    search_with_depth(item, current_depth + 1)

                if found_directories: # Stop if we've found the directory
                    return

        # Start the search
        search_with_depth(search_dir, 0)

        # Return the first found directory, or None if none were found
        return found_directories[0] if found_directories else None
