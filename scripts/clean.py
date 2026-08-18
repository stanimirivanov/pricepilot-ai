"""Clean up cache files and temporary directories"""

import shutil
from pathlib import Path

import click
from loguru import logger


@click.command()
def clean():
    """Remove cache directories and temporary files"""
    project_root = Path(__file__).parent.parent
    removed_count = 0

    # Directories to remove
    cache_dirs = [
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "htmlcov",
        ".ipynb_checkpoints",
    ]

    # Remove directories
    for dir_name in cache_dirs:
        for path in project_root.rglob(dir_name):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                logger.info(f"Removed: {path.relative_to(project_root)}")
                removed_count += 1

    # Remove specific files
    files_to_remove = [".coverage", "coverage.xml"]
    for filename in files_to_remove:
        filepath = project_root / filename
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Removed: {filename}")
            removed_count += 1

    if removed_count == 0:
        logger.info("Nothing to clean. Project is already clean.")
    else:
        logger.info(f"Cleanup complete. Removed {removed_count} items.")


if __name__ == "__main__":
    clean()
