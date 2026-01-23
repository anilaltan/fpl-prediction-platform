#!/usr/bin/env python3
"""
Directory structure scanner for FPL Prediction Platform
Scans the project directory and generates a comprehensive structure report.
"""
import os
from pathlib import Path

IGNORED_DIRS = {
    '.git', 'node_modules', '__pycache__', '.idea', '.vscode', 
    '.next', 'dist', 'build', '.pytest_cache', 'venv', 'env',
    '.venv', 'target', '*.egg-info'
}

IGNORED_FILES = {
    '.DS_Store', '*.pyc', '*.pyo', '*.pyd', '.env.local'
}

def should_ignore(path):
    """Check if a path should be ignored."""
    name = os.path.basename(path)
    
    # Check directory names
    if name in IGNORED_DIRS:
        return True
    
    # Check if it's a hidden file/directory (except specific ones we want)
    if name.startswith('.') and name not in ['.env', '.gitignore', '.dockerignore', '.taskmaster']:
        return True
    
    # Check file extensions
    for pattern in IGNORED_FILES:
        if pattern.startswith('*') and name.endswith(pattern[1:]):
            return True
        if name == pattern:
            return True
    
    return False

def get_file_size(path):
    """Get human-readable file size."""
    try:
        size = os.path.getsize(path)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}TB"
    except:
        return "?"

def scan_directory(root_path, prefix="", is_last=True, max_depth=5, current_depth=0):
    """Recursively scan directory structure."""
    if current_depth >= max_depth:
        return
    
    try:
        entries = sorted(os.listdir(root_path), key=lambda x: (not os.path.isdir(os.path.join(root_path, x)), x.lower()))
    except PermissionError:
        print(f"{prefix}[Permission denied] {os.path.basename(root_path)}")
        return
    
    # Filter and process entries
    valid_entries = []
    for entry in entries:
        full_path = os.path.join(root_path, entry)
        if not should_ignore(full_path):
            valid_entries.append((entry, full_path))
    
    for i, (entry, full_path) in enumerate(valid_entries):
        is_last_entry = i == len(valid_entries) - 1
        connector = "└── " if is_last_entry else "├── "
        
        if os.path.isdir(full_path):
            print(f"{prefix}{connector}{entry}/")
            extension = "    " if is_last_entry else "│   "
            scan_directory(full_path, prefix + extension, is_last_entry, max_depth, current_depth + 1)
        else:
            size = get_file_size(full_path)
            print(f"{prefix}{connector}{entry} ({size})")

def generate_summary(root_path):
    """Generate a summary of the directory structure."""
    stats = {
        'files': 0,
        'dirs': 0,
        'py_files': 0,
        'ts_files': 0,
        'tsx_files': 0,
        'js_files': 0,
        'total_size': 0
    }
    
    for root, dirs, files in os.walk(root_path):
        # Filter ignored directories
        dirs[:] = [d for d in dirs if not should_ignore(os.path.join(root, d))]
        
        for file in files:
            if should_ignore(os.path.join(root, file)):
                continue
            
            stats['files'] += 1
            file_path = os.path.join(root, file)
            try:
                stats['total_size'] += os.path.getsize(file_path)
            except:
                pass
            
            if file.endswith('.py'):
                stats['py_files'] += 1
            elif file.endswith('.ts'):
                stats['ts_files'] += 1
            elif file.endswith('.tsx'):
                stats['tsx_files'] += 1
            elif file.endswith('.js') and not file.endswith('.json'):
                stats['js_files'] += 1
        
        stats['dirs'] += len(dirs)
    
    return stats

if __name__ == "__main__":
    root = Path(__file__).parent
    print("=" * 80)
    print(f"FPL Prediction Platform - Directory Structure Scan")
    print("=" * 80)
    print(f"\nRoot: {root}\n")
    
    print("Directory Tree:")
    print("-" * 80)
    scan_directory(str(root))
    
    print("\n" + "-" * 80)
    print("Summary Statistics:")
    print("-" * 80)
    stats = generate_summary(str(root))
    print(f"Total Directories: {stats['dirs']}")
    print(f"Total Files: {stats['files']}")
    print(f"  - Python files (.py): {stats['py_files']}")
    print(f"  - TypeScript files (.ts): {stats['ts_files']}")
    print(f"  - TSX files (.tsx): {stats['tsx_files']}")
    print(f"  - JavaScript files (.js): {stats['js_files']}")
    
    total_size_gb = stats['total_size'] / (1024 ** 3)
    print(f"Total Size: {total_size_gb:.2f} GB")
    print("=" * 80)
