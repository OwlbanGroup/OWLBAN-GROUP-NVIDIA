#!/usr/bin/env python3
"""
Whitespace Fixing Utility for JPMorgan Financial APIs Project

This script automatically fixes common whitespace issues in Python files:
- Removes trailing whitespace
- Fixes inconsistent indentation
- Ensures proper blank lines between definitions
- Removes multiple consecutive blank lines
- Ensures files end with a single newline
- Fixes mixed tabs and spaces
- Removes whitespace-only lines
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import argparse


class WhitespaceFixer:
    """Utility class to fix whitespace issues in Python files"""

    def __init__(self, dry_run: bool = False, verbose: bool = False) -> None:
        self.dry_run = dry_run
        self.verbose = verbose
        self.stats = {
            'files_processed': 0,
            'files_modified': 0,
            'trailing_whitespace_removed': 0,
            'blank_lines_fixed': 0,
            'indentation_fixed': 0,
            'tabs_converted': 0,
            'eof_newlines_fixed': 0
        }

    def fix_trailing_whitespace(self, lines: List[str]) -> Tuple[List[str], int]:
        """Remove trailing whitespace from lines"""
        fixed_lines = []
        count = 0

        for line in lines:
            original = line
            fixed = line.rstrip() + '\n' if line.endswith('\n') else line.rstrip()
            if original != fixed and original.rstrip('\n').strip():  # Don't count blank lines
                count += 1
            fixed_lines.append(fixed)

        return fixed_lines, count

    def fix_tabs(self, lines: List[str]) -> Tuple[List[str], int]:
        """Convert tabs to spaces (4 spaces per tab)"""
        fixed_lines = []
        count = 0

        for line in lines:
            if '\t' in line:
                fixed = line.replace('\t', '    ')
                count += 1
                fixed_lines.append(fixed)
            else:
                fixed_lines.append(line)

        return fixed_lines, count

    def fix_blank_lines(self, lines: List[str]) -> Tuple[List[str], int]:
        """
        Fix blank line issues:
        - Remove multiple consecutive blank lines (max 2)
        - Remove whitespace-only lines
        - Ensure proper spacing around class/function definitions
        """
        fixed_lines = []
        count = 0
        consecutive_blanks = 0

        for line in lines:
            stripped = line.strip()

            # Convert whitespace-only lines to truly blank lines
            if not stripped and line != '\n':
                line = '\n'
                count += 1

            # Track consecutive blank lines
            if not stripped:
                consecutive_blanks += 1
                # Allow max 2 consecutive blank lines
                if consecutive_blanks <= 2:
                    fixed_lines.append(line)
                else:
                    count += 1
            else:
                consecutive_blanks = 0
                fixed_lines.append(line)

        return fixed_lines, count

    def ensure_eof_newline(self, lines: List[str]) -> Tuple[List[str], int]:
        """Ensure file ends with exactly one newline"""
        if not lines:
            return lines, 0

        count = 0

        # Remove trailing blank lines except one
        while len(lines) > 1 and not lines[-1].strip():
            lines.pop()
            count += 1

        # Ensure last line ends with newline
        if lines and not lines[-1].endswith('\n'):
            lines[-1] += '\n'
            count += 1

        return lines, count

    def fix_indentation(self, lines: List[str]) -> Tuple[List[str], int]:
        """Fix inconsistent indentation (ensure multiples of 4 spaces)"""
        fixed_lines = []
        count = 0

        for line in lines:
            if not line.strip():
                fixed_lines.append(line)
                continue

            # Count leading spaces
            leading_spaces = len(line) - len(line.lstrip(' '))

            # If indentation is not a multiple of 4, fix it
            if leading_spaces % 4 != 0:
                correct_indent = (leading_spaces // 4) * 4
                if leading_spaces % 4 >= 2:
                    correct_indent += 4
                fixed = ' ' * correct_indent + line.lstrip()
                if fixed != line:
                    count += 1
                    fixed_lines.append(fixed)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        return fixed_lines, count

    def fix_file(self, filepath: Path) -> bool:
        """Fix whitespace issues in a single file"""
        try:
            # Read file
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if not lines:
                return False

            original_content = ''.join(lines)

            # Apply fixes
            lines, trailing_count = self.fix_trailing_whitespace(lines)
            lines, tabs_count = self.fix_tabs(lines)
            lines, blank_count = self.fix_blank_lines(lines)
            lines, indent_count = self.fix_indentation(lines)
            lines, eof_count = self.ensure_eof_newline(lines)

            new_content = ''.join(lines)

            # Check if file was modified
            if original_content != new_content:
                self.stats['trailing_whitespace_removed'] += trailing_count
                self.stats['tabs_converted'] += tabs_count
                self.stats['blank_lines_fixed'] += blank_count
                self.stats['indentation_fixed'] += indent_count
                self.stats['eof_newlines_fixed'] += eof_count

                if not self.dry_run:
                    # Write fixed content
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(new_content)

                    if self.verbose:
                        print(f"✓ Fixed: {filepath}")  # noqa: W1309
                        if trailing_count:
                            print(f"  - Removed {trailing_count} trailing whitespace(s)")
                        if tabs_count:
                            print(f"  - Converted {tabs_count} tab(s) to spaces")
                        if blank_count:
                            print(f"  - Fixed {blank_count} blank line issue(s)")
                        if indent_count:
                            print(f"  - Fixed {indent_count} indentation issue(s)")
                        if eof_count:
                            print(f"  - Fixed EOF newline")
                else:
                    print(f"[DRY RUN] Would fix: {filepath}")  # noqa: W1309
                    if trailing_count:
                        print(f"  - Would remove {trailing_count} trailing whitespace(s)")
                    if tabs_count:
                        print(f"  - Would convert {tabs_count} tab(s) to spaces")
                    if blank_count:
                        print(f"  - Would fix {blank_count} blank line issue(s)")
                    if indent_count:
                        print(f"  - Would fix {indent_count} indentation issue(s)")
                    if eof_count:
                        print(f"  - Would fix EOF newline")

                return True

            return False

        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"✗ Error processing {filepath}: {e}", file=sys.stderr)
            return False

    def process_directory(self, directory: Path,
                            exclude_dirs: Optional[List[str]] = None) -> None:
        """Process all Python files in a directory recursively"""
        if exclude_dirs is None:
            exclude_dirs = [
                '__pycache__', '.git', '.mypy_cache', '.pytest_cache',
                'venv', 'env', '.venv', 'node_modules', '.vscode',
                'backups', 'logs', 'temp', 'models'
            ]

        print(f"\n{'='*60}")
        print(f"Processing directory: {directory}")
        print(f"{'='*60}\n")

        for root, dirs, files in os.walk(directory):
            # Remove excluded directories from search
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file.endswith('.py'):
                    filepath = Path(root) / file
                    self.stats['files_processed'] += 1

                    if self.fix_file(filepath):
                        self.stats['files_modified'] += 1

    def print_summary(self) -> None:
        """Print summary of fixes applied"""
        print(f"\n{'='*60}")
        print("WHITESPACE FIX SUMMARY")
        print(f"{'='*60}")
        print(f"Files processed:              {self.stats['files_processed']}")
        print(f"Files modified:               {self.stats['files_modified']}")
        print(f"Trailing whitespace removed:  {self.stats['trailing_whitespace_removed']}")
        print(f"Tabs converted to spaces:     {self.stats['tabs_converted']}")
        print(f"Blank lines fixed:            {self.stats['blank_lines_fixed']}")
        print(f"Indentation issues fixed:     {self.stats['indentation_fixed']}")
        print(f"EOF newlines fixed:           {self.stats['eof_newlines_fixed']}")
        print(f"{'='*60}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Fix whitespace issues in Python files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fix all Python files in current directory
    python fix_whitespace.py .

    # Dry run to see what would be changed
    python fix_whitespace.py . --dry-run

    # Fix specific directory with verbose output
    python fix_whitespace.py ./src --verbose

    # Fix specific file
    python fix_whitespace.py ./src/jpmorgan_client.py
        """
    )

    parser.add_argument(
        'path',
        nargs='?',
        default='.',
        help='Path to file or directory to process (default: current directory)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed information about fixes'
    )
    parser.add_argument(
        '--exclude',
        nargs='+',
        help='Additional directories to exclude'
    )

    args = parser.parse_args()

    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path '{path}' does not exist", file=sys.stderr)
        sys.exit(1)

    fixer = WhitespaceFixer(dry_run=args.dry_run, verbose=args.verbose)

    if path.is_file():
        if path.suffix == '.py':
            fixer.stats['files_processed'] = 1
            if fixer.fix_file(path):
                fixer.stats['files_modified'] = 1
        else:
            print(f"Error: '{path}' is not a Python file", file=sys.stderr)
            sys.exit(1)
    else:
        exclude_dirs = args.exclude if args.exclude else None
        fixer.process_directory(path, exclude_dirs)

    fixer.print_summary()

    if args.dry_run:
        print("Note: This was a dry run. No files were actually modified.")
        print("Run without --dry-run to apply changes.\n")


if __name__ == '__main__':
    main()
