#!/usr/bin/env python3
"""
Add downloads configuration to frontmatter of all notebook markdown files.
"""

import re
from pathlib import Path

def add_downloads_to_frontmatter(file_path):
    """Add downloads configuration to a markdown file's frontmatter."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Extract the frontmatter
    frontmatter_match = re.match(r'^---\n(.*?)\n---\n(.*)$', content, re.DOTALL)
    if not frontmatter_match:
        print(f"⚠️  No frontmatter found in {file_path}")
        return False

    frontmatter = frontmatter_match.group(1)
    body = frontmatter_match.group(2)

    # Check if downloads already exists
    if 'downloads:' in frontmatter:
        print(f"ℹ️  Downloads already configured in {file_path}")
        return False

    # Generate the ipynb filename
    stem = file_path.stem
    ipynb_file = f"notebooks/{stem}.ipynb"

    # Add downloads configuration
    downloads_config = f"\ndownloads:\n  - file: {ipynb_file}\n"
    new_frontmatter = frontmatter + downloads_config

    # Write back
    new_content = f"---\n{new_frontmatter}---\n{body}"
    with open(file_path, 'w') as f:
        f.write(new_content)

    print(f"✓ Added downloads to {file_path}")
    return True

def main():
    """Add downloads configuration to all notebook files."""
    print("=" * 60)
    print("Adding Downloads Configuration to Frontmatter")
    print("=" * 60)

    # Get all markdown notebook files
    files = []

    # Chapter files
    for i in range(1, 22):
        file = Path(f"chapter_{i:02d}.md")
        if file.exists():
            files.append(file)

    # Lab files
    for lab_file in ["chapter_04_lab.md", "chapter_05_lab.md"]:
        file = Path(lab_file)
        if file.exists():
            files.append(file)

    # Exercise files
    for exercise_file in ["chapter_04_exercises_a.md", "chapter_04_exercises_b.md"]:
        file = Path(exercise_file)
        if file.exists():
            files.append(file)

    # Appendix files
    for letter in ['a', 'b', 'c', 'd', 'e']:
        file = Path(f"appendix_{letter}.md")
        if file.exists():
            files.append(file)

    print(f"\nProcessing {len(files)} files...\n")

    updated = 0
    for file in files:
        if add_downloads_to_frontmatter(file):
            updated += 1

    print(f"\n✓ Updated {updated} files")
    print("=" * 60)

if __name__ == "__main__":
    main()
