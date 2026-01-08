#!/usr/bin/env python3
"""
Generate .ipynb files from MyST Markdown notebook files.
This script converts all chapter, lab, exercise, and appendix markdown files
to Jupyter notebook format and creates a downloadable zip archive.
"""

import os
import shutil
import zipfile
from pathlib import Path
import jupytext

def get_notebook_files():
    """Get list of all markdown files that should be converted to notebooks."""
    notebook_files = []

    # Chapter files
    for i in range(1, 22):
        file = f"chapter_{i:02d}.md"
        if os.path.exists(file):
            notebook_files.append(file)

    # Lab files
    for lab_file in ["chapter_04_lab.md", "chapter_05_lab.md"]:
        if os.path.exists(lab_file):
            notebook_files.append(lab_file)

    # Exercise files
    for exercise_file in ["chapter_04_exercises_a.md", "chapter_04_exercises_b.md"]:
        if os.path.exists(exercise_file):
            notebook_files.append(exercise_file)

    # Appendix files
    for letter in ['a', 'b', 'c', 'd', 'e']:
        file = f"appendix_{letter}.md"
        if os.path.exists(file):
            notebook_files.append(file)

    return notebook_files

def convert_md_to_ipynb(md_file, output_dir):
    """Convert a single markdown file to ipynb format."""
    try:
        # Read the markdown file
        notebook = jupytext.read(md_file)

        # Generate output filename
        output_file = output_dir / f"{Path(md_file).stem}.ipynb"

        # Write the ipynb file
        jupytext.write(notebook, output_file, fmt='ipynb')

        print(f"✓ Converted {md_file} -> {output_file}")
        return output_file
    except Exception as e:
        print(f"✗ Error converting {md_file}: {e}")
        return None

def create_zip_archive(notebook_dir, zip_path):
    """Create a zip archive of all notebook files."""
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in sorted(notebook_dir.glob("*.ipynb")):
                zipf.write(file, file.name)
                print(f"  Added {file.name} to archive")
        print(f"✓ Created archive: {zip_path}")
        return True
    except Exception as e:
        print(f"✗ Error creating archive: {e}")
        return False

def main():
    """Main function to generate all notebooks and archive."""
    print("=" * 60)
    print("Generating Jupyter Notebooks from MyST Markdown")
    print("=" * 60)

    # Create output directory in project root (not _build)
    output_dir = Path("notebooks")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all notebook files
    notebook_files = get_notebook_files()
    print(f"\nFound {len(notebook_files)} notebook files to convert\n")

    # Convert each file
    converted_files = []
    for md_file in notebook_files:
        result = convert_md_to_ipynb(md_file, output_dir)
        if result:
            converted_files.append(result)

    print(f"\n✓ Successfully converted {len(converted_files)} notebooks")

    # Create zip archive in project root
    if converted_files:
        print("\nCreating zip archive...")
        zip_path = Path("learn_probability_notebooks.zip")
        create_zip_archive(output_dir, zip_path)

        # Calculate total size
        total_size = sum(f.stat().st_size for f in converted_files)
        zip_size = zip_path.stat().st_size
        print(f"\nTotal notebooks size: {total_size / 1024:.1f} KB")
        print(f"Archive size: {zip_size / 1024:.1f} KB")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
