#!/usr/bin/env python3
"""
Generate sitemap.xml and robots.txt for the MyST Markdown book.
Parses myst.yml to extract all pages and creates SEO files.
"""

import yaml
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


def extract_pages_from_toc(toc: List[Dict[str, Any]], pages: List[str] = None) -> List[str]:
    """
    Recursively extract all page files from the table of contents.

    Args:
        toc: Table of contents structure from myst.yml
        pages: Accumulator list for pages

    Returns:
        List of page paths (without .md extension)
    """
    if pages is None:
        pages = []

    for item in toc:
        # Add the file if it exists
        if 'file' in item:
            # Remove .md extension and add to pages
            page = item['file'].replace('.md', '')
            pages.append(page)

        # Recursively process children
        if 'children' in item:
            extract_pages_from_toc(item['children'], pages)

    return pages


def generate_sitemap(base_url: str, output_path: str = '_build/html/sitemap.xml'):
    """
    Generate sitemap.xml file.

    Args:
        base_url: Base URL of the site (e.g., https://username.github.io/repo-name)
        output_path: Path where sitemap.xml will be written
    """
    # Read myst.yml
    with open('myst.yml', 'r') as f:
        config = yaml.safe_load(f)

    # Extract pages from TOC
    toc = config.get('project', {}).get('toc', [])
    pages = extract_pages_from_toc(toc)

    # Ensure base_url doesn't end with /
    base_url = base_url.rstrip('/')

    # Get current date for lastmod
    today = datetime.now().strftime('%Y-%m-%d')

    # Generate sitemap XML
    sitemap_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]

    # Add each page to sitemap
    for page in pages:
        # Convert page path to URL
        # index.md -> /
        # chapter_01.md -> /chapter_01.html
        if page == 'index':
            url = f'{base_url}/'
        else:
            url = f'{base_url}/{page}.html'

        sitemap_lines.extend([
            '  <url>',
            f'    <loc>{url}</loc>',
            f'    <lastmod>{today}</lastmod>',
            '    <changefreq>weekly</changefreq>',
            '    <priority>0.8</priority>',
            '  </url>',
        ])

    sitemap_lines.append('</urlset>')

    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write sitemap.xml
    with open(output_path, 'w') as f:
        f.write('\n'.join(sitemap_lines))

    print(f'✓ Generated sitemap.xml with {len(pages)} pages')
    print(f'✓ Output: {output_path}')
    print(f'✓ Base URL: {base_url}')


def generate_robots_txt(base_url: str, output_path: str = '_build/html/robots.txt'):
    """
    Generate robots.txt file.

    Args:
        base_url: Base URL of the site (e.g., https://username.github.io/repo-name)
        output_path: Path where robots.txt will be written
    """
    # Ensure base_url doesn't end with /
    base_url = base_url.rstrip('/')

    # Generate robots.txt content
    robots_lines = [
        '# robots.txt for Learn Probability',
        '',
        'User-agent: *',
        'Allow: /',
        '',
        f'Sitemap: {base_url}/sitemap.xml',
    ]

    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write robots.txt
    with open(output_path, 'w') as f:
        f.write('\n'.join(robots_lines))

    print(f'✓ Generated robots.txt')
    print(f'✓ Output: {output_path}')
    print(f'✓ Sitemap URL: {base_url}/sitemap.xml')


def main():
    """Main entry point."""
    # Get base URL from environment or use default
    repo_name = os.environ.get('GITHUB_REPOSITORY', 'learn_probability_fork').split('/')[-1]
    github_pages_url = os.environ.get('GITHUB_PAGES_URL', '')

    if github_pages_url:
        # Use the full GitHub Pages URL if provided
        base_url = github_pages_url.rstrip('/')
    else:
        # Construct from repository name
        base_url = f'https://snowch.github.io/{repo_name}'

    # Allow override via BASE_URL environment variable
    base_url = os.environ.get('BASE_URL_FULL', base_url)

    generate_sitemap(base_url)
    generate_robots_txt(base_url)


if __name__ == '__main__':
    main()
