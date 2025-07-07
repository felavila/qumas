# Configuration file for the Sphinx documentation builder.
#
# This file configures the Read the Docs server that continuously
# builds the documentation pages of the quma project.

import os
import sys
import sphinx_rtd_theme

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute.
sys.path.insert(0, os.path.abspath('..'))


def get_templated_vars():
    return type(
        'TemplatedVariables',
        (),
        dict(
            project_slug='quma',
            package_name='quma',
            author_name='Felipe Avila Vera',
            year='2025',
            version='0.0.1',
            github_username='favila',
            repo_name='quma',
        ),
    )


variables = get_templated_vars()


# -- Project information -----------------------------------------------------

project = variables.project_slug
author = variables.author_name
copyright = f"{variables.year}, {variables.author_name}"
release = variables.version


# -- General configuration ---------------------------------------------------

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
}

# The master toctree document.
master_doc = 'index'

# Sphinx extensions
extensions = [
    'sphinx.ext.autodoc',      # Pull in Python docstrings
    'sphinx.ext.autosummary',  # Generate summary tables + stub pages
    'sphinx.ext.napoleon',     # Google / NumPy style docstrings
    'sphinx.ext.viewcode',     # “View source” links
    'sphinx.ext.coverage',
    'sphinx.ext.extlinks',     # External Links helper
    'sphinx.ext.ifconfig',
    'sphinx.ext.todo',
    'sphinx.ext.doctest',
    'sphinxcontrib.spelling',
    "autoapi.extension"
    "sphinx_rtd_theme"
]

autoapi_type = 'python'
autoapi_dirs = ['../quma']
# Where in your docs tree to put the generated API pages:
autoapi_root = 'api'

# Which members to include:
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
]

# Automatically generate autosummary stub pages
autosummary_generate = True

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Mock imports for modules not needed during docs build
autodoc_mock_imports = [
    'pandas',
    'lmfit',
    'astropy',
    'uncertainties',
]

# Paths for templates and excluded patterns
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# Use the Read the Docs theme
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
}

# Add any paths that contain custom static files (such as style sheets)
html_static_path = ['_static']
# html_logo = '_static/logo.png'  # Uncomment if you add a logo


# -- External Links Configuration -------------------------------------------

# Enables :issue:`123` links to GitHub issues
extlinks = {
    'issue': (
        'https://github.com/{username}/{repository}/issues/'.format(
            username=variables.github_username,
            repository=variables.repo_name,
        ) + '%s',
        'issue ',
    ),
}


# -- Todo and spelling options -----------------------------------------------

todo_include_todos = True
spelling_warning = True
