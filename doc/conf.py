#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# example_vibroacoustics documentation build configuration file, created by
# sphinx-quickstart on Thu Mar 12 09:03:54 2020.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- General configuration ------------------------------------------------

html_logo = './_static/sfepy_logo.png'

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.imgmath',
    'sphinx.ext.extlinks',
    'sphinx_rtd_theme',
]
# extensions = ['sphinx.ext.intersphinx', 'sphinx.ext.mathjax']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'sfepy_example_vibroacoustics'
copyright = u'2021, Vladimír Lukeš, Eduard Rohan'
author = u'Vladimír Lukeš, Eduard Rohan'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = '1.0'
# The full version, including alpha/beta/rc tags.
release = '1.0'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

numfig = True

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_context = {
    # Enable the "Edit in GitHub link within the header of each page.
    # 'display_github': True,
    # Set the following variables to generate the resulting github URL for each page.
    # Format Template: https://{{ github_host|default("github.com") }}/{{ github_user }}/{{ github_repo }}/blob/{{ github_version }}{{ conf_py_path }}{{ pagename }}{{ suffix }}
    'github_user': 'sfepy',
    'github_repo': 'example_vibroacoustics',
    'menu_links_name': 'Connections',
    'menu_links': [
        ('<i class="fa fa-github fa-fw"></i> Source Code', 'https://github.com/sfepy/example_vibroacoustics'),
        ('<i class="fa fa-file-text fa-fw"></i> The Paper', 'https://arxiv.org/abs/2104.01367'),
        ('<i class="fa fa-external-link fa-fw"></i> SfePy', 'https://sfepy.org'),
    ],
}
html_copy_source = False
html_show_sourcelink = False

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}
html_theme_options = {
    'logo_only': True,
    'display_version': False,
    'navigation_depth': 4,
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_style = 'sfepy.css'

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
# html_sidebars = {}

# Output file base name for HTML help builder.
htmlhelp_basename = 'example_doc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
}

imgmath_latex_preamble = r"""
\usepackage{bm}
"""
