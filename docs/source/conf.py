# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join('..', '..', '..', 'motornet')))
from motornet import __version__ as version

# Workaround for issue https://github.com/sphinx-contrib/googleanalytics/issues/2
# Note that a warning still will be issued "unsupported object from its setup() function"
# Remove this workaround when the issue has been resolved upstream
import sphinx.application
import sphinx.errors
sphinx.application.ExtensionError = sphinx.errors.ExtensionError

# -- Project information -----------------------------------------------------

project = 'MotorNet'
copyright = '2022, Olivier Codol'
author = 'Olivier Codol'

# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'm2r2',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'nbsphinx',
    'sphinxcontrib.googleanalytics',
]

googleanalytics_id = "G-NSCHXS3MTN"

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'
# html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['custom.css']


def autodoc_skip_member(app, what, name, obj, skip, options):
    # from https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#skipping-members
    exclude = callable(obj) is False and what != "module"
    return True if exclude else None


def setup(app):
    app.connect('autodoc-skip-member', autodoc_skip_member)
