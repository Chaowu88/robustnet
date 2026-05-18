# Configuration file for the Sphinx documentation builder.

import sys
from os.path import dirname, join

SRC_PATH = join(dirname(dirname(dirname(__file__))), 'src')
sys.path.insert(0, SRC_PATH)


# -- Project information
project = 'robustnet'
copyright = '2026, Chao Wu'
author = 'Chao Wu'
version = '0.2.0'
release = version

pygments_style = 'sphinx'


# -- General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'nbsphinx',
    'autoapi.extension'
]

autoapi_type = 'python'
autoapi_dirs = [join(SRC_PATH, 'robustnet')]
autoapi_add_toctree_entry = True

source_suffix = '.rst'
master_doc = 'index'
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build']

html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    'show_toc_level': 2,
    'navigation_with_keys': True,
    'navbar_end': ['theme-switcher', 'navbar-icon-links'],
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/Chaowu88/robustnet',
            'icon': 'fa-brands fa-github',
        },
    ],
    'show_nav_level': 2
}

html_sidebars = {
    '**': ['sidebar-nav-bs']

pygments_light_style = 'tango'
pygments_dark_style = 'monokai'


# ------------------------ Options for HTML output ------------------------
mathjax_path = ('https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML')


# ------------------------ Options for LaTeX output ------------------------
# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, 
#  documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 
     project+'.tex', 
     project+' Documentation', 
     author, 
     'manual'
    )
]


# ------------------------ Options for manual page output ------------------------
# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 
     project, 
     project+' Documentation', 
     [author], 
     1
    )
]


# ------------------------ Options for Texinfo output ------------------------
# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc,
     project,
     project+' Documentation',
     author,
     'robustnet',
     'A python package for multi-omics-informed metabolic robustness',
     'Miscellaneous'
    )
]


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pymc': ('https://www.pymc.io/projects/docs/en/stable/learn.html', None)
}
