# Configuration file for the Sphinx documentation builder.

import sys
from os.path import dirname, join
from datetime import datetime
#from importlib.metadata import version as get_version


SRC_PATH = join(dirname(dirname(dirname(__file__))), 'src')
sys.path.insert(0, SRC_PATH)


# -- Project information
project = 'RobustNet'
copyright = f'{datetime.now():%Y}, Chao Wu'
author = 'Chao Wu'
version = '0.2.0'#get_version('robustnet')
release = version


# -- General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'nbsphinx',
    'autoapi.extension',
    'sphinx_togglebutton'
]

autoapi_type = 'python'
autoapi_dirs = [join(SRC_PATH, 'robustnet')]
autoapi_add_toctree_entry = True

#source_suffix = '.rst'
master_doc = 'index'
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build']

autosummary_generate = True

pygments_style = 'sphinx'

togglebutton_selector = '.toggle, .admonition.dropdown'
nbsphinx_allow_errors = True

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pymc': ('https://www.pymc.io/projects/docs/en/stable/learn.html', None)
}


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
