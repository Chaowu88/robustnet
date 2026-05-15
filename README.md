# RobustNet

RobustNet is a Python package for simulating metabolic responses and evaluating metabolic robustness of a biosystem to catalytic enzyme level changes informed by multi-omics data of fluxomics, metabolomics, proteomics and enyzme databases. Our goal is to elucidate how intracellular metabolite levels and metabolic fluxes will adjust accordingly with respect to metabolic engineering inverventions i.e., enzyme expression level interventions and assess the system viability and fitness after the perturbations, and provide insightful guidance for biosystem designs and metabolic engineering efforts.

For further details, please refer to the `documentation <https://robustnet.readthedocs.io/en/latest/index.html>`__. We also provide demenstrative `scripts <https://github.com/Chaowu88/robustnet/tree/main/scripts>`__ of performing our workflow to analyze the metabolic robustness of two representive `E.coli <https://github.com/Chaowu88/robustnet/tree/main/models/e_coli>`__ and `Synechocystis <https://github.com/Chaowu88/robustnet/tree/main/models/synechocystis>`__.
metabolic engineering inverventions i.e., enzyme expression level interventions informed by multi-omics data of fluxomics, metabolomics, proteomics and enyzme databases.

## Installation

The package has been tested in Python 3.10 through 3.13. It can be installed using *pip* from PyPI.

.. code-block:: python

  python -m pip install --upgrade pip
  pip install robustnet

Or can be installed from the source (assuming `git <https://git-scm.com/>`__ is installed):

.. code-block:: python

  git clone https://github.com/Chaowu88/robustnet.git /path/to/
  pip install /path/to/robustnet

Note. It is recommended to install the package within a `virtual <https://docs.python.org/3.13/tutorial/venv.html>`__ or `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__ environment.

For better perfomance and running efficiency, it is also recommended to deploy in a high-performance computing cluster with parallel jobs enabled in massive amout.

Example Usage
=============

Our workflow consists of three steps with corresponding functions provided.

### 1. Estimate the reference-state flux distribution

