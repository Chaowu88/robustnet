Installation
============

Using PIP
---------

RobustNet is compatible with Python versions 3.10 through 3.13 and can be directly installed from PyPI using *pip*. It is recommended to first upgrade ``pip`` to the latest version:

.. code-block:: python

  python -m pip install --upgrade pip

Then install RobusNet with:

.. code-block:: python

  pip install robustnet

Alternatively, RobustNet can be installed from source by cloning the GitHub repository using the following commands (assuming `git <https://git-scm.com/>`__ is installed):

.. code-block:: python

  git clone https://github.com/Chaowu88/robustnet.git /path/to/robustnet
  pip install /path/to/robustnet

.. Note::
  It's recommended to install RobustNet within a virtual environment. Please refer to the Python `venv documentation <https://docs.python.org/3.13/tutorial/venv.html>`__ or the `Conda environment guide <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__ for environment setup.

  For large-scale simulations and improved computational efficiency, deploying RobustNet on a high-performance computing system with parallel exccution enabled is strongly recommended.

Optional Solver installation
----------------------------

RobustNet uses ``SciPy`` `optimizers <https://docs.scipy.org/doc/scipy/tutorial/optimize.html>`__`` by default for estimating reference fluxes. The ``NLopt`` optimization library is also supported but must be installed seperately. Please refer to their `guide <https://nlopt.readthedocs.io/en/latest/NLopt_Installation/>`__ for instructions.