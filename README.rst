=========
RobustNet
=========

RobustNet is a Python package for simulating metabolic responses and evaluating the robustness of metabolic systems to perturbations in enzyme expression levels. The framework integrates multi-omics data, including fluxomics, metabolomics, proteomics, and enzyme kinetic databases, to construct ensembles of data-constrained kinetic models for systems-level analysis. 

RobustNet is designed to characterize how intracellular metabolite concentrations and metabolic fluxes respond to metabolic engineering interventions, particularly enzyme upregulation and downregulation. By combining Bayesian parameter inference with ensemble-based perturbation simulations, the package enables quantitative assessment of metabolic stability, system viability, and sensitivity under perturbations, providing mechanistic insights to guide biosystem design and metabolic engineering strategies.

For additional details, please refer to the `documentation <https://robustnet.readthedocs.io/en/latest/index.html>`__. We also provide demonstrative `scripts <https://github.com/Chaowu88/robustnet/tree/main/scripts>`__ illustrating the complete workflow for analyzing metabolic robustness in representative models of `E.coli <https://github.com/Chaowu88/robustnet/tree/main/models/e_coli>`__ and `Synechocystis <https://github.com/Chaowu88/robustnet/tree/main/models/synechocystis>`__.

Installation
============

The package has been tested with Python 3.10-3.13 and can be installed from PyPI using *pip*.

.. code-block:: python

  python -m pip install --upgrade pip
  pip install robustnet

Alternatively, the package can be installed directly from source (assuming `git <https://git-scm.com/>`__ is installed):

.. code-block:: python

  git clone https://github.com/Chaowu88/robustnet.git /path/to/robustnet
  pip install /path/to/robustnet

It is recommended to install the package within a `virtual environment <https://docs.python.org/3.13/tutorial/venv.html>`__ or `conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__.

For large-scale simulations, deployment on a high-performance computing system with parallel execution enabled is recommended to improve computational efficiency.

Example Usage
=============

The RobustNet workflow consists of three major steps. First, let's initialize and load the metablic model:

.. code-block:: python

  from robustnet import Model
  
  model = Model('model_name')
  model.read_from_file('path/to/model_file')

1. Estimate the reference-state fluxes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first step is to estimate a reference steady-state flux distribution consistent with the defined metabolic network using available fluxomics data (typically from :sup:`13`\ C metabolic flux analysis).

.. code-block:: python

  model.load_priors(
    'fluxomics',
    data=flux_measurements,
    std=flux_measurement_uncertainties
  )
  
  fit_res = model.estimate_reference_fluxes(
    bounds=flux_bounds,
    exclude_metabolites=excluded_from_mass_balance
  )

2. Generate ensemble model sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, posterior parameter distributions are inferred using multi-omics measurements and enzyme kinetic priors. The resulting sampled parameter sets are used to construct ensembles of kinetic models for downstream simulations.

.. code-block:: python

  model.load_priors(
    'reference_fluxes',
    data=ref_fluxes,
    std=ref_flux_uncertainties
  )
  model.load_priors(
    'metabolomics',
    data=metab_measurements,
    std=metab_measurement_uncertainties
  )
  model.load_priors(
    'proteomics',
    data=enz_measurements,
    std=enz_measurement_uncertainties
  )
  model.load_priors(
    'kparameters',
    data=kparam_measurements,
    std=kparam_measurement_uncertainties
  )
  
  samp_res = model.generate_parameter_sets(
    alpha=pentality_strength,
    n_samples=n_samples_to_draw,
    n_jobs=n_parallel_jobs
  )

The sampling procedure integrates information from experimental measurements and enzyme databases while maintaining consistency across heterogeneous data sources and mechanistic constraints.

3. Simulate metabolic responses to perturbations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using the parameterized ensemble models, metabolic responses and robustness can be evaluated under specified enzyme perturbations.

.. code-block:: python

  model.load_parameter_sets(
    mconc_set=sampled_metab_concentrations,
    econc_set=sampled_enz_concentrations,
    kparam_set=sampled_kparams
  )
  
  rob_res = model.evaluate_robustness(
    perturb_enzymes=enz_list,
    fold_change=perturbation_amplitude,
    n_models=n_models_to_generate,
    n_jobs=n_parallel_jobs
  )

This workflow enables multi-omics-informed analysis of metabolic robustness and system-level responses to enzyme perturbations. The package also provides helper utilities and visualization functions for inspecting model parameterization, perturbation trajectories, robustness metrics, and sensitivity analyses. 

For more detailed tutorials and API references, please refer to the `documentation <https://robustnet.readthedocs.io/en/latest/index.html>`__.
