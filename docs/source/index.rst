RobustNet
=========

RobustNet is a Python package for ensemble-based simulation of metabolic responses and robustness analysis of enzyme-catalyzed metabolic systems. Leveraing a Bayesian approach, RobustNet integrates multi-omics data, including fluxomics, metabolomics, proteomics, and experimental enzyme kinetic measurements, to infer posterior model paramters and generate ensemble metabolic models. The framework provides controllable balancing between parameter consistency and sampling flexibility during parameter estimation. 

The typical RobustNet workflow consists of three steps:

- Estimate the reference-state flux distibution
- Generate parameter sets for ensemble models
- Simulate metabolic responses and evaluate system robustness

.. toctree::
   :numbered:
   :maxdepth: 2
   :caption: Contents:

   installation
   build_model
   API </autoapi/robustness/index.rst>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
