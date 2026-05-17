Documentation for RobustNet
===========================

RobustNet is a Python package for ensemble-based simulation of metabolic responses and robustness analysis of enzyme-catalyzed metabolic systems. Leveraing a Bayesian approach, RobustNet integrates multi-omics data, including fluxomics, metabolomics, proteomics, and experimental enzyme kinetic measurements, to infer posterior model paramters and generate ensemble metabolic models. The framework provides controllable balancing between parameter consistency and sampling flexibility during parameter estimation. 

The typical RobustNet workflow consists of three steps:

- 1. Estimate the reference-state flux distibution
- 2. Generate parameter sets for ensemble models
- 3. Simulate metabolic responses and evaluate system robustness

.. toctree::
   :numbered:
   :maxdepth: 2
   :caption: Contents:

   API </autoapi/robustness/index.rst>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
