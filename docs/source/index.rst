RobustNet
=========

RobustNet is a Python package for ensemble-based simulation of metabolic responses and robustness analysis of enzyme-catalyzed metabolic systems. Leveraging a Bayesian approach, RobustNet integrates multi-omics data, including fluxomics, metabolomics, proteomics, and experimental enzyme kinetic measurements, to infer posterior model parameters and generate ensemble metabolic models. The framework provides controllable balancing between parameter consistency and sampling flexibility during parameter estimation. 

The typical RobustNet workflow consists of three steps:

- Estimate the reference-state flux distribution
- Generate parameter sets for ensemble models
- Simulate metabolic responses and evaluate system robustness

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Installation <installation>
   Building a model <tutorials/build_model>
   API reference <autoapi/robustness/index>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
