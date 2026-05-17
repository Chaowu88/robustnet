'''Define Model class.
'''


import re
from collections import namedtuple
from platform import system
import numpy as np
import pandas as pd
from sympy.parsing.sympy_parser import parse_expr
from .utils import (read_reaction_file, read_data_file, FluxFitResults, 
                    NonTCSampleResults, EnsembleResults)
from .kinetics import Fitter
from .sampling import Sampler
from .sensitivity import EnsembleSimulator
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


Reaction = namedtuple('Reaction', ['enzyme', 'substrates', 'products', 'ratelaw'])

Ratelaw = namedtuple('Ratelaw', ['enz', 'metabs', 'kparams', 'expr'])


class Model:

    def __init__(self, name=None):
        '''
        Parameters
        ----------
        name: str, optional
            Model name.
        '''

        self.name = name
        
        self.reaction_list = []
        self.enzyme_list = []
        self.metabolite_list = []
        self.reaction_info = {}
        
        self.reaction_list_temp = set()
        self.enzyme_list_temp = set()
        self.metabolite_list_temp = set()
        self.reaction_info_temp = {}   

    
    def _parse_reactants(self, expr):
        '''
        Parameters
        ----------
        expr : str
            Expression describing substrates or products. Spaces between 
            coefficients and metabolite names are allowed. Metabolite names
            cannot start with a number.

        Returns
        -------
        metab_coe : dict
            Dictionary mapping metabolite IDs to their stoichiometric coefficients.
        '''
        
        pattern_re = re.compile(r'^(\d*\.?\d*)\s*([\w_-]+)$')

        metab_coe = {}
        if expr is not np.nan:
            for coe_metab_str in expr.split('+'):
                coe, metab = pattern_re.match(coe_metab_str.strip()).groups()

                if re.match(r'\d', metab):
                    logging.warning(f"{metab}: metabolite name cannot start "
                                    "with a number.")

                if coe == '':
                    coe = 1.
                
                metab_coe[metab] = metab_coe.get(metab, 0.) + float(coe)   

        return metab_coe
    

    def _parse_expression(self, rxn, expr_str, enzymes, metabolites):
        '''
        Parse a reaction rate expression.

        Parameters
        ----------
        expr_str : str
            Mathematical expression defining the reaction rate.
        rxn : str
            Reaction ID.
        enzymes : list of str
            List of enzyme IDs.
        metabolites : list of str
            List of metabolite IDs.

        Returns
        -------
        ratelaw : namedtuple
            Parsed reaction rate information with the following fields:

            - ``enz`` (str or None): Enzyme variable.
            - ``metabs`` (list of str): Sorted metabolite variables.
            - ``kparams`` (list of str): Sorted kinetic parameter variables.
            - ``expr``: Symbolic expression of the reaction rate.
        '''
        
        expr_sym = parse_expr(expr_str)
        vars_sym = expr_sym.free_symbols
        vars_set = set(list(map(str, vars_sym)))

        enz_set = set(enzymes) & vars_set
        if len(enz_set) == 0:
            enz = None
        elif len(enz_set) == 1:
            enz = list(enz_set).pop()
        else:
            logging.warning(f'Rate expression for reaction {rxn} contains more than'
                            ' one enzyme variable, please check the rate law.')

        metabs_set = set(metabolites) & vars_set
        metabs = sorted(metabs_set)

        kparams_set = vars_set - enz_set - metabs_set
        kparams = sorted(kparams_set)

        ratelaw = Ratelaw(enz, metabs, kparams, expr_sym)

        return ratelaw


    def _update_model(self):
        '''
        Update reaction information in the model. Variables appearing in the
        rate expression that are not identified as enzymes, substrates, or
        products are treated as kinetic parameters.
        '''

        self.reaction_list = sorted(self.reaction_list_temp)
        self.enzyme_list = sorted(self.enzyme_list_temp)
        self.metabolite_list  = sorted(self.metabolite_list_temp)
        
        self.reaction_info = {}
        for rxn, items in self.reaction_info_temp.items():
            enz, sub_coe, pro_coe, rate_expr = items
            ratelaw = self._parse_expression(
                rxn, rate_expr, self.enzyme_list, self.metabolite_list
            )
            self.reaction_info[rxn] = Reaction(
                enz, sub_coe, pro_coe, ratelaw
            )
        

    def read_from_file(self, filename):
        '''
        Read reaction information from a file. Previously imported reactions with 
        the same reaction ID will be overwritten.

        Parameters
        ----------
        filename : str
            Path to the input file containing reaction information. Supported
            file formats are ``.xlsx``, ``.tsv``, and ``.csv``. 
            
            The input file must contain the following columns:

            - ``Reaction``: Reaction ID.
            - ``Enzyme``: Catalytic enzyme associated with the reaction.
              Leave empty if the reaction is non-enzymatic.
            - ``Substrates``: Reaction substrates separated by ``+``.
              Spaces between metabolite names are allowed. Metabolite names
              cannot start with a number, but may contain ``-`` and ``_``.
              Metabolite names must not overlap with enzyme names.
            - ``Products``: Reaction products separated by ``+``.
              Spaces between metabolite names are allowed. Metabolite names
              cannot start with a number, but may contain ``-`` and ``_``.
              Metabolite names must not overlap with enzyme names.
            - ``Rate expression``: Mathematical expression defining the
              reaction rate. Enzyme and metabolite names must be consistent
              with those provided in the ``Enzyme``, ``Substrates``, and
              ``Products`` columns. Any additional variables appearing in the
              expression are treated as kinetic parameters. Both reversible
              and irreversible reactions are supported. Spaces in the
              expression are allowed.
        '''

        data = read_reaction_file(filename)

        for _, row in data.iterrows():
            rxn, enz, sub_str, pro_str, rate_expr = row
            
            sub_coe = self._parse_reactants(sub_str)
            pro_coe = self._parse_reactants(pro_str)

            self.reaction_list_temp.add(rxn)
            if enz is not np.nan:
                self.enzyme_list_temp.add(enz)
            self.metabolite_list_temp.update(sub_coe.keys())
            self.metabolite_list_temp.update(pro_coe.keys())
            self.reaction_info_temp[rxn] = [enz, sub_coe, pro_coe, rate_expr]

        self._update_model()


    def add_reaction(
            self, 
            name=None, 
            enzyme=None, 
            substrates=None, 
            products=None, 
            rate_expression=None
        ):
        '''
        Add reaction information into the model. Previously imported reactions with 
        the same reaction ID will be overwritten.

        Parameters
        ----------
        name : str, optional
            reaction ID.
        enzyme : str or None, optional
            Catalytic enzyme associated with the reaction. Use ``None`` for
            non-enzymatic reactions.
        substrates : dict or None, optional
            Dictionary mapping substrate metabolite IDs to stoichiometric
            coefficients. Metabolite names cannot start with a number, but may
            contain ``-`` and ``_``. Metabolite names must not overlap with
            enzyme names. Use ``None`` if the reaction has no substrates.
        products : dict or None, optional
            Dictionary mapping product metabolite IDs to stoichiometric
            coefficients. Metabolite names cannot start with a number, but may
            contain ``-`` and ``_``. Metabolite names must not overlap with
            enzyme names. Use ``None`` if the reaction has no products.
        rate_expression : str or None, optional
            Mathematical expression defining the reaction rate. Enzyme and
            metabolite names must be consistent with those used in the
            ``enzyme``, ``substrates``, and ``products`` arguments. Any
            additional variables in the expression are treated as kinetic
            parameters. Spaces in the expression are allowed.
        '''

        self.reaction_list_temp.add(name)
        if enzyme is not None:
            self.enzyme_list_temp.add(enzyme)
        self.metabolite_list_temp.update(substrates.keys())
        self.metabolite_list_temp.update(products.keys())
        self.reaction_info_temp[name] = [
            enzyme, substrates, products, rate_expression
        ]

        self._update_model()


    def remove_reaction(self, reactions):
        '''
        Remove reaction(s) from the model.

        Parameters
        ----------
        reactions : str or list of str
            Reaction ID or list of reaction IDs to remove.
        '''

        if isinstance(reactions, str):
            reactions = [reactions]
        
        delete_rxns =[]
        metabs_in_delete_rxns = []
        for rxn in reactions:
            if rxn in self.reaction_info_temp:
                enz, sub_coe, pro_coe, _ = self.reaction_info_temp[rxn]

                metabs_in_delete_rxns.extend(
                    list(sub_coe.keys()) + list(pro_coe.keys())
                )

                self.reaction_info_temp.pop(rxn)
                self.reaction_list_temp.discard(rxn)
                if enz is not None and enz is not np.nan:
                    self.enzyme_list_temp.discard(enz)

                delete_rxns.append(rxn)
            
            else:
                logging.warning(f"Cannot remove reaction {rxn}, "
                                "which is not in the model.")
        
        stoy_mat = self._make_stoichiometric_matrix(
            self.reaction_info,
            tuple(self.metabolite_list),
            tuple(self.reaction_list)
        ).drop(delete_rxns, axis=1)
        metabs_rest = self._get_live_metabolites(stoy_mat)
        for metab in set(metabs_in_delete_rxns):
            if metab not in metabs_rest:
                self.metabolite_list_temp.discard(metab)
        
        self._update_model()
        

    def parsed_kinetic_parameters(self, reactions='all'):
        '''
        Show parsed kinetic parameters in reactions.

        Parameters
        ----------
        reactions : str or list of str, optional
            Reactions for which parsed kinetic parameters are returned.
            Use ``"all"`` to return kinetic parameters for all reactions
            in the model.
        '''

        if reactions == 'all':
            reactions = self.reaction_list
        elif isinstance(reactions, str):
            reactions = [reactions]

        kin_params = {}
        for rxn in reactions:
            if rxn in self.reaction_info:
                kin_params[rxn] = self.reaction_info[rxn].ratelaw.kparams

        return kin_params


    def __repr__(self):
        model_name = self.name or 'unknown'
        if len(self.reaction_list) != 0 and len(self.metabolite_list) != 0:
            return (
                f'Model {model_name} with '
                f'{len(self.reaction_list)} reactions and '
                f'{len(self.metabolite_list)} metabolites'
            )
        else:
            return f'Model {model_name} empty'
        

    @staticmethod
    def _make_stoichiometric_matrix(
        reaction_info, 
        metabolites, 
        reactions, 
        exclude=None,
    ):
        '''
        Build the stoichiometric matrix from reaction information.

        Parameters
        ----------
        reaction_info : dict
            Dictionary of reactions in the format
            ``{reaction_id: Reaction_namedtuple}``.
        metabolites : tuple or list
            Metabolite IDs defining the matrix rows.
        reactions : tuple or list
            Reaction IDs defining the matrix columns.
        exclude : tuple, list or None, optional
            Metabolites to exclude from the stoichiometric matrix.
        '''

        stoy_mat = pd.DataFrame(0., index=metabolites, columns=reactions)
        for rxn in reactions:
            for sub, coe in reaction_info[rxn].substrates.items():
                stoy_mat.loc[sub, rxn] = -coe
            for pro, coe in reaction_info[rxn].products.items():
                stoy_mat.loc[pro, rxn] = coe

        if exclude is not None:
            stoy_mat = stoy_mat.drop(exclude, errors='ignore')

        return stoy_mat
        
    
    def stoichiometric_matrix(self, exclude=None):
        '''
        Compute the stoichiometric matrix, where rows correspond to metabolites
        and columns correspond to reactions.

        Parameters
        ----------
        exclude : list, optional
            Metabolites to exclude from the stoichiometric matrix.
        '''

        if len(self.reaction_list) == 0 or len(self.metabolite_list) == 0:
            logging.error("Cannot compute stoichiometric matrix: "
                          "no metabolites or reactions were found.")
        
        stoy_mat = self._make_stoichiometric_matrix(
            self.reaction_info,
            tuple(self.metabolite_list),
            tuple(self.reaction_list),
            exclude=exclude
        )

        return stoy_mat
    

    @staticmethod
    def _get_live_metabolites(stoy_mat):
        return stoy_mat[(stoy_mat != 0).any(axis=1)].index.to_list()
    

    def initial_substrates(self, exclude=None):
        '''
        Get the initial substrates of the model. Initial substrates are
        metabolites that only participate in only one consuming reaction 
        and are not produced by any reaction.

        Parameters
        ----------
        exclude : list, optional
            Metabolites to exclude from the stoichiometric matrix and,
            consequently, from the returned initial substrate list.
        '''

        stoy_mat = self.stoichiometric_matrix(exclude=exclude)
        sel = (stoy_mat <= 0).all(axis=1) & (stoy_mat < 0).sum(axis=1) == 1
        init_metabs = stoy_mat.index[sel].tolist()

        return init_metabs
        

    def final_products(self, exclude=None):
        '''
        Get the final products of the model. Final products are metabolites
        that only participate in only one producing reaction and are not consumed
        by any reaction.

        Parameters
        ----------
        exclude : list, optional
            Metabolites to exclude from the stoichiometric matrix and,
            consequently, from the returned final product list.
        '''
        
        stoy_mat = self.stoichiometric_matrix(exclude=exclude)
        sel = (stoy_mat >= 0).all(axis=1) & (stoy_mat > 0).sum(axis=1) == 1
        fin_metabs = stoy_mat.index[sel].tolist()
        
        return fin_metabs        


    def _check_data_consistency(self, data, data_type, kind, std):
        std_label = 'standard deviation of ' if std else ''
        if not isinstance(data, data_type):
            raise TypeError(f'Expecting {std_label}{kind} data.')
        else:
            return data


    def load_priors(self, kind, data, std=None):
        '''
        Metabolomics and proteomics data are expected in units of mM, while
        fluxomics data are expected in units of mmol/L/s. For kinetic parameters, 
        catalytic constants have units of 1/s, Michaelis, activation,
        and inhibition constants have units of mM, and equilibrium
        constants are dimensionless. All concentration and flux units are
        assumed to be cell-volume based. Missing values are allowed and may be 
        indicated by ``"nan"``, ``"na"``, or blank entries in the input file.
        
        Parameters
        ----------
        kind : {"metabolomics", "proteomics", "fluxomics", "reference_fluxes", 
                "kparameters"}
            Type of prior measurement data.
        data : file or pandas.Series
            Prior measurements. File names must end with ``.xlsx``,
            ``.tsv``, or ``.csv``. The input file with headers must
            contain species name and its measurement in each row.
            Alternatively, a ``pandas.Series`` can be provided.
        std : file or pandas.Series, optional
            Standard deviations of prior measurements. File names must end
            with ``.xlsx``, ``.tsv``, or ``.csv``. The input
            file with headers must contain species name and its measurement 
            in each row. Alternatively, a ``pandas.Series`` can be provided. 
            The shape of ``std`` must match that of ``data``. Use ``None`` if 
            uncertainty information is unavailable.
        '''

        if isinstance(data, str):
            prior_data = read_data_file(data, n_dims=1)
        elif isinstance(data, pd.Series):
            prior_data = data
        else:
            raise ValueError('Invalid format for prior data.')

        if isinstance(std, str):
            prior_std = read_data_file(std, n_dims=1)
        elif isinstance(data, (type(None), pd.Series)):
            prior_std = std
        else:
            raise ValueError('Invalid format for prior uncertainty data.')

        kind = kind.lower()
        
        if kind == 'metabolomics':
            self.metabolomics = self._check_data_consistency(
                prior_data, pd.Series, kind, False
            )
            if prior_std is not None:
                self.metabolomics_std = self._check_data_consistency(
                    prior_std, pd.Series, kind, True
                )
        elif kind == 'proteomics':
            self.proteomics = self._check_data_consistency(
                prior_data, pd.Series, kind, False
            )
            if prior_std is not None:
                self.proteomics_std = self._check_data_consistency(
                    prior_std, pd.Series, kind, True
                )
        elif kind == 'fluxomics':
            self.fluxomics = self._check_data_consistency(
                prior_data, pd.Series, kind, False
            )
            if prior_std is not None:
                self.fluxomics_std = self._check_data_consistency(
                    prior_std, pd.Series, kind, True
                )
        elif kind == 'reference_fluxes':
            self.ref_fluxes = self._check_data_consistency(
                prior_data, pd.Series, kind, False
            )
            if prior_std is not None:
                self.ref_fluxes_std = self._check_data_consistency(
                    prior_std, pd.Series, kind, True
                )
        elif kind == 'kparameters':
            self.kparameters = self._check_data_consistency(
                prior_data, pd.Series, kind, False
            )
            if prior_std is not None:
                self.kparameters_std = self._check_data_consistency(
                    prior_std, pd.Series, kind, True
                )
        else:
            raise ValueError('kind must be one of {"metabolomics", '
                             '"proteomics", "fluxomics", "reference_fluxes", '
                             '"kparameters"}.')


    def estimate_reference_fluxes(
            self,
            bounds,
            exclude_metabolites=None,
            optimizer='scipy',
            method='COBYQA',
            tol=1e-4,
        ):
        '''
        Estimate the reference flux distribution by fitting measured fluxes
        (e.g., from 13C fluxomics) using least-squares regression.

        Fluxes are expected in units of mmol/L/s with cell-based volumes.

        Note that it is recommended to run the optimization multiple times to 
        improve the chance of obtaining a solution with minimal SSR.

        Parameters
        ----------
        bounds : 2-tuple or dict of 2-tuples
            Lower and upper bounds used during fitting. If a tuple is
            provided, the same bounds are applied to all fluxes. If a dict
            is provided, bounds can be specified for individual fluxes,
            which is useful for defining flux reversibility. For example,
            Setting ``lower_bound >= 0`` or ``upper_bound <= 0`` enforces
            irreversibility in the forward or reverse direction,
            respectively.

            Fluxes without explicit bounds use the minimum lower bound and 
            maximum upper bound from the provided bounds.
        exclude_metabolites : list of str or None, optional
            Metabolites excluded from steady-state mass balance constraints,
            such as initial substrates and final products. If ``None``,
            these metabolites are inferred automatically from the
            stoichiometric matrix.

            Because inferred initial substrates and final products depend on
            reaction direction definitions, explicitly specifying this
            argument is recommended.
        optimizer : {"scipy", "nlopt"}, optional
            Optimizer used to solve the fitting problem. The NLopt package
            must be installed if ``optimizer="nlopt"`` is selected.
        method : {"COBYQA", "SLSQP"}, optional
            optimization method. ``COBYQA`` is gradient-free and generally
            more robust for non-smooth problems, but may be slower.
            ``SLSQP`` is gradient-based and typically faster, but may
            struggle with highly nonlinear problems.

            Currently, ``COBYQA`` is only available when ``optimizer="scipy"``.
        tol : float, optional
            Tolerance criterion for optimization convergence.

        Returns
        -------
        res: FluxFitResults
            Object containing the estimated reference flux distribution. 
            
            Important attributes and methods include:
            
            - ``estimated_fluxes``: Estimated reference flux distribution.
            - ``estimated_flux_errors``: Estimated errors in the reference flux 
              distribution.
            - ``plot_simulated_vs_measured_fluxes``: Visualize simulated versus 
              measured fluxes.
        '''
        
        if not hasattr(self, 'fluxomics'):
            raise ValueError(
                'No fluxomics data found. Call load_priors first.'
            )
        else:
            exclude_metabolites = exclude_metabolites or []

            fit_res = Fitter(
                self,
                exclude_metabolites
            ).fit_reference_fluxes(bounds, optimizer, method, tol)

            return FluxFitResults(*fit_res)


    def generate_parameter_sets(
            self,
            ref_flux_mu=None,
            kparam_prior_mu=None,
            mconc_prior_mu=None,
            econc_prior_mu=None,
            ref_flux_sigma=None,
            kparam_prior_sigma=None,
            mconc_prior_sigma=None,
            econc_prior_sigma=None,
            ref_flux_initvalues=None,
            kparam_initvalues=None,
            mconc_initvalues=None,
            econc_initvalues=None,
            exclude_metabolites=None,
            alpha=None,   #1e6,
            n_tunes=5000,
            n_samples=2000,
            n_chains=10,
            n_jobs=1,
        ):
        '''
        Sample model parameters based on prior knowledge of reference fluxes, 
        metabolite concentrations, enzyme concentrations and enzyme kinetic 
        parameters, which can be collected from omics data and enzyme database. 
        Missing values are allowed.

        Prior measurements can be provided directly through the corresponding
        arguments or loaded beforehand using ``load_priors``. Directly supplied
        arguments take precedence over loaded priors.

        Parameters
        ----------
        ref_flux_mu : dict, pandas.Series or None
            Reference-state flux distribution in units of mmol/L/s
            (cell-based). Typically obtained from ``estimate_reference_fluxes``.
            If ``None``, reference fluxes loaded by ``load_priors``
            are used.
        kparam_prior_mu : dict, pandas.Series or None
            Mean values of the prior distributions for kinetic parameters.
            Catalytic constants have units of 1/s, Michaelis, activation,
            and inhibition constants have units of mM, and equilibrium
            constants are dimensionless. Missing kinetic parameters are
            allowed. If ``None``, kinetic parameters loaded by ``load_priors``
            are used.
        mconc_prior_mu : dict, pandas.Series, or None
            Mean values of prior metabolite concentrations in mM
            (cell-based). Missing metabolites are allowed. If ``None``,
            metabolomics data loaded by ``load_priors`` are used.
        econc_prior_mu : dict, pandas.Series, or None
            Mean values of prior enzyme concentrations in mM
            (cell-based). Missing enzymes are allowed. If ``None``,
            proteomics data loaded by ``load_priors`` are used.
        ref_flux_sigma : scalar, dict, pandas.Series, or None, optional
            Standard deviations of reference-state fluxes. If a scalar is
            provided, the same value is used for all fluxes. Missing flux
            values are allowed when using a dict or ``pandas.Series``.
            Typically obtained from ``estimate_reference_fluxes``. If ``None``, 
            standard deviations from reference fluxes loaded by ``load_priors`` 
            are used. Defaults to ``0.01``.
        kparam_prior_sigma : scalar, dict, pandas.Series, or None, optional
            Standard deviations of kinetic parameters. If a scalar is
            provided, the same value is used for all parameters. Missing
            parameter values are allowed. If ``None``, standard deviations
            from kinetic parameters loaded by ``load_priors``
            are used. Default to ``0.1``.
        mconc_prior_sigma : scalar, dict, pandas.Series, or None, optional
            Standard deviations of metabolite concentrations. If a scalar is
            provided, the same value is used for all metabolites. Missing
            metabolite values are allowed. If ``None``, standard deviations
            from metabolomics data loaded by ``load_priors`` are used. Default 
            to ``0.1``.
        econc_prior_sigma : scalar, dict, pandas.Series, or None, optional
            Standard deviations of enzyme concentrations. If a scalar is
            provided, the same value is used for all enzymes. Missing enzyme
            values are allowed. If ``None``, standard deviations from
            proteomics data loaded by ``load_priors`` are used. Default to 
            ``0.001``.
        ref_flux_initvalues : dict, pandas.Series, or None, optional
            Initial values for reference flux sampling. Missing fluxes are
            allowed. If ``None``, ``ref_flux_mu`` is used.
        kparam_initvalues : dict, pandas.Series, or None, optional
            Initial values for kinetic parameter sampling. Missing parameter
            values are allowed. If ``None``, ``kparam_prior_mu`` is used.
        mconc_initvalues : dict, pandas.Series, or None, optional
            Initial values for metabolite concentration sampling. Missing
            metabolite values are allowed. If ``None``,
            ``mconc_prior_mu`` is used.
        econc_initvalues : dict, pandas.Series, or None, optional
            Initial values for enzyme concentration sampling. Missing enzyme
            values are allowed. If ``None``, ``econc_prior_mu`` is used.
        exclude_metabolites : list of str or None, optional
            Reserved argument. Currently unused.
        alpha : float or None, optional
            Gaussian penalty strength used in parameter balancing. Larger
            values impose stronger penalties in log-posterior space.

            A reasonable choice is often on the same order of magnitude as
            ``1 / ref_flux_sigma**2``. If ``None``,
            ``geomean(1 / ref_flux_sigma**2)`` is used.

            Excessively large values of ``alpha`` may produce sparse or
            discontinuous samples and significantly reduce the number of
            effective models in robustness analysis.
        n_tunes : int, optional
            Number of tuning iterations performed before sampling in each
            chain.
        n_samples : int, optional
            Number of samples drawn in each chain.
        n_chains : int, optional
            Number of sampling chains.
        n_jobs : int, optional
            Number of parallel jobs to run in parallel. 
            For Windows platforms, ``n_jobs`` is forced to ``1`` for
            compatibility reasons.

        Returns
        -------
        res: NonTCSampleResults
            Object containing the sampled parameter sets. 
            
            Important attributes and methods include:

            - ``sampled_reference_fluxes``: Sampled reference flux distributions.
            - ``sampled_kinetic_parameters``: Sampled kinetic parameters.
            - ``sampled_metabolite_concentrations``: Sampled metabolite  
              concentrations.
            - ``sampled_enzyme_concentrations``: Sampled enzyme concentrations.
            - ``trace``: ``arviz.InferenceData`` object containing the sampling 
              trace.
            - ``plot_sampled_vs_prior_fluxes``: Visualize sampled versus prior 
              reference fluxes.
            - ``plot_sampled_vs_prior_kinetic_parameters``: Visualize sampled
               versus prior kinetic parameters.
            - ``plot_sampled_vs_prior_metabolites``: Visualize sampled versus prior 
              metabolite concentrations.
            - ``plot_sampled_vs_prior_enzymes``: Visualize sampled versus prior 
              enzyme concentrations.
        '''
        
        if ref_flux_mu is None and not hasattr(self, 'ref_fluxes'):
            raise ValueError('No reference flux data provided through '
                             'ref_flux_mu or loaded via load_priors.')
        
        if mconc_prior_mu is None and not hasattr(self, 'metabolomics'):
            raise ValueError('No metabolomics data provided through '
                             'mconc_prior_mu or loaded via load_priors.')
        
        if econc_prior_mu is None and not hasattr(self, 'proteomics'):
            raise ValueError('No proteomics data is provided through '
                             'econc_prior_mu or loaded via load_priors.')
        
        if kparam_prior_mu is None and not hasattr(self, 'kparameters'):
            raise ValueError('No kinetic parameter data is provided through '
                             'kparam_prior_mu or load via load_priors.')

        if ref_flux_sigma is None and not hasattr(self, 'ref_fluxes_std'):
            ref_flux_sigma = 0.01
        
        if mconc_prior_sigma is None and not hasattr(self, 'metabolomics_std'):
            mconc_prior_sigma = 0.1
        
        if econc_prior_sigma is None and not hasattr(self, 'proteomics_std'):
            econc_prior_sigma = 0.001
        
        if kparam_prior_sigma is None and not hasattr(self, 'kparameters_std'):
            kparam_prior_sigma = 0.1
        
        exclude_metabolites = []

        if system() == 'Windows':
            n_jobs = 1

        samp_res = Sampler(
            self,
            exclude_metabolites
        ).sample_with_omics(
            ref_flux_mu,
            kparam_prior_mu,
            mconc_prior_mu,
            econc_prior_mu,
            ref_flux_sigma,
            kparam_prior_sigma,
            mconc_prior_sigma,
            econc_prior_sigma,
            ref_flux_initvalues,
            kparam_initvalues,
            mconc_initvalues,
            econc_initvalues,
            alpha,
            n_tunes,
            n_samples,
            n_chains,
            n_jobs
        )

        return NonTCSampleResults(*samp_res)


    def load_parameter_sets(
            self,
            mconc_set, 
            econc_set,
            kparam_set,
        ):
        '''
        Load sampled parameter sets for robustness analysis.

        Sampled parameter sets must be mutually compatible by index. For
        example, the ``i``-th sampled metabolite concentration set must
        correspond to the ``i``-th sampled kinetic parameter set and enzyme
        concentration set.

        Parameters
        ----------
        mconc_set : file or pandas.DataFrame
            Sampled metabolite concentration sets. File names must end with
            ``.xlsx``, ``.tsv``, or ``.csv``. Columns correspond to
            metabolites. Alternatively, a ``pandas.DataFrame`` can be
            provided.
        econc_set : file or pandas.DataFrame
            Sampled enzyme concentration sets. File names must end with
            ``.xlsx``, ``.tsv``, or ``.csv``. Columns correspond to enzymes.
            Alternatively, a ``pandas.DataFrame`` can be provided.
        kparam_set : file or pandas.DataFrame
            Sampled kinetic parameter sets. File names must end with
            ``.xlsx``, ``.tsv``, or ``.csv``. Columns correspond to kinetic
            parameters. Alternatively, a ``pandas.DataFrame`` can be
            provided.

            Note that some metabolites may be treated as kinetic parameters if they
            only appear in reaction rate expressions. These metabolites must
            therefore also be included in the kinetic parameter sets.
        '''
        
        if isinstance(mconc_set, str):
            self.init_metab_sets = read_data_file(mconc_set)
        elif isinstance(mconc_set, pd.DataFrame):
            self.init_metab_sets = mconc_set
        else:
            raise ValueError('Invalid format of metabolite '
                             'concentration sets.')
        
        if isinstance(econc_set, str):
            self.enz_conc_sets = read_data_file(econc_set)
        elif isinstance(econc_set, pd.DataFrame):
            self.enz_conc_sets = econc_set
        else:
            raise ValueError('Invalid format of enzyme concentration sets.')

        if isinstance(kparam_set, str):
            self.kin_param_sets = read_data_file(kparam_set)
        elif isinstance(kparam_set, pd.DataFrame):
            self.kin_param_sets = kparam_set
        else:
            raise ValueError('Invalid format of kinetic parameter sets.')
        
        self.influx_sets = pd.DataFrame(index=self.init_metab_sets.index)
        
        self.efflux_sets = pd.DataFrame(index=self.init_metab_sets.index)

        if not (self.init_metab_sets.shape[0] 
                == self.influx_sets.shape[0]
                == self.efflux_sets.shape[0] 
                == self.kin_param_sets.shape[0]
                == self.enz_conc_sets.shape[0]):
            raise ValueError('Inconsistent sizes detected among parameter sets.')
        

    def evaluate_robustness(
            self,
            perturb_enzymes='all',
            fold_change=(0.2, 5),
            exclude_metabolites=None,
            n_steps=200,
            log_spacing=False,
            n_models=5000,
            n_jobs=1,
            flux_sensitivity=True,
            check_jacobian=True,
            check_metabolite=True
        ):
        '''
        Simulate perturbation responses and evaluate the robustness of the 
        metabolic system with respect to enzyme expression perturbations.

        Parameters
        ----------
        perturb_enzymes : list of str or "all", optional
            Enzymes to perturb simultaneously over their specified expression
            ranges. If ``"all"``, all enzymes in the model are perturbed.
        fold_change : 2-tuple or dict, optional
            Relative perturbation bounds for enzyme concentrations, defined
            as fold changes with respect to the reference state.

            The fold-change interval must always include ``1`` (the
            reference state). Perturbations are simulated from the reference
            state toward both left and right bounds. For example,
            ``(0.1, 10)`` and ``(10, 0.1)`` both indicate simulations from
            ``1 → 0.1`` and ``1 → 10``.

            If one bound equals ``1``, perturbation is only performed in
            one direction. For example, ``(1, 50)`` and ``(50, 1)`` both
            indicate perturbation from ``1 → 50``.

            If a dict is provided, fold-change ranges can be assigned
            individually to enzymes. This allows coordinated or opposed
            regulation patterns. For example, 
            ``{"A": (0.1, 10), "B": (50, 0.5)}`` indicates that in the left 
            direction enzyme A is perturbed from ``1 -> 0.1`` while enzyme B 
            is perturbed simultaneously from ``1 -> 50``, whereas in the right 
            direction enzyme A is perturbed from ``1 -> 10`` while enzyme B is 
            perturbed simultaneously from ``1 -> 0.5``.

            Enzymes without explicitly assigned ranges use the default
            fold-change range ``(0.2, 5)``.

            Enzymes specified in ``fold_change`` but not explicitly listed
            in ``perturb_enzymes`` are still perturbed if their names are
            valid.

            This argument can also be used when
            ``perturb_enzymes="all"``. In this case, specified fold-change
            ranges override the default values.
        exclude_metabolites : list of str or None, optional
            Metabolites excluded from the analysis. Excluded metabolites are
            not included in the ODE system.
        n_steps : int, optional
            Number of perturbation steps in each direction. If perturbation
            is performed in both directions, ``2 * n_steps`` total steps are
            applied.

            Large fold changes may cause model failure at early perturbation
            steps, resulting in no effective models. Increasing ``n_steps``
            may improve stability in such cases.
        log_spacing : bool, optional
            If ``True``, perturbation points are logarithmically spaced.
            Otherwise, they are evenly spaced.
        n_models : int, optional
            Number of sampled models used in the simulation. Increasing this
            value may improve consensus robustness estimates.

            The maximum number of models is limited by the number of loaded
            parameter sets.
        n_jobs : int, optional
            Number of parallel jobs. 
        flux_sensitivity : bool, optional
             If ``True``, derivatives of steady-state fluxes with respect to
             enzyme concentrations are also computed.
        check_jacobian : bool, optional
            If ``True``, verify that all eigenvalues of the Jacobian matrix
            have negative real parts.
        check_metabolite : bool, optional
            If ``True``, enforce positive metabolite concentrations during
            analysis.

        Returns
        -------
        res: EnsembleResults
            Object containing robustness analysis results. 
            
            Important attributes and methods include:

            - ``robust_index``: Robustness index under the specified
              perturbation condition.
            - ``robust_model_probability``: Plot the probability of models 
              remaining robust across perturbations.
            - ``metabolite_sensitivity``: Plot metabolite concentration
              responses across perturbations.
            - ``bifurcation_diagram``: Alias for ``metabolite_sensitivity``.
            - ``metabolite_distribution``: Plot distributions of
              metabolite concentrations during perturbations.
            - ``flux_sensitivity``: Plot steady-state flux responses across
              perturbations. Available only when ``flux_sensitivity=True``.
            - ``flux_distribution``: Plot distributions of steady-state
              fluxes during perturbations. Available only when 
              ``flux_sensitivity=True``.
            - ``eigreal_sensitivity``: Plot responses of the maximum real
              part of Jacobian eigenvalues across perturbations.
            - ``eigreal_distribution``: Plot distributions of the maximum
              real part of Jacobian eigenvalues during perturbations.
        '''

        if not hasattr(self, 'init_metab_sets'):
            raise ValueError('No metabolite concentration sets found. '
                             'Call load_parameter_sets first.')
        elif not hasattr(self, 'influx_sets'):
            raise ValueError('No influx sets found. '
                             'Call load_parameter_sets first.')
        elif not hasattr(self, 'efflux_sets'):
            raise ValueError('No efflux sets found. '
                             'Call load_parameter_sets first.')
        elif not hasattr(self, 'kin_param_sets'):
            raise ValueError('No kinetic parameter sets found. '
                             'Call load_parameter_sets first.')
        elif not hasattr(self, 'enz_conc_sets'):
            raise ValueError('No enzyme concentration sets found. '
                             'Call load_parameter_sets first.')
        else:
            exclude_metabolites = exclude_metabolites or []

            if not check_jacobian:
                logging.warning('Jacobian matrix will not be checked.')

            if not check_metabolite:
                logging.warning('Metabolite concentration will not be checked.')

            sim_res = EnsembleSimulator(
                self, 
                exclude_metabolites
            ).simulate(
                perturb_enzymes, 
                fold_change, 
                n_steps,
                log_spacing,
                n_models,
                n_jobs,
                flux_sensitivity,
                check_jacobian,
                check_metabolite
            )

            return EnsembleResults(
                *sim_res, 
                self.stoichiometric_matrix(exclude_metabolites),
                n_steps
            )
    