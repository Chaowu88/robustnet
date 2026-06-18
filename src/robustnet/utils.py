'''Define utility and helper functions.
'''


import os
from functools import lru_cache
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import norm, truncnorm
from tqdm import tqdm
from threading import Thread
from contextlib import contextmanager
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import logging
from time import sleep


METAB_SEN_PLOT_YLIM = (1e-2, 1e2)
FLUX_SEN_PLOT_YLIM = (0.5, 1.5)
SEN_PLOT_N_BINS = 49   # use odd number


def read_reaction_file(filename):
    '''
    Read reaction information form file.

    Parameters
    ----------
    filename : str
        File containing reaction information. Supported file formats are
        ``.xlsx``, ``.tsv``, and ``.csv``.

        Required columns are:

        - ``Reaction``: Reaction identifier.
        - ``Enzyme``: Catalytic enzyme.
        - ``Substrates``: Reaction substrates.
        - ``Products``: Reaction products.
        - ``Rate expression``: Reaction rate expression.
    '''

    fields = {
        0: 'reaction', 
        1: 'enzyme', 
        2: 'substrates', 
        3: 'products', 
        4: 'expression'
    }

    ext = Path(filename).suffix
    if ext == '.xlsx':
        data = pd.read_excel(
            filename, 
            header=0, 
            index_col=None, 
            usecols=list(fields.keys()), 
            names=list(fields.values())
        )
    elif ext == '.csv':
        data = pd.read_csv(
            filename, 
            header=0, 
            index_col=None, 
            usecols=list(fields.keys()), 
            names=list(fields.values())
        )
    elif ext == '.tsv':
        data = pd.read_csv(
            filename, 
            header=0, 
            index_col=None, 
            usecols=list(fields.keys()), 
            names=list(fields.values()),
            sep='\t'
        )
    
    return data


def read_data_file(filename, n_dims=2):
    '''
    Read omics data or sampled parameter sets from file.

    Parameters
    ----------
    filename : str
        File containing omics data, or sampled parameter sets. Supported file 
        formats are ``.xlsx``, ``.tsv``, and ``.csv``. Headers are required.

        For sampled parameter sets, required columns include
        ``set_id`` and metabolites/influxes/effluxes/kinetic 
        parameters/enzymes. A ``pandas.DataFrame`` is
        returned. 
    
        For steady-state omics data, a ``pandas.Series`` is returned.
    '''

    ext = Path(filename).suffix
    if n_dims == 2:
        if ext == '.xlsx':
            data = pd.read_excel(
                filename, header=0, index_col=0
            ).dropna(axis=1, how='all')
        elif ext == '.csv':
            data = pd.read_csv(
                filename, header=0, index_col=0
            ).dropna(axis=1, how='all')
        elif ext == '.tsv':
            data = pd.read_csv(
                filename, header=0, index_col=0, sep='\t'
            ).dropna(axis=1, how='all')
    else:
        if ext == '.xlsx':
            data = pd.read_excel(
                filename, header=None, index_col=0
            ).squeeze(axis=1).dropna()
        elif ext == '.csv':
            data = pd.read_csv(
                filename, header=None, index_col=0
            ).squeeze(axis=1).dropna()
        elif ext == '.tsv':
            data = pd.read_csv(
                filename, header=None, index_col=0, sep='\t'
            ).squeeze(axs=1).dropna()
    
    return data


def generate_distribution(mean, std, n_samples, varnames, nonneg=True):
    '''
    Generate random values subject to specified normal or truncated normal 
    distribution.

    Parameters
    ----------
    mean : numpy.ndarray
        Mean values with shape ``(n_vars,)``.
    std : numpy.ndarray
        Standard deviations with shape ``(n_vars,)``.
    n_samples : int
        Number of samples to generate.
    varnames : list
        Variable names corresponding to the sampled variables.
    nonneg : bool
        If ``True``, constrain sampled values to be nonnegative, i.e., using 
        truncated normal distribution.
    '''

    if nonneg:
        dist = truncnorm.rvs(
            a=-mean/std, b=np.inf, 
            loc=mean, scale=std, 
            size=(n_samples, mean.size)
        )
    else:
        dist = norm.rvs(
            loc=mean, scale=std, 
            size=(n_samples, mean.size)
        )

    dist = pd.DataFrame(dist, columns=varnames)

    return dist


def plot_sampled_vs_prior_distribution(
        out_dir, 
        kind, 
        data, 
        xlabel, 
        show_fig, 
        output_data
    ):
    '''
    Plot distribution comparison of sampled versus prior data.

    Parameters
    ----------
    out_dir : str or None
        Output directory.
    kind : {'flux', 'metab', 'enz', 'kparam'}
        Type of variable to plot.
    data : pandas.DataFrame
        Data used for plotting with columns ``['Prior', 'Sampled']``.
    xlabel : str
        Label for the x-axis.
    show_fig : bool
        If ``True``, display the figure.
    output_data : bool
        If ``True``, export the data used for plotting. Requires ``out_dir`` 
        to be specified.
    '''

    fig, ax = plt.subplots()
    sns.histplot(
        data, bins=100, stat='probability', kde=True, 
        kde_kws={'bw_adjust': 5}, line_kws={'linewidth': 4}, 
        element='step',
    )

    if kind == 'flux':
        unit = '(mmol L$^{-1}$ s$^{-1}$)'
    elif kind in ['metab', 'enz']:
        unit = '(mM)'
    else:
        unit = ''
    ax.set_xlabel(f'{xlabel} {unit}', fontsize=25)
    ax.yaxis.label.set_size(25)
    ax.tick_params(labelsize=20)

    sns.move_legend(obj=ax, loc='best', frameon=False, fontsize=15)

    if show_fig:
        plt.show()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

        fig.savefig(
            f'{out_dir}/{xlabel}.jpg', dpi=300, bbox_inches='tight'
        )    

        if output_data:
            data.to_csv(
                f'{out_dir}/{xlabel}.tsv', header=True, index=False, sep='\t'
            )
    
    plt.close()


class FluxFitResults:

    def __init__(self, vs, vs_std, opt_obj, dof, v_exp):
        '''
        Parameters
        ----------
        vs : pandas.Series
            Estimated reference flux distribution.
        vs_std : pandas.Series or None
            Estimated standard deviations of the fitted fluxes.
        opt_obj : float
            Optimal objective value of the least-squares regression.
        dof : int
            Degree of freedom in the fitting problem.
        v_exp : pandas.Series
            Experimental fluxomics data actually used in the fitting.
        '''

        self.vs = vs
        self.vs_std = vs_std
        self.opt_obj = opt_obj
        self.dof = dof
        self.v_exp = v_exp

    
    @property
    def estimated_fluxes(self):
        '''
        Returns pandas.Series.
        '''

        return self.vs
    

    @property
    def estimated_flux_errors(self):
        '''
        Returns pandas.Series or None.
        '''

        if self.dof <= 0:
            logging.warning('Uncertainty estimates for fitted fluxes are '
                            'unavailable because the degree of freedom (DOF) is '
                            'non-positive.')
        return self.vs_std
    

    def plot_simulated_vs_measured_fluxes(
            self, 
            out_dir=None, 
            reactions='all', 
            show_fig=False
        ):
        '''
        Plot simulated versus measured fluxes.

        Parameters
        ----------
        out_dir : str or None, optional
            Output directory.
        reactions : list of str, optional
            Reactions to plot. If ``all``, all fluxes are plotted.
        show_fig : bool, optional
            If ``True``, display the figure.
        '''

        if reactions == 'all':
            reactions = self.v_exp.index
        else:
            reactions = sorted(set(reactions) & set(self.v_exp.index))

        for rxn in reactions:
            plt.bar(x=-0.15, height=self.v_exp[rxn], width=0.3, label='Measured')
            plt.bar(x=0.15, height=self.vs[rxn], width=0.3, label='Fitted')
            plt.xlim((-1, 1))
            plt.xlabel(rxn, fontsize=25)
            plt.ylabel('Flux (mmol L$^{-1}$ s$^{-1}$)', fontsize=25)
            plt.tick_params(labelsize=20, bottom=False, labelbottom=False)
            plt.legend(frameon=False, fontsize=12)
            
            if show_fig:
                plt.show()
            
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

                plt.savefig(
                    f'{out_dir}/{rxn}.jpg', dpi=300, bbox_inches='tight'
                )
            
            plt.close()
    

    def __repr__(self):
        msg = ('<FluxFitResults> Use estimated_fluxes to access estimated'
               ' reference fluxes.')
        
        return msg


class NonTCSampleResults:

    def __init__(
            self, idata, 
            v_exp_mu, v_exp_sigma,
            k_exp_mu, k_exp_sigma, 
            x_exp_mu, x_exp_sigma, 
            e_exp_mu, e_exp_sigma
        ):
        '''
        Parameters
        ----------
        idata : arviz.InferenceData
            Data structure containing sampled parameter traces.
        v_exp_mu : pandas.Series
            Mean values of the prior distributions for reference fluxes.
            Missing values are imputed using the geometric mean of the
            available values.
        v_exp_sigma : pandas.Series
            Standard deviations of the prior distributions for reference
            fluxes. Missing values are imputed using the geometric mean of
            the available values.
        k_exp_mu : pandas.Series
            Mean values of the prior distributions for kinetic parameters.
            Missing values are imputed using the geometric mean of the
            available values.
        k_exp_sigma : pandas.Series
            Standard deviations of the prior distributions for kinetic
            parameters. Missing values are imputed using the geometric mean
            of the available values.
        x_exp_mu : pandas.Series
            Mean values of the prior distributions for metabolite
            concentrations. Missing values are imputed using the geometric
            mean of the available values.
        x_exp_sigma : pandas.Series
            Standard deviations of the prior distributions for metabolite
            concentrations. Missing values are imputed using the geometric
            mean of the available values.
        e_exp_mu : pandas.Series
            Mean values of the prior distributions for enzyme
            concentrations. Missing values are imputed using the geometric
            mean of the available values.
        e_exp_sigma : pandas.Series
            Standard deviations of the prior distributions for enzyme
            concentrations. Missing values are imputed using the geometric
            mean of the available values.
        '''

        self.idata = idata
        self.posterior = az.extract(idata)
        self.v_exp_mu = v_exp_mu
        self.v_exp_sigma = v_exp_sigma
        self.k_exp_mu = k_exp_mu
        self.k_exp_sigma = k_exp_sigma
        self.x_exp_mu = x_exp_mu
        self.x_exp_sigma = x_exp_sigma
        self.e_exp_mu = e_exp_mu
        self.e_exp_sigma = e_exp_sigma
        
        self.rxns = self.posterior.coords['vs'].values.tolist()
        self.kparams = self.posterior.coords['kparams'].values.tolist()
        self.metabs = self.posterior.coords['ini_xs'].values.tolist()
        self.enzs = self.posterior.coords['es'].values.tolist()
        
        self.sampled_vs = pd.DataFrame(
            self.posterior.data_vars['v'].values.T, columns=self.rxns
        )
        self.sampled_kparams = pd.DataFrame(
            self.posterior.data_vars['kparam'].values.T, columns=self.kparams
        )
        self.sampled_xs = pd.DataFrame(
            self.posterior.data_vars['x'].values.T, columns=self.metabs
        )
        self.sampled_es = pd.DataFrame(
            self.posterior.data_vars['e'].values.T, columns=self.enzs
        )
        

    @property
    def sampled_reference_fluxes(self):
        '''
        Return
        ------
        pandas.DataFrame
            Sampled reference fluxe distribution with samples as rows and fluxes 
            as columns.
        '''

        return self.sampled_vs
    

    @property
    def sampled_kinetic_parameters(self):
        '''
        Returns
        -------
        pandas.DataFrame
            Sampled kinetic parameters with samples as rows and kinetic
            parameters as columns.
        '''

        return self.sampled_kparams
    

    @property
    def sampled_metabolite_concentrations(self):
        '''
        Returns
        -------
        pandas.DataFrame
            Sampled metabolite concentrations with samples as rows and
            metabolites as columns.
        '''

        return self.sampled_xs
    

    @property
    def sampled_enzyme_concentrations(self):
        '''
        Returns
        -------
        pandas.DataFrame
            Sampled enzyme concentrations with samples as rows and enzymes
            as columns.
        '''

        return self.sampled_es


    @property
    def trace(self):
        '''
        Returns
        -------
        arviz.InferenceData
            Inference data object containing posterior samples and sampling
            statistics.
        '''

        return self.idata
    

    def plot_sampled_vs_prior_fluxes(
            self, 
            out_dir=None, 
            fluxes='all',
            show_fig=False,
            output_data=False
        ):
        '''
        Plot sampled versus prior reference flux distribution.

        Parameters
        ----------
        out_dir : str or None, optional
            Output directory.
        fluxes : list of str, optional
            Fluxes to plot. If ``all``, all fluxes are plotted.
        show_fig : bool, optional
            If ``True``, display the figure.
        output_data : bool, optional
            If ``True``, export the data used for plotting. Requires ``out_dir`` 
            to be specified.
        '''

        if fluxes == 'all':
            fluxes = self.rxns
        else:
            fluxes = sorted(set(fluxes) & set(self.rxns))

        fluxes_prior = generate_distribution(
            self.v_exp_mu,
            self.v_exp_sigma,
            self.sampled_reference_fluxes.shape[0],
            self.rxns,
            nonneg=False
        )

        for flux in fluxes:
            data = pd.concat(
                (fluxes_prior[flux], self.sampled_reference_fluxes[flux]),
                axis=1
            )
            data.columns = ['Prior', 'Sampled']

            plot_sampled_vs_prior_distribution(
                out_dir, 'flux', data, flux, show_fig, output_data
            )


    def plot_sampled_vs_prior_kinetic_parameters(
            self, 
            out_dir=None, 
            parameters='all',
            show_fig=False,
            output_data=False
        ):
        '''
        Plot sampled versus prior kinetic parameters.

        Parameters
        ----------
        out_dir : str or None, optional
            Output directory.
        parameters : list of str, optional
            Kinetic parameters to plot. If ``all``, all parameters are
            plotted.
        show_fig : bool, optional
            If ``True``, display the figure.
        output_data : bool, optional
            If ``True``, export the data used for plotting. Requires ``out_dir`` 
            to be specified.
        '''

        if parameters == 'all':
            parameters = self.kparams
        else:
            parameters = sorted(set(parameters) & set(self.kparams))

        kparams_prior = generate_distribution(
            self.k_exp_mu, 
            self.k_exp_sigma, 
            self.sampled_kparams.shape[0], 
            self.kparams
        )
        
        for param in parameters:
            data = pd.concat(
                (kparams_prior[param], self.sampled_kparams[param]), 
                axis=1
            )
            data.columns = ['Prior', 'Sampled']

            plot_sampled_vs_prior_distribution(
                out_dir, 'kparam', data, param, show_fig, output_data
            )


    def plot_sampled_vs_prior_metabolites(
            self, 
            out_dir=None, 
            metabolites='all',
            show_fig=False,
            output_data=False
        ):
        '''
        Plot sampled versus prior metabolite concentrations.

        Parameters
        ----------
        out_dir : str or None, optional
            Output directory.
        metabolites : list of str, optional
            Metabolites to plot. If ``all``, all metabolites are plotted.
        show_fig : bool, optional
            If ``True``, display the figure.
        output_data : bool, optional
            If ``True``, export the data used for plotting. Requires ``out_dir`` 
            to be specified.
        '''

        if metabolites == 'all':
            metabolites = self.metabs
        else:
            metabolites = sorted(set(metabolites) & set(self.metabs))
        
        mconcs_prior = generate_distribution(
            self.x_exp_mu,
            self.x_exp_sigma,
            self.sampled_xs.shape[0],
            self.metabs
        )

        for metab in metabolites:
            data = pd.concat(
                (mconcs_prior[metab], self.sampled_xs[metab]),
                axis=1
            )
            data.columns = ['Prior', 'Sampled']

            plot_sampled_vs_prior_distribution(
                out_dir, 'metab', data, metab, show_fig, output_data
            )


    def plot_sampled_vs_prior_enzymes(
            self, 
            out_dir=None, 
            enzymes='all',
            show_fig=False,
            output_data=False
        ):
        '''
        Plot sampled versus prior enzyme concentrations.
        
        Parameters
        ----------
        out_dir : str or None, optional
            Output directory.
        enzymes : list of str, optional
            Enzymes to plot. If ``all``, all enzymes are plotted.
        show_fig : bool, optional
            If ``True``, display the figure.
        output_data : bool, optional
            If ``True``, export the data used for plotting. Requires ``out_dir`` 
            to be specified.
        '''

        if enzymes == 'all':
            enzymes = self.enzs
        else:
            enzymes = sorted(set(enzymes) & set(self.enzs))

        econcs_prior = generate_distribution(
            self.e_exp_mu,
            self.e_exp_sigma,
            self.sampled_es.shape[0],
            self.enzs
        )

        for enz in enzymes:
            data = pd.concat(
                (econcs_prior[enz], self.sampled_es[enz]),
                axis=1
            )
            data.columns = ['Prior', 'Sampled']

            plot_sampled_vs_prior_distribution(
                out_dir, 'enz', data, enz, show_fig, output_data
            )
        

class EnsembleResults:

    def __init__(self, pert_enzs, sim_res, v_sen, stoy_mat, n_steps):
        '''
        Parameters
        ----------
        pert_enzs : dict of tuple
            Perturbed enzymes and their relative perturbation bounds in the
            form ``enzyme -> (lb, rb)``. 
        sim_res : list of dict
            Simulation results for all models. Each dictionary contains the
            simulation outputs and model information for one model.

            Keys include:

            - ``model_idx``: model index
            - ``x_r``: metabolite concentrations for perturbation toward the
              right bound
            - ``v_r``: fluxes for perturbation toward the right bound
            - ``steps_r``: perturbation steps toward the right bound
            - ``x_l``: metabolite concentrations for perturbation toward the
              left bound
            - ``v_l``: fluxes for perturbation toward the left bound
            - ``steps_l``: perturbation steps toward the left bound
            - ``maxreals_r``: maximum Jacobian eigenvalue real parts for
              perturbation toward the right bound
            - ``maxreals_l``: maximum Jacobian eigenvalue real parts for
              perturbation toward the left bound
            - ``x_ini``: initial metabolite concentrations
            - ``e_ini``: initial enzyme concentrations
            - ``kparams``: kinetic parameters
            - ``vins``: influxes
            - ``vouts``: effluxes
            - ``rel_lbs``: relative left perturbation bounds
            - ``rel_rbs``: relative right perturbation bounds
        v_sen : bool
            If ``True``, derivatives of steady-state fluxes with respect to
            enzyme concentrations are also computed.
        stoy_mat : pandas.DataFrame
            Stoichiometric matrix.
        n_steps: int
            Number of perturbation integration steps.
        '''

        self.pert_enzs = pert_enzs
        self.sim_res = sim_res
        self.v_sen = v_sen
        self.n_eff_models = len(sim_res)
        self.metabs = stoy_mat.index.to_list()
        self.rxns = stoy_mat.columns.to_list()
        self.n_steps = n_steps


    @property
    @lru_cache
    def effective_models(self):
        '''
        Return indices of models that remain valid before perturbation and at
        the first perturbation step.

        These indices also correspond to the loaded parameter sets, including
        metabolite concentrations, enzyme concentrations, kinetic parameters,
        and influxes or effluxes if available.

        A small number of valid models may lead to biased simulation results.
        In such cases, consider increasing ``n_models`` or ``n_steps`` in
        ``evaluate_robustness``, or generating additional parameter sets.
        '''
        
        eff_models = sorted([d['model_idx'] for d in self.sim_res])

        return eff_models
    

    def _check_simulation_results(self):
        '''
        Check whether simulation results are empty.
        '''

        if len(self.sim_res) == 0:
            raise ValueError('No valid simulation results are available. Try '
                             'increasing n_models or n_steps in '
                             'evaluate_robustness or generating '
                             'additional parameter sets.')

    
    def _plot_helper_pert_direction(self):
        l_only = False
        r_only = False
        if len(self.pert_enzs) > 1:
            xlabel = 'Multiple perturbation'
            suffix = 'mul-enz'
            if all([bnd[1] == 1 for bnd in self.pert_enzs.values()]):
                xticks = [0, self.n_steps]
                xticklabels = ['LB', 1]
                l_only = True
            elif all([bnd[0] == 1 for bnd in self.pert_enzs.values()]):
                xticks = [0, self.n_steps]
                xticklabels = [1, 'RB']
                r_only = True
            else:
                xticks = [0, self.n_steps, 2*self.n_steps]
                xticklabels = ['LB', 1, 'RB']
        else:
            enz = list(self.pert_enzs.keys())[0]
            xlabel = f'{enz} perturbation'
            suffix = enz
            if self.pert_enzs[enz][1] == 1:
                xticks = [0, self.n_steps]
                xticklabels = [self.pert_enzs[enz][0], 1]
                l_only = True
            elif self.pert_enzs[enz][0] == 1:
                xticks = [0, self.n_steps]
                xticklabels = [1, self.pert_enzs[enz][1]]
                r_only = True
            else:
                xticks = [0, self.n_steps, 2*self.n_steps]
                xticklabels = [self.pert_enzs[enz][0], 1, self.pert_enzs[enz][1]]

        return l_only, r_only, xticks, xticklabels, xlabel, suffix
    

    def _get_robust_model_probability(self, l_only, r_only):
        for i, data in enumerate(self.sim_res):
            if not r_only:
                l_count = np.zeros(self.n_steps)
                if data['x_l'].shape[0] > 0:
                    l_count[-data['x_l'].shape[0]:] = 1
            
            if not l_only:
                r_count = np.zeros(self.n_steps)
                if data['x_r'].shape[0] > 0:
                    r_count[:data['x_r'].shape[0]] = 1
            
            if i == 0:
                if l_only or r_only:
                    count_base = np.zeros(self.n_steps+1)
                else:
                    count_base = np.zeros(2*self.n_steps+1)
            else:
                count_base = count

            if l_only:
                count = count_base + np.concatenate((l_count, [1]))
            elif r_only:
                count = count_base + np.concatenate(([1], r_count))
            else:
                count = count_base + np.concatenate((l_count, [1], r_count))
        
        prob = count/self.n_eff_models

        return prob
        

    def _plot_helper_sensitivity_stats_plot(
            self,
            out_dir,
            kind,
            item,
            l_only, 
            r_only, 
            xticks, 
            xticklabels, 
            xlabel, 
            suffix,
            indices,
            ylim,
            n_bins,
            show_fig,
            output_data,
        ):
        '''
        Helper function to plot sensitivity based on statistics of all results.

        Parameters
        ----------
        kind: {'metab', 'flux'}
        '''

        if platform.system() == 'Linux':
            os.sched_setaffinity(os.getpid(), range(os.cpu_count()))
        
        if kind == 'metab':
            key_l = 'x_l'
            key_r = 'x_r'
            key_ini = 'x_ini'
            cmap = sns.cubehelix_palette(
                start=3, rot=0.5, dark=0.2, light=1, gamma=1.5, 
                as_cmap=True, reverse=False
            )
            ylabel = f'Rel. conc. of {item}\nat steady state'
        elif kind == 'flux':
            key_l = 'v_l'
            key_r = 'v_r'
            key_ini = 'v_ini'
            cmap = sns.cubehelix_palette(
                start=3, rot=-0.5, dark=0.2, light=1, gamma=1.5, 
                as_cmap=True, reverse=False
            )
            ylabel = f'Rel. flux of {item}\nat steady state'
        
        fig, ax = plt.subplots()
        rel_data_all = []
        for i in indices:
            data = self.sim_res[i]
            if not r_only:
                rel_data_l = np.concatenate((
                    np.full(self.n_steps-data[key_l][item].size, np.nan),
                    data[key_l][item][::-1], [data[key_ini][item]],
                ))/data[key_ini][item]   # left padding

            if not l_only:
                rel_data_r = np.concatenate((
                    [data[key_ini][item]], data[key_r][item],
                    np.full(self.n_steps-data[key_r][item].size, np.nan)
                ))/data[key_ini][item]   # right paddding

            if l_only:
                rel_data_all.append(rel_data_l)
            elif r_only:
                rel_data_all.append(rel_data_r)
            else:
                rel_data_all.append(
                    np.concatenate((rel_data_l, rel_data_r[1:]))
                )
        rel_data_all = np.abs(np.vstack(rel_data_all))
        if kind == 'metab':
            bins = np.logspace(*np.log10(ylim), n_bins+1)
        elif kind == 'flux':
            bins = np.linspace(*ylim, n_bins+1)
        
        rel_data_all_count = np.apply_along_axis(
            lambda col: np.histogram(col, bins=bins)[0][::-1],
            axis=0,
            arr=rel_data_all
        )
        
        sns.heatmap(rel_data_all_count/self.n_eff_models, cmap=cmap)
        
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=0)
        ax.set_yticks([0+0.5, n_bins//2+0.5, n_bins-0.5])
        ax.set_yticklabels([ylim[1], 1, ylim[0]])
        ax.set_xlabel(xlabel, fontsize=25)
        ax.set_ylabel(ylabel, fontsize=25)
        ax.tick_params(labelsize=20)

        cbar = ax.collections[0].colorbar
        cbar.locator = mtick.MaxNLocator(nbins=2)
        cbar.set_label('Probability', fontsize=25)
        cbar.ax.tick_params(labelsize=20)

        if show_fig:
            plt.show()

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

            fig.savefig(
                f'{out_dir}/{kind}_sen_stats_{item}_by_{suffix}.jpg', 
                dpi=300, bbox_inches='tight'
            )

            if output_data:
                rel_data_all = pd.DataFrame(rel_data_all, index=indices)
                rel_data_all.index.name = 'Model'
                rel_data_all.to_csv(
                    f'{out_dir}/{kind}_sen_stats_{item}_by_{suffix}_fc.tsv',
                    header=True, index=True, sep='\t'
                )

                rel_data_all_count = pd.DataFrame(rel_data_all_count)
                rel_data_all_count.index.name = 'Bin'
                rel_data_all_count.to_csv(
                    f'{out_dir}/{kind}_sen_stats_{item}_by_{suffix}.tsv',
                    header=True, index=True, sep='\t'
                )

        plt.close()


    def _plot_helper_sensitivity_sample_plot(
            self, 
            out_dir,
            kind,
            item, 
            l_only, 
            r_only, 
            xticks, 
            xticklabels, 
            xlabel, 
            suffix,
            indices,
            ylim,
            n_bins,
            show_fig,
            output_data
        ):
        '''
        Help function to plot sensitivity using sampled results.

        Parameters
        ----------
        kind: {'metab', 'flux'}
        '''
        
        if platform.system() == 'Linux':
            os.sched_setaffinity(os.getpid(), range(os.cpu_count()))
            
        if kind == 'metab':
            key_l = 'x_l'
            key_r = 'x_r'
            key_ini = 'x_ini'
            ylabel = f'Rel. conc. of {item}\nat steady state'
        elif kind == 'flux':
            key_l = 'v_l'
            key_r = 'v_r'
            key_ini = 'v_ini'
            ylabel = f'Rel. flux of {item}\nat steady state'

        fig, ax = plt.subplots()
        abs_data_all = []
        for i in indices:
            data = self.sim_res[i]
            if not r_only:
                rel_data_l = np.concatenate((
                    data[key_l][item][::-1], [data[key_ini][item]], 
                ))/data[key_ini][item]
                abs_data_l = np.concatenate((
                    np.full(self.n_steps-data[key_l][item].size, np.nan),
                    data[key_l][item][::-1], [data[key_ini][item]],
                ))   # left padding

            if not l_only:
                rel_data_r = np.concatenate(( 
                    [data[key_ini][item]], data[key_r][item]
                ))/data[key_ini][item]
                abs_data_r = np.concatenate((
                    [data[key_ini][item]], data[key_r][item],
                    np.full(self.n_steps-data[key_r][item].size, np.nan)
                ))   # right paddding

            if l_only:
                xpos = np.arange(
                    self.n_steps-data[key_l][item].size, self.n_steps+1
                )
                rel_mconc = rel_data_l
                abs_data_all.append(abs_data_l)
            elif r_only:
                xpos = np.arange(0, data[key_r][item].size+1)
                rel_mconc = rel_data_r
                abs_data_all.append(abs_data_r)
            else:
                xpos = np.arange(
                    self.n_steps-data[key_l][item].size, 
                    self.n_steps+data[key_r][item].size+1
                )
                rel_mconc = np.concatenate((rel_data_l, rel_data_r[1:]))
                abs_data_all.append(
                    np.concatenate((abs_data_l, abs_data_r[1:]))
                )
                
            sns.lineplot(x=xpos, y=np.abs(rel_mconc), linewidth=4)
        
        ax.set_ylim(top=ylim[1], bottom=ylim[0])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        if kind == 'metab':
            ax.set_yscale('log', base=10)
        ax.set_xlabel(xlabel, fontsize=25)
        ax.set_ylabel(ylabel, fontsize=25)
        ax.tick_params(labelsize=20)
        
        if show_fig:
            plt.show()

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

            fig.savefig(
                f'{out_dir}/{kind}_sen_samp_{item}_by_{suffix}.jpg', 
                dpi=300, bbox_inches='tight'
            )

            if output_data:
                abs_data_all = pd.DataFrame(abs_data_all, index=indices).abs()
                abs_data_all.index.name = 'Model'
                abs_data_all.to_csv(
                    f'{out_dir}/{kind}_sen_samp_{item}_by_{suffix}.tsv', 
                    header=True, index=True, sep='\t'
                )

        plt.close()


    def _plot_helper_distribution_plot(
            self, 
            out_dir, 
            kind, 
            item, 
            r_only, 
            l_only,
            show_fig,
            output_data
        ):
        '''
        Help function to plot distribution.

        Parameters
        ----------
        kind: {'metab', 'flux', 'maxeigreal'}
        '''

        if kind == 'metab':
            key_l = 'x_l'
            key_r = 'x_r'
            color_idx = 1
            xlabel = f'SS conc. of {item} (mM)'
            basename = f'{out_dir}/{kind}_dist_{item}'
            columns = ['Conc.']
        elif kind == 'flux':
            key_l = 'v_l'
            key_r = 'v_r'
            color_idx = 2
            xlabel = f'SS flux of {item}\n'+'(mmol L$^{-1}$ s$^{-1}$)'
            basename = f'{out_dir}/{kind}_dist_{item}'
            columns = ['Flux']
        elif kind == 'maxeigreal':
            key_l = 'maxreals_l'
            key_r = 'maxreals_r'
            color_idx = 3
            xlabel = r'Jacobian $\lambda_{Re}^{max}$'
            basename = f'{out_dir}/{kind}_dist'
            columns = ['Max(Real(eig))']
        
        fig, ax = plt.subplots()
        dist_data = []
        for data in self.sim_res:
            if kind in ['metab', 'flux']:
                if not r_only:
                    dist_data.append(data[key_l][item].values)
                if not l_only:
                    dist_data.append(data[key_r][item].values)
            else:
                if not r_only:
                    dist_data.append(data[key_l])
                if not l_only:
                    dist_data.append(data[key_r])
        dist_data = np.concatenate(dist_data)
        
        if kind in ['metab', 'flux']:   # Note value could be negative
            if (dist_data.size > 0 
                and dist_data.min() > 0
                and dist_data.max()/dist_data.min() > 100):
                log_scale = True
            else:
                log_scale = False
        else:
            log_scale = False
        sns.histplot(
            dist_data, bins=100, log_scale=log_scale, stat='probability',
            kde=True, kde_kws={'bw_adjust': 5}, line_kws={'linewidth': 4},
            element='step', color=sns.color_palette()[color_idx]
        ) 
        
        ax.set_xlabel(xlabel, fontsize=25)
        if kind == 'maxeigreal':
            ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
        ax.yaxis.label.set_size(25)
        ax.tick_params(labelsize=20)

        if show_fig:
            plt.show()
        
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

            fig.savefig(basename+'.jpg', dpi=300, bbox_inches='tight')
            
            if output_data:
                pd.DataFrame(dist_data, columns=columns).to_csv(
                    basename+'.tsv', header=True, index=False, sep='\t'
                )

        plt.close()
        

    @property
    def robust_index(self):
        self._check_simulation_results()
        
        l_only, r_only, xticks, *_ = self._plot_helper_pert_direction()
        prob = self._get_robust_model_probability(l_only, r_only)
        auc_norm = trapezoid(prob)/(xticks[-1]+1)

        return auc_norm


    def robust_model_probability(
            self, 
            out_dir=None, 
            show_fig=False, 
            output_data=False
        ):
        '''
        Plot the probability of robust models as a function of enzyme
        perturbation.

        The normalized area under the curve (norm AUC), ranging from 0 to 1,
        is used as a robustness metric. Higher norm AUC values indicate
        greater robustness against the specified perturbation strategy.

        Parameters
        ----------
        out_dir : str or None, optional
            Output directory.
        show_fig : bool, optional
            If ``True``, display the figure.
        output_data : bool, optional
            If ``True``, export the data used for plotting. Requires ``out_dir`` 
            to be specified.
        '''

        self._check_simulation_results()

        (l_only, 
         r_only, 
         xticks, 
         xticklabels, 
         xlabel, 
         suffix) = self._plot_helper_pert_direction()

        prob = self._get_robust_model_probability(l_only, r_only)
        auc_norm = trapezoid(prob)/(xticks[-1]+1)

        fig, ax = plt.subplots()
        
        sns.lineplot(prob, linewidth=5)
        ax.plot([], label=f'norm AUC: {auc_norm:.2f}')
        ax.legend(handlelength=0, fontsize=15, frameon=False)
        
        ax.set_ylim((-0.05, 1.05))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel(xlabel, fontsize=25)
        ax.set_ylabel('Probability of\nrobust model', fontsize=25)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.tick_params(labelsize=20)
        
        if show_fig:
            plt.show()

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

            fig.savefig(
                f'{out_dir}/robust_prob_by_{suffix}.jpg', 
                dpi=300, bbox_inches='tight'
            )
            
            
            if output_data:
                pd.DataFrame(prob, columns=['Prob.']).to_csv(
                    f'{out_dir}/robust_prob_by_{suffix}.tsv', header=True, sep='\t'
                )

        plt.close()


    def metabolite_sensitivity(
            self, 
            out_dir=None, 
            kind='stats',
            metabolites='all', 
            ylim=None,
            n_sets=100,
            rng_seed=None,
            n_bins=49,
            show_fig=False,
            output_data=False
        ):
        '''
        Plot responses of steady-state metabolite concentrations
        (relative values, i.e., ``|x| / |x0|``) to enzyme expression
        perturbations.

        Equivalent to ``bifurcation_diagram``.

        Parameters
        ----------
        out_dir : str or None, optional
            Output directory.
        kind: {"stats", "sample"}, optional
            Plot type.
            - ``"stats"``: plot probability distributions of relative
            concentrations across perturbation levels.
            - ``"sample"``: plot trajectories from sampled models.
        metabolites : list of str, optional
            Metabolites to plot. If ``all``, all metabolites are plotted.
        ylim : 2-tuple or None, optional
           Lower and upper y-axis limits. If ``None``, the default range is
            ``(1e-2, 1e2)`` for relative concentrations.
        n_sets : int, optional
            Number of sampled model sets used for plotting.

            If ``n_sets`` does not exceed the total number of sets, models are
            randomly selected. Otherwise, all sets are used.

            Valid only when ``kind="sample"``.
        rng_seed : int or None, optional
            Random seed used for model selection.

            Valid only when ``kind="sample"``.
        n_bins : int, optional
            Number of bins used in statistical summaries of relative
            concentrations or fluxes.

            Must be an odd number.

            Valid only when ``kind="stats"``.
        show_fig : bool, optional
            If ``True``, display the figure.
        output_data : bool, optional
            If ``True``, export the data used for plotting. Requires ``out_dir`` 
            to be specified.
        '''

        self._check_simulation_results()

        if metabolites == 'all':
            metabolites = self.metabs
        else:
            metabolites = sorted(set(metabolites) & set(self.metabs))

        plot_infos = self._plot_helper_pert_direction()

        ylim = ylim or (1e-2, 1e2)

        if kind == 'stats':
            indices = range(len(self.sim_res))
            worker = self._plot_helper_sensitivity_stats_plot
        elif kind == 'sample':
            if n_sets <= len(self.sim_res):
                indices = np.random.default_rng(rng_seed).choice(
                    range(len(self.sim_res)), n_sets, replace=False
                )
            else:
                indices = range(len(self.sim_res))
            worker = self._plot_helper_sensitivity_sample_plot
        else:
            raise ValueError('kind must be either "stats" or "sample".')

        # with Pool(n_jobs) as pool:
        #     results = []
        #     for metab in metabolites:
        #         res = pool.apply_async(
        #             func=worker,
        #             args=(
        #                 out_dir, 
        #                 'metab',
        #                 metab, 
        #                 *plot_infos,
        #                 indices,
        #                 ylim,
        #                 n_bins,
        #                 output_data
        #             )
        #         )
        #         results.append(res)

        #     for res in results:
        #         res.get()


        for metab in metabolites:
            worker(out_dir, 'metab', metab, *plot_infos, indices, ylim, 
                   n_bins, show_fig, output_data)


    def bifurcation_diagram(
            self, 
            out_dir=None, 
            kind='stats', 
            metabolites='all', 
            ylim=None,
            n_sets=100,
            rng_seed=None,
            n_bins=49,
            show_fig=False,
            output_data=False
        ):
        '''
       Plot responses of steady-state metabolite concentrations
        (relative values, i.e., ``|x| / |x0|``) to enzyme expression
        perturbations.

        Equivalent to ``metabolite_sensitivity``.

        Parameters
        ----------
        out_dir : str or None, optional
            Output directory.
        kind: {"stats", "sample"}, optional
            Plot type.
            - ``"stats"``: plot probability distributions of relative
            concentrations across perturbation levels.
            - ``"sample"``: plot trajectories from sampled models.
        metabolites : list of str, optional
            Metabolites to plot. If ``all``, all metabolites are plotted.
        ylim : 2-tuple or None, optional
           Lower and upper y-axis limits. If ``None``, the default range is
            ``(1e-2, 1e2)`` for relative concentrations.
        n_sets : int, optional
            Number of sampled model sets used for plotting.

            If ``n_sets`` does not exceed the total number of sets, models are
            randomly selected. Otherwise, all sets are used.

            Valid only when ``kind="sample"``.
        rng_seed : int or None, optional
            Random seed used for model selection.

            Valid only when ``kind="sample"``.
        n_bins : int, optional
            Number of bins used in statistical summaries of relative
            concentrations or fluxes.

            Must be an odd number.

            Valid only when ``kind="stats"``.
        show_fig : bool, optional
            If ``True``, display the figure.    
        output_data : bool, optional
            If ``True``, export the data used for plotting. Requires ``out_dir`` 
            to be specified.
        '''

        self.metabolite_sensitivity(out_dir, kind, metabolites, ylim, n_sets, 
                                    rng_seed, n_bins, show_fig, output_data)


    def metabolite_distribution(
            self, 
            out_dir=None, 
            metabolites='all',
            show_fig=False,
            output_data=False
        ):
        '''
        Plot distributions of metabolite concentrations (absolute values)
        during enzyme expression perturbations.
        
        Parameters
        ----------
        out_dir : str or None, optional
            Output directory.
        metabolites : list of str, optional
            Metabolites to plot. If ``all``, all metabolites are plotted.
        show_fig : bool, optional
            If ``True``, display the figure.
        output_data : bool, optional
            If ``True``, export the data used for plotting. Requires ``out_dir`` 
            to be specified.
        '''

        self._check_simulation_results()

        if metabolites == 'all':
            metabolites = self.metabs
        else:
            metabolites = sorted(set(metabolites) & set(self.metabs))

        l_only, r_only, *_ = self._plot_helper_pert_direction()
        
        for metab in metabolites:
            self._plot_helper_distribution_plot(
                out_dir, 'metab', metab, r_only, l_only, show_fig, output_data
            )
    

    def flux_sensitivity(
            self, 
            out_dir=None,
            kind='stats',
            reactions='all', 
            ylim=None,
            n_sets=100,
            rng_seed=None,
            n_bins=49,
            show_fig=False,
            output_data=False
        ):
        '''
        Plot responses of steady-state fluxes to enzyme expression
        perturbations.

        Parameters
        ----------
        out_dir : str or None, optional
            Output directory.
        kind: {"stats", "sample"}, optional
            Plot type. 
            - ``"stats"``: plot probability distributions of relative fluxes
            across perturbation levels.
            - ``"sample"``: plot trajectories from sampled models.
        reactions : list of str, optional
            Reactions to plot. If ``all``, all fluxes are plotted.
        ylim : 2-tuple or None, optional
            Lower and upper y-axis limits. If ``None``, the default range is
            ``(0.5, 1.5)`` for relative fluxes.
        n_sets : int, optional
            Number of sampled model sets used for plotting.

            If ``n_sets`` does not exceed the total number of sets, models are
            randomly selected. Otherwise, all sets are used.

            Valid only when ``kind="sample"``.
        rng_seed : int or None, optional
            Random seed used for model selection.

            Valid only when ``kind="sample"``.
        n_bins : int, optional
            Number of bins used in statistical summaries of relative
            concentrations or fluxes.

            Must be an odd number.

            Valid only when ``kind="stats"``.
        show_fig : bool, optional
            If ``True``, display the figure.
        output_data : bool, optional
            If ``True``, export the data used for plotting. Requires ``out_dir`` 
            to be specified.
        '''

        if self.v_sen:
            self._check_simulation_results()

            if reactions == 'all':
                reactions = self.rxns
            else:
                reactions = sorted(set(reactions) & set(self.rxns))

            plot_infors = self._plot_helper_pert_direction()

            ylim = ylim or (0.5, 1.5)

            if kind == 'stats':
                indices = range(len(self.sim_res))
                worker = self._plot_helper_sensitivity_stats_plot
            elif kind == 'sample':
                if n_sets <= len(self.sim_res):
                    indices = np.random.default_rng(rng_seed).choice(
                        range(len(self.sim_res)), n_sets, replace=False
                    )
                else:
                    indices = range(len(self.sim_res))
                worker = self._plot_helper_sensitivity_sample_plot
            else:
                raise ValueError('kind must be either "stats" or "sample".')
            
            # with Pool(n_jobs) as pool:
            #     results = []
            #     for rxn in reactions:
            #         res = pool.apply_async(
            #             func=worker,
            #             args=(
            #                 out_dir,
            #                 'flux',
            #                 rxn, 
            #                 *plot_infors,
            #                 indices,
            #                 ylim,
            #                 n_bins,
            #                 output_data
            #             )
            #         )
            #         results.append(res)

            #     for res in results:
            #         res.get()

            for rxn in reactions:
                worker(out_dir, 'flux', rxn, *plot_infors, indices, ylim, 
                       n_bins, show_fig, output_data)
            
        else:
            raise ValueError('Flux sensitivity results are not available because '
                             'flux_sensitivity=False in evaluate_robustness.')
        
        
    def flux_distribution(
            self, 
            out_dir=None, 
            reactions='all', 
            show_fig=False, 
            output_data=False
        ):
        '''
        Plot distributions of reaction fluxes (absolute values) during enzyme
        expression perturbations.

        Parameters
        ----------
        out_dir : str or None, optional
            Output directory.
        reactions : list of str, optional
            Reactions to plot. If ``all``, all fluxes are plotted.
        show_fig : bool, optional
            If ``True``, display the figure.
        output_data : bool, optional
            If ``True``, export the data used for plotting. Requires ``out_dir`` 
            to be specified.
        '''
        
        if self.v_sen:
            self._check_simulation_results()

            if reactions == 'all':
                reactions = self.rxns
            else:
                reactions = sorted(set(reactions) & set(self.rxns))

            l_only, r_only, *_ = self._plot_helper_pert_direction()

            for rxn in reactions:
                self._plot_helper_distribution_plot(
                    out_dir, 'flux', rxn, r_only, l_only, show_fig, output_data
                )
        else:
            raise ValueError('Flux distribution results are not available because '
                             'flux_sensitivity=False in evaluate_robustness.')


    def eigreal_sensitivity(self, out_dir=None, show_fig=False, output_data=False):
        '''
        Plot the average and median maximum real parts of Jacobian
        eigenvalues across all models as functions of enzyme perturbation.

        Parameters
        ----------
        out_dir : str or None, optional
            Output directory.
        show_fig : bool, optional
            If ``True``, display the figure.
        output_data : bool, optional
            If ``True``, export the data used for plotting. Requires ``out_dir`` 
            to be specified.
        '''

        self._check_simulation_results()

        (l_only, 
         r_only, 
         xticks, 
         xticklabels, 
         xlabel, 
         suffix) = self._plot_helper_pert_direction()
        
        fig, ax = plt.subplots()
        maxreal_all = []
        for i, data in enumerate(self.sim_res):
            if not r_only:
                maxreals_l = np.concatenate((
                    np.full(self.n_steps-data['maxreals_l'].size, np.nan), 
                    data['maxreals_l'][::-1]
                ))   # left padding
            
            if not l_only:
                maxreals_r = np.concatenate((
                    data['maxreals_r'],
                    np.full(self.n_steps-data['maxreals_r'].size, np.nan)
                ))   # right paddding

            if l_only:
                maxreal_all.append(maxreals_l)
            elif r_only:
                maxreal_all.append(maxreals_r)
            else:
                maxreal_all.append(
                    np.concatenate((maxreals_l, maxreals_r[1:]))
                )
        
        maxreal_all = pd.DataFrame(maxreal_all)
        sns.lineplot(maxreal_all.mean(), linewidth=5, label='Mean')
        sns.lineplot(maxreal_all.median(), linewidth=5, label='Median')
        sns.lineplot(
            x=maxreal_all.mean().index, y=0, 
            color='grey', linestyle='--', alpha=0.5
        )
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel(xlabel, fontsize=25)
        ax.set_ylabel(r'Jacobian $\lambda_{Re}^{max}$', fontsize=25)
        ax.tick_params(labelsize=20)
        plt.legend(frameon=False, fontsize=15)

        if show_fig:
            plt.show()

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

            fig.savefig(
                f'{out_dir}/maxeigreal_sen_by_{suffix}.jpg', 
                dpi=300, bbox_inches='tight'
            )

            if output_data:
                maxreal_all.index.name = 'Model'
                maxreal_all.to_csv(
                    f'{out_dir}/maxeigreal_sen_by_{suffix}.tsv', sep='\t'
                )

        plt.close()


    def eigreal_distribution(self, out_dir=None, show_fig=False, output_data=False):
        '''
        Plot distributions of the maximum real parts of Jacobian eigenvalues
        during enzyme perturbations.

        Parameters
        ----------
        out_dir : str or None, optional
            Output directory.
        show_fig : bool, optional
            If ``True``, display the figure.
        output_data : bool, optional
            If ``True``, export the data used for plotting. Requires ``out_dir`` 
            to be specified.
        '''

        self._check_simulation_results()

        l_only, r_only, *_ = self._plot_helper_pert_direction()
        
        self._plot_helper_distribution_plot(
            out_dir, 'maxeigreal', None, r_only, l_only, show_fig,output_data
        )

    
class Prograss():
    
    def __init__(self, mininterval=1, miniters=1, bar_format='{desc}{elapsed}'):
        self.mininterval = mininterval
        self.miniters = miniters
        self.bar_format = bar_format

        self.done = False


    def _refresh_pbar(self, pbar):
        '''
        Refresh pbar every second if not done.
        '''

        while not self.done:
            pbar.refresh()
            sleep(1)


    @contextmanager
    def context(self):
        with tqdm(
            mininterval=self.mininterval, 
            miniters=self.miniters, 
            bar_format=self.bar_format
        ) as pbar:
            refresher = Thread(target=self._refresh_pbar, args=(pbar,))
            refresher.start()
            
            try:
                yield pbar
            finally:
                self.done = True
                refresher.join()