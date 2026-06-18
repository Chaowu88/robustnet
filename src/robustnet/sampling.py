'''Define class for sampling model parameters.
'''


from functools import partial
from numbers import Number
import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean
import pymc as pm
from pytensor.compile.ops import as_op
import pytensor.tensor as pt
import logging
logger = logging.getLogger('pymc')
logger.setLevel(logging.CRITICAL)
logging.getLogger('pytensor').setLevel(logging.CRITICAL)
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)
from .kinetics import Fitter
from .utils import Prograss


class Sampler(Fitter):

    def __init__(
            self,
            model, 
            exclude_metabs=None, 
        ):
        '''
        Parameters
        ----------
        model : Model
            RobustNet model instance.
        exclude_metabs : list, optional
            Metabolites excluded from sampling.
        '''
        
        super().__init__(model, exclude_metabs, exclude_end_metabs=False)

        if len(self.metabs_only_in_rate) > 0:
            if len(self.metabs_only_in_rate) == 1:
                plural = ['it', 'has', 'is', '', 'its']
            else:
                plural = ['they', 'have', 'are', 's', 'their']

            logging.warning(
                f"{', '.join(self.metabs_only_in_rate)} appear in rate expressions "
                'but not in the stoichiometric matrix. This likely means '
                f"{plural[0]} {plural[2]} excluded or considered as "
                f"unbalanced metabolite{plural[3]}. When performing sampling, "
                f"{plural[0]} {plural[2]} treated as kinetic parameter{plural[3]}."
            )


    def _make_value_array(self, data, varnames, argname, method='geomean'):
        '''
        Make an array with elements ordered according to ``varnames``.

        If ``data`` is a dict or ``pandas.Series``, missing values
        (either absent or ``NaN``) are imputed by ``method``.
        '''
        
        if method == 'geomean':
            mean_fun = gmean
        elif method == 'mean':
            mean_fun = np.mean
        
        if isinstance(data, Number):
            return np.full(len(varnames), data)
        
        elif isinstance(data, (dict, pd.Series)):
            if isinstance(data, dict):
                mean = mean_fun(
                    list(filter(lambda v: v is not np.nan, data.values()))
                )
            else:
                mean = mean_fun(data.dropna())
            arr = np.array([data.get(var, mean) for var in varnames])
            arr = np.nan_to_num(arr, nan=mean)
            return arr
            
        elif data is None:
            return np.array([])
        
        else:
            raise TypeError(f'{argname} provided with an invalid data type.')
    

    def _prepare_prior(self, mu, sigma, init, name, label):
        '''
        Prepare prior ``mu``, ``sigma`` and ``initvalues`` for variables used in 
        sampling.
        '''
        
        return (
            self._make_value_array(
                mu, self.var_names[name], 
                f'{label}_mu' if label == 'ref_flux' else f'{label}_prior_mu'
            ),
            self._make_value_array(
                sigma, self.var_names[name], 
                f'{label}_sigma' if label == 'ref_flux' else f'{label}_prior_sigma'
            ),
            self._make_value_array(
                init, self.var_names[name], 
                f'{label}_initvalues'
            )
        )


    @staticmethod
    def log_transform(log_mu, log_sigma):
        '''
        Assume ``X ~ LogNormal(log_mu, log_sigma)``. 
        Compute the ``mu`` and ``sigma`` of the corresponding normal distribution:
        ``log(X) ~ Normal(mu, sigma)``.
        '''

        sigma = np.log10(1 + (log_sigma/log_mu)**2)**0.5
        mu = np.log10(log_mu) - 0.5*sigma**2*np.log(10)
        
        return mu, sigma
    

    def _to_ser(self, data, name):
        '''
        Convert a ``numpy.ndarray`` to a ``pandas.Series``
        '''

        return pd.Series(data, index=self.var_names[name])


    def sample_with_omics(
            self,
            ref_v_prior_mu,
            kparam_prior_mu,
            x_prior_mu,
            e_prior_mu,
            ref_v_prior_sigma=0.01,
            kparam_prior_sigma=0.1,
            x_prior_sigma=0.1,
            e_prior_sigma=0.001,
            ref_v_initvalues=None,
            kparam_initvalues=None,
            x_initvalues=None,
            e_initvalues=None,
            alpha=None,
            n_tunes=10000,
            n_samples=10000,
            n_chains=10,
            n_jobs=1
        ):
        '''
        Sample model parameters with fluxomics, metabolomics, proteomics data and
        enzyme kinetic parameters.

        Parameters
        ----------
        ref_v_prior_mu: dict or pandas.Series
            Reference-state flux distribution in units of mmol/L/s
            (cell-based). If ``None``, reference fluxes loaded by ``load_priors``
            are used.
        kparam_prior_mu : dict or pandas.Series
            Mean values of the prior distributions for kinetic parameters.
            Catalytic constants have units of 1/s, Michaelis, activation,
            and inhibition constants have units of mM, and equilibrium
            constants are dimensionless. Missing kinetic parameters are
            allowed. If ``None``, kinetic parameters loaded by ``load_priors``
            are used.
        x_prior_mu : dict, pandas.Series or None
            Mean values of prior metabolite concentrations in mM
            (cell-based). Missing metabolites are allowed. If ``None``,
            metabolomics data loaded by ``load_priors`` are used.
        e_prior_mu : dict, pandas.Series or None
            Mean values of prior enzyme concentrations in mM
            (cell-based). Missing enzymes are allowed. If ``None``,
            proteomics data loaded by ``load_priors`` are used.
        ref_v_prior_sigma : scalar, dict, pandas.Series, optional
            Standard deviations of reference-state fluxes. If a scalar is
            provided, the same value is used for all fluxes. Missing flux
            values are allowed when using a dict or ``pandas.Series``. If ``None``, 
            standard deviations from reference fluxes loaded by ``load_priors`` 
            are used. Defaults to ``0.01``.
        kparam_prior_sigma : scalar, dict or pandas.Series, optional
            Standard deviations of kinetic parameters. If a scalar is
            provided, the same value is used for all parameters. Missing
            parameter values are allowed. If ``None``, standard deviations
            from kinetic parameters loaded by ``load_priors``
            are used. Default to ``0.1``.
        x_prior_sigma : scalar, dict or pandas.Series or None, optional
            Standard deviations of metabolite concentrations. If a scalar is
            provided, the same value is used for all metabolites. Missing
            metabolite values are allowed. If ``None``, standard deviations
            from metabolomics data loaded by ``load_priors`` are used. Default 
            to ``0.1``.
        e_prior_sigma : scalar, dict or pandas.Series or None, optional
            Standard deviations of enzyme concentrations. If a scalar is
            provided, the same value is used for all enzymes. Missing enzyme
            values are allowed. If ``None``, standard deviations from
            proteomics data loaded by ``load_priors`` are used. Default to 
            ``0.001``.
        ref_v_initvalues : dict, pandas.Series or None, optional
            Initial values for reference flux sampling. Missing fluxes are
            allowed. If ``None``, ``ref_v_prior_mu`` is used.
        kparam_initvalues : dict, pandas.Series or None, optional
            Initial values for kinetic parameter sampling. Missing parameter
            values are allowed. If ``None``, ``kparam_prior_mu`` is used.
        x_initvalues : dict, pandas.Series or None, optional
            Initial values for metabolite concentration sampling. Missing
            metabolite values are allowed. If ``None``,
            ``x_prior_mu`` is used.
        e_initvalues : dict, pandas.Series or None, optional
            Initial values for enzyme concentration sampling. Missing enzyme
            values are allowed. If ``None``, ``e_prior_mu`` is used.
        alpha : float or None, optional
            Gaussian penalty strength used in parameter balancing. Larger
            values impose stronger penalties in log-posterior space.

            A reasonable choice is often on the same order of magnitude as
            ``1 / ref_flux_sigma**2``. If ``None``,
            ``geomean(1 / ref_flux_sigma**2)`` is used.
        n_tunes : int, optional
            Number of tuning iterations performed before sampling in each
            chain.
        n_samples : int, optional
            Number of samples drawn in each chain.
        n_chains : int, optional
            Number of sampling chains.
        n_jobs : int, optional
            Number of parallel jobs to run in parallel.
        '''
        
        if ref_v_prior_mu is None:
            ref_v_prior_mu = self.model.ref_fluxes
        if x_prior_mu is None:
            x_prior_mu = self.model.metabolomics
        if e_prior_mu is None:
            e_prior_mu = self.model.proteomics
        if kparam_prior_mu is None:
            kparam_prior_mu = self.model.kparameters
        
        if ref_v_prior_sigma is None:
            ref_v_prior_sigma = self.model.ref_fluxes_std
        if x_prior_sigma is None:
            x_prior_sigma = self.model.metabolomics_std
        if e_prior_sigma is None:
            e_prior_sigma = self.model.proteomics_std
        if kparam_prior_sigma is None:
            kparam_prior_sigma = self.model.kparameters_std
        
        if ref_v_initvalues is None:
            ref_v_initvalues = ref_v_prior_mu
        if kparam_initvalues is None:
            kparam_initvalues = kparam_prior_mu
        if x_initvalues is None:
            x_initvalues = x_prior_mu
        if e_initvalues is None:
            e_initvalues = e_prior_mu
        
        ref_v_mu, ref_v_sigma, ref_v_init = self._prepare_prior(
            ref_v_prior_mu, ref_v_prior_sigma, ref_v_initvalues, 
            'vs', 'ref_flux'
        )
        kparam_mu, kparam_sigma, kparam_init = self._prepare_prior(
            kparam_prior_mu, kparam_prior_sigma, kparam_initvalues, 
            'kparams', 'kparam'
        )
        x_mu, x_sigma, x_init = self._prepare_prior(
            x_prior_mu, x_prior_sigma, x_initvalues, 
            'ini_xs', 'mconc'
        )
        e_mu, e_sigma, e_init = self._prepare_prior(
            e_prior_mu, e_prior_sigma, e_initvalues, 
            'es', 'econc'
        )
        
        if alpha is None:
            alpha = gmean(1/ref_v_sigma**2, nan_policy='omit')
        
        simulate_v_partial = partial(self._simulate_flux, v_fun=self.v_fun_simple)
        simulate_v = as_op(
            itypes=[pt.dvector, pt.dvector, pt.dvector],
            otypes=[pt.dvector]
        )(simulate_v_partial)

        # sample
        with pm.Model(coords=self.var_names) as pm_model:
            kparams = pm.TruncatedNormal('kparam', mu=kparam_mu, 
                                         sigma=kparam_sigma, lower=0, 
                                         initval=kparam_init, dims='kparams')

            xs = pm.TruncatedNormal('x', mu=x_mu, sigma=x_sigma,
                                    lower=0, initval=x_init, dims='ini_xs')

            es = pm.TruncatedNormal('e', mu=e_mu, sigma=e_sigma,
                                    lower=0, initval=e_init, dims='es')

            vs = pm.Normal('v', mu=ref_v_mu, sigma=ref_v_sigma,
                           initval=ref_v_init, dims='vs')

            vs_sim = pm.Deterministic('v_sim', var=simulate_v(es, xs, kparams), 
                                      dims='vs')
            
            pm.Potential('v_const', var=-alpha*(vs_sim - vs)**2/2, dims='vs')

            print('Parameter sampling')
            with Prograss().context() as pbar:
                idata = pm.sample(
                    step=pm.Metropolis(),
                    tune=n_tunes,
                    draws=n_samples,
                    chains=n_chains,
                    discard_tuned_samples=True,
                    cores=n_jobs,
                    progressbar=False,
                    callback=SamplingCallback(pbar)
                )

        ref_v_mu = self._to_ser(ref_v_mu, 'vs')
        ref_v_sigma = self._to_ser(ref_v_sigma, 'vs')
        kparam_mu = self._to_ser(kparam_mu, 'kparams')
        kparam_sigma = self._to_ser(kparam_sigma, 'kparams')
        x_mu = self._to_ser(x_mu, 'ini_xs')
        x_sigma = self._to_ser(x_sigma, 'ini_xs')
        e_mu = self._to_ser(e_mu, 'es')
        e_sigma = self._to_ser(e_sigma, 'es')
        
        return (idata, ref_v_mu, ref_v_sigma, kparam_mu, kparam_sigma,
                x_mu, x_sigma, e_mu, e_sigma)


class SamplingCallback():
    
    def __init__(self, pbar):
        self.pbar = pbar
        self.traces = {}


    def __call__(self, trace, draw):
        self.traces[draw.chain] = trace
        self.pbar.set_description(f'{sum(map(len, self.traces.values()))} samples')