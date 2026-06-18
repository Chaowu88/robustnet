'''Define classes for metabolic kinetics.
'''


import numpy as np
import pandas as pd
from sympy import symbols, Matrix, lambdify
from scipy.stats.mstats import gmean
from scipy.linalg import null_space, pinv
from scipy.optimize import minimize
import logging
from .utils import Prograss


class Simulator:

    def __init__(
            self, 
            model, 
            exclude_metabs=None, 
            exclude_end_metabs=True, 
            exclude_stoy_only_metabs=False
        ):
        '''
        Parameters
        ----------
        model : Model
            RobustNet model instance.
        exclude_metabs : list, optional
            Metabolites excluded from simulation.
        exclude_end_metabs : bool, optional
            If ``True``, exclude initial substates and final products from
            stoichiometric matrix.
        exclude_stoy_only_metabs : bool, optional
            If ``True``, exclude metabolites that appear in the stoichiometric
            matrix but not in reaction rate expressions. Such metabolites
            impair numerical stability and should be excluded from robustness
            simulations.
        '''
        
        self.model = model
        self.exclude_metabs = exclude_metabs or []
        
        if exclude_end_metabs:
            self.exclude_metabs.extend(
                self.model.initial_substrates(self.exclude_metabs)+
                self.model.final_products(self.exclude_metabs)
            )

        self.stoy_mat = self.model.stoichiometric_matrix(self.exclude_metabs)

        enz_vars = []
        metab_vars_in_rate = []
        kparams = []
        rate_exprs = []
        for rxnid in self.stoy_mat.columns:
            ratelaw_info = self.model.reaction_info[rxnid].ratelaw
            if ratelaw_info.enz is not None:
                enz_vars.append(ratelaw_info.enz)
            metab_vars_in_rate.extend(ratelaw_info.metabs)
            kparams.extend(ratelaw_info.kparams)
            rate_exprs.append(ratelaw_info.expr)

        if exclude_stoy_only_metabs:
            self.metabs_only_in_stoy = sorted(
                set(self.stoy_mat.index) - set(metab_vars_in_rate)
            )
            self.stoy_mat = self.model.stoichiometric_matrix(
                self.exclude_metabs+self.metabs_only_in_stoy
            )
        
        self.ini_subs = []
        self.fin_pros = []
        
        self.enz_vars = sorted(set(enz_vars))
        
        self.metabs_only_in_rate = sorted(
            set(metab_vars_in_rate) - set(self.stoy_mat.index)
        )
        
        self.metab_vars = sorted(self.stoy_mat.index)
        self.kparams = sorted(set(kparams))   
        self.rate_exprs = rate_exprs   
        self.rxn_vars = sorted(self.stoy_mat.columns)
        
        self.var_names = {
            'ini_xs': self.metab_vars,
            'kparams': self.kparams+self.metabs_only_in_rate,
            'vins': [f'vin_{m}' for m in self.ini_subs],
            'vouts': [f'vout_{m}' for m in self.fin_pros],
            'es': self.enz_vars,
            'eparams': [f'{k}_{e}' for e in self.enz_vars 
                                   for k in ['kf', 'km', 'kl']],
            'vs': self.rxn_vars
        }
        
        self.v_fun = self._lambdify_v()
        self.v_fun_simple = self._lambdify_v(exchange=False)
        self.e_fun = self._lambdify_e()

        self.aug_stoy_mat = self._augment_stoichiometric_matrix()
        (self.stoy_mat_rank, 
         self.red_aug_stoy_mat, 
         self.link_mat) = self._rank_decomposition()
    
    
    def _get_involved_enzymes(self, exclude_enzs):
        return sorted(set(self.var_names['es']) - set(exclude_enzs))


    def _augment_stoichiometric_matrix(self):
        '''
        Augment the stoichiometric matrix by adding influxes (``vins``) for
        initial substrates and effluxes (``vouts``) for final products.

        Note that adding influxes and effluxes may change the rank of the original
        stoichiometric matrix.
        '''
        
        left_mat = pd.DataFrame(
            0., 
            index = self.stoy_mat.index, 
            columns = self.ini_subs + self.fin_pros
        )
        for sub in self.ini_subs:
            left_mat.at[sub, sub] = 1.
        for pro in self.fin_pros:
            left_mat.at[pro, pro] = -1.
        
        return pd.concat((left_mat, self.stoy_mat), axis=1)
    

    def _rank_decomposition(self):
        '''
        Decompose the stoichiometric matrix into a link matrix and a reduced
        stoichiometric matrix, both with full rank. Metabolites included in
        the reduced stoichiometric matrix are treated as independent.

        Note that the assignments of independent and dependent metabolites within a
        conservation relationship are arbitrary. For example, if
        ``A + B = constant``, either ``A`` or ``B`` may be selected as the
        independent metabolite, while the other is treated as dependent.
        '''

        C_mat, F_mat = Matrix(self.aug_stoy_mat.values).T.rank_decomposition()
        red_aug_stoy_mat = np.array(C_mat.T).astype(float)
        link_mat = np.array(F_mat.T).astype(float)
        stoy_mat_rank = red_aug_stoy_mat.shape[0]

        return stoy_mat_rank, red_aug_stoy_mat, link_mat


    def _lambdify_v(self, exchange=True, derivative=False):
        '''
        Get callable reaction rate functions, including optional influxes
        (``vins``) and effluxes (``vouts``).

        The callable takes inputs in the order:
        ``[enz_vars] + [metab_vars] + [kparams] + [vins] + [vouts]`` 
        and returns fluxes with shape:
        ``(len(vins) + len(vouts) + n_reactions, 1)``.
        
        If ``derivative=True``, callable derivatives with respect to
        metabolites (``dv_dx``) and enzymes (``dv_de``) are also returned.
        Their inputs are identical to the reaction-rate callable. 

        The output shapes are:
        ``dvdx_fun``: ``(len(vins) + len(vouts) + n_reactions, n_metab_vars)`` 
        ``dvde_fun``: ``(len(vins) + len(vouts) + n_reactions, n_enz_vars)``

        Note that ``vins`` and ``vouts`` may be empty.

        Parameters
        ----------
        exchange : bool
            If ``True``, influxes and effluxes are included in the system,
            both in the function inputs and outputs.
        derivative  : bool
            If ``True``, callable derivatives with respect to metabolites
            (``x``) and enzymes (``e``) are also returned.
        '''

        enzs_sym = symbols(self.var_names['es'])
        metabs_sym = symbols(self.var_names['ini_xs']) 
        kparams_sym = symbols(self.var_names['kparams'])
        vins_sym = symbols(self.var_names['vins'])
        vouts_sym = symbols(self.var_names['vouts'])
        
        if exchange:
            input_sym = enzs_sym + metabs_sym + kparams_sym + vins_sym + vouts_sym
            v_sym = Matrix(vins_sym+vouts_sym+self.rate_exprs)
        else:
            input_sym = enzs_sym + metabs_sym + kparams_sym
            v_sym = Matrix(self.rate_exprs)
        
        v_fun = lambdify(input_sym, v_sym, modules='numpy')   # jax?

        if derivative == False:
            return v_fun
        else:
            dvdx_sym = v_sym.jacobian(metabs_sym)
            dvdx_fun = lambdify(input_sym, dvdx_sym, modules='numpy')   # jax?

            dvde_sym = v_sym.jacobian(enzs_sym)
            dvde_fun = lambdify(input_sym, dvde_sym, modules='numpy')   # jax?

            return v_fun, dvdx_fun, dvde_fun
            

    def _lambdify_e(self):
        '''
        Get a callable Hill equation function. 
        
        The callable takes inputs in the order: 
        ``[t] + [eparams]``
        where ``eparams`` includes ``kf``, ``km``, and ``kl`` for each enzyme.

        The output is enzyme concentrations with shape:
        ``(n_enz_vars, 1)``.
        The order of enzymes follows ``self.enz_vars``.
        '''

        eparams_sym = symbols(self.var_names['eparams'])
        e_expr = []
        t = symbols('t')
        for i in range(0, len(eparams_sym), 3):
            kf, km, kl = eparams_sym[i:i+3]
            e_expr.append(kf*t/(km + t) + kl)
        
        input_sym = [t] + eparams_sym
        e_sym = Matrix(e_expr)

        e_fun = lambdify(input_sym, e_sym, modules='numpy')
        
        return e_fun
    

    @staticmethod
    def _check_nan(x, label):
        if np.isnan(x).any():
            logging.warning(f'NaN values detected in simulated {label}. '
                            'Try running again or checking the model.')
    

    @staticmethod
    def _simulate_flux(es, xs, kparams, v_fun, check_nan=False):
        '''
        Parameters
        ----------
        es : numpy.array
            Enzyme concentrations in order of ``self.enz_vars``.
        xs : numpy.array
            Metabolite concentrations in order of ``self.metab_vars``.
        kparams : list
            Kinetic parameters in order of ``self.kparams``.
        v_fun : callable
            Function that that computes reaction fluxes from metabolite
            concentrations and other parameters.
        check_nan : bool, optional
            If ``True``, check whether the simulation results contain NaN
            values.

        Returns
        -------
        v : numpy.array
            Simulated fluxes with shape ``(n_rxns,)``.
        '''

        v = v_fun(*es, *xs, *kparams).flatten()

        if check_nan:
            Simulator._check_nan(v, 'fluxes')

        return v
        

    def _check_bounds(self, lb, ub):
        if lb > ub:
            raise ValueError('Lower bound must not be greater than upper bound.')
        

    def _get_bounds(self, bounds, varnames):
        '''
        Get lower and upper bounds for variables listed in ``varnames``. 
        
        For variables not explicitly specified in ``bounds``, the minimum
        lower bound and maximum upper bound from the provided bounds are used.

        Parameters
        ----------
        bounds : 2-tuple or dict of 2-tuple
            Bounds for variables.
        varnames: list
            Variable names for which bounds are retrieved.
        '''
        
        if isinstance(bounds, tuple):
            self._check_bounds(bounds[0], bounds[1])
            var_lbs = np.full(len(varnames), bounds[0])
            var_ubs = np.full(len(varnames), bounds[1])
        
        elif isinstance(bounds, dict):
            lbs = []
            ubs = []
            for lb, ub in bounds.values():
                self._check_bounds(lb, ub)
                lbs.append(lb)
                ubs.append(ub)
            min_lb = min(lbs)
            max_ub = max(ubs)
            
            var_lbs = []
            var_ubs = []
            for var in varnames:
                if var in bounds:
                    var_lbs.append(bounds[var][0])
                    var_ubs.append(bounds[var][1])
                else:
                    var_lbs.append(min_lb)
                    var_ubs.append(max_ub)
            var_lbs = np.array(var_lbs)
            var_ubs = np.array(var_ubs)
        
        elif bounds is None:
            var_lbs = np.array([])
            var_ubs = np.array([])
        
        else:
            raise TypeError('Bounds provided with an invalid data type.')
        
        return var_lbs, var_ubs
    

    def _get_initial_values(self, varnames, lbs, ubs, rng_seed=None, ref=None):
        '''
        Get initial values for variables listed in ``varnames``. 
        
        If initial values are not provided in ``ref``, random values bounded
        by ``lbs`` and ``ubs`` are generated.

        Parameters
        ----------
        varnames : list
            Variable names.
        lbs : numpy.array
            Lower bounds in the same order as ``varnames``.
        ubs : numpy.array
            Upper bounds in the same order as ``varnames``.
        rng_seed : float or None.
            Seed of random number generator.
        ref : dict, pandas.Series or None
            Reference values used as initial guesses.
        '''

        rng = np.random.default_rng(rng_seed)

        if isinstance(ref, (dict, pd.Series)):
            ini = np.array([ref.get(var, rng.uniform(lb, ub)) 
                            for var, lb, ub in zip(varnames, lbs, ubs)])
        elif ref is None:
            ini = rng.uniform(lbs, ubs)
        else:
            raise TypeError('Initial values provided with an invalid '
                            'data type.')
        
        return ini
        

class Fitter(Simulator):

    def __init__(
            self, 
            model, 
            exclude_metabs=None,
            exclude_end_metabs=True
        ):
        '''
        Parameters
        ----------
        model : Model
            RobustNet model instance.
        exclude_metabs : list, optional
            Metabolites excluded from fitting.
        exclude_end_metabs : bool, optional
            If ``True``, exclude initial substates and final products from
            stoichiometric matrix
        '''

        super().__init__(model, exclude_metabs, exclude_end_metabs)


    def _check_fluxomics(self):
        '''
        Get the fluxomics data used for flux fitting.

        Returns
        -------
        flux_exp_fit: 1-D numpy.array
            Experimental fluxomics measurements actually used for fitting.
        flux_exp_std_fit : 1-D numpy.array
            Standard deviations of the experimental fluxomics measurements actually 
            used for fitting.
        fluxes_fit : list
            Reaction IDs included in the fitting process.
        indices : list
            Indices of the fitted reaction IDs in ``self.var_names['vs']``.
        '''
        
        flux_exp = self.model.fluxomics
        fluxes_fit = sorted(set(self.var_names['vs']) & set(flux_exp.index))
        if len(fluxes_fit) > 0:
            logging.info(f'Fluxes to fit: {", ".join(fluxes_fit)}')

            flux_exp_fit = flux_exp[fluxes_fit].values
            indices = [self.var_names['vs'].index(r) for r in fluxes_fit]
            
            if hasattr(self.model, 'fluxomics_std'):
                flux_exp_std = self.model.fluxomics_std
                flux_exp_std_fit = flux_exp_std[fluxes_fit].values
                nonzero_std_gmean = gmean(
                    flux_exp_std[flux_exp_std>0], nan_policy='omit'
                )
                to_replace = (flux_exp_std_fit == 0) | np.isnan(flux_exp_std_fit)
                flux_exp_std_fit[to_replace] = nonzero_std_gmean
            else:
                flux_exp_std_fit = None

            return flux_exp_fit, flux_exp_std_fit, fluxes_fit, indices
        else:
            raise ValueError('None of the fluxes to fit are provided with '
                             'experimental data.')


    def _check_optimizer(self, optimizer):
        optimizer = optimizer.lower()
        if optimizer not in ['scipy', 'nlopt']:
            raise ValueError('Optimizer should be selected from '
                             '{"scipy", "nlopt"}.')
        
        return optimizer
        

    def _check_method(self, method):
        method = method.upper()
        if method not in ['COBYQA', 'SLSQP']:
            raise ValueError('Optimization method should be selected from '
                             '{"COBYQA", "SLSQP"}')
        
        return method
    

    def fit_reference_fluxes(
            self, 
            bounds, 
            optimizer='scipy',
            method='COBYQA',
            tol=1e-4,
            maxtime=600
        ):
        '''
        Estimate reference fluxes by fitting to fluxomics data.

        Parameters
        ----------
        bounds : 2-tuple or dict of 2-tuple
            Lower and upper bounds used during fitting. If a tuple is
            provided, the same bounds are applied to all fluxes. If a dict
            is provided, bounds can be specified for individual fluxes,
            which is useful for defining flux reversibility. For example,
            Setting ``lower_bound >= 0`` or ``upper_bound <= 0`` enforces
            irreversibility in the forward or reverse direction,
            respectively.

            Fluxes without explicit bounds use the minimum lower bound and 
            maximum upper bound from the provided bounds.
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
        maxtime : float, optional
            Maximum optimization time (s) allowed. Only applicable when
            ``optimizer="nlopt"``.
        '''
        
        optimizer = self._check_optimizer(optimizer)
        method = self._check_method(method)
        
        flux_exp, flux_exp_std, fluxes_fit, flux_indices = self._check_fluxomics()

        if flux_exp_std is not None:
            weight_mat = np.diag(1/flux_exp_std**2)
        else:
            weight_mat = np.eye(flux_exp.size)
        
        null_mat = null_space(self.stoy_mat.values)
        if null_mat.size == 0:
            raise ValueError('The mass balance constraints of current network yield'
                             ' only trivial solution, i.e., all fluxes are zero.')
        
        n_netfluxes, n_freefluxes = null_mat.shape
        
        trans_mat = np.zeros((len(flux_indices), n_netfluxes))
        trans_mat[np.arange(len(flux_indices)), flux_indices] = 1
        
        v_lbs, v_ubs = self._get_bounds(bounds, self.var_names['vs'])
        v_ini = self._get_initial_values(
            self.var_names['vs'], v_lbs, v_ubs, ref=self.model.fluxomics
        )
        u_ini = pinv(null_mat)@v_ini

        print('Reference flux fitting')
        with Prograss().context() as pbar:
            if optimizer == 'scipy':
                opt_vars, opt_obj = self._fit_flux_with_scipy(
                    method, tol, u_ini, null_mat, trans_mat, weight_mat, 
                    flux_exp, v_lbs, v_ubs, pbar
                )
            elif optimizer == 'nlopt':
                opt_vars, opt_obj = self._fit_flux_with_nlopt(
                    method, tol, maxtime, u_ini, null_mat, trans_mat, weight_mat, 
                    flux_exp, v_lbs, v_ubs, n_freefluxes, n_netfluxes, pbar
                )

        hess = 2*null_mat.T@trans_mat.T@weight_mat@trans_mat@null_mat
        dof = flux_exp.size - n_freefluxes
        
        if dof > 0:
            u_cov = pinv(hess)*opt_obj/dof
            v_cov = null_mat@u_cov@null_mat.T
            v_std = np.diag(v_cov)**0.5    
            opt_vs_std = pd.Series(v_std, index=self.var_names['vs'])
        else:
            opt_vs_std = None
        
        opt_vs = pd.Series(null_mat@opt_vars, index=self.var_names['vs'])

        flux_exp = pd.Series(flux_exp, index=fluxes_fit)
        
        return opt_vs, opt_vs_std, opt_obj, dof, flux_exp
    

    def _fit_flux_with_scipy(self, method, tol, u_ini, N, T, W, v_exp, 
                             v_lbs, v_ubs, pbar):
        def obj(u, N, T, W, v_exp):
            resid = T@N@u - v_exp
            obj = resid@W@resid
            return obj

        def obj_jac(u, N, T, W, v_exp):
            resid = T@N@u - v_exp
            grad = 2*resid@W@T@N
            return grad

        def lb_consts(u, N, lbs): return N@u - lbs
            
        def lb_consts_jac(u, N, lbs): return N
        
        def ub_consts(u, N, ubs): return ubs- N@u
        
        def ub_consts_jac(u, N, ubs): return -N

        def callback(vars):
            ssr = obj(vars, N, T, W, v_exp)
            if ssr > 1e5:
                pbar.set_description(f'SSR {ssr:.3e}')
            else:
                pbar.set_description(f'SSR {ssr:.3f}')

        if method == 'COBYQA':
            jac=None
        elif method == 'SLSQP':
            jac=obj_jac
        
        res = minimize(
            fun=obj,
            x0=u_ini,
            args=(N, T, W, v_exp),
            method=method,
            jac=jac,
            constraints=[
                {'type': 'ineq', 
                 'fun': lb_consts, 
                 'jac': lb_consts_jac, 
                 'args': (N, v_lbs)},
                {'type': 'ineq', 
                 'fun': ub_consts, 
                 'jac': ub_consts_jac, 
                 'args': (N, v_ubs)}
            ],
            tol=tol,
            callback=callback
        )

        return res.x, res.fun
    

    def _fit_flux_with_nlopt(self, method, tol, maxtime, u_ini, N, T, W, v_exp,
                             v_lbs, v_ubs, n_vars, n_constrs, pbar):
        try:
            import nlopt
        except ModuleNotFoundError:
            raise ValueError('NLopt needs to be installed first.')
        
        res_vars = []

        def obj(u, grad, N, T, W, v_exp, pbar):
            res_vars.append(u.copy())
            resid = T@N@u - v_exp
            if grad.size > 0:
                grad[:] = 2*resid@W@T@N
            obj = resid@W@resid
            if obj > 1e5:
                pbar.set_description(f'SSR {obj:.3e}')
            else:
                pbar.set_description(f'SSR {obj:.3f}')
            return obj
        
        def make_obj():
            return lambda u, grad: obj(u, grad, N, T, W, v_exp, pbar)

        def lb_consts(res, u, grad, N, lbs):
            if grad.size > 0:
                grad[:,:] = -N
            res[:] = lbs - N@u

        def make_lb_consts():
            return lambda res, u, grad: lb_consts(res, u, grad, N, v_lbs)
        
        def ub_consts(res, u, grad, N, ubs):
            if grad.size > 0:
                grad[:,:] = N
            res[:] = N@u - ubs

        def make_ub_consts():
            return lambda res, u, grad: ub_consts(res, u, grad, N, v_ubs)
        
        if method == 'COBYQA':
            raise NotImplementedError(
                'COBYQA in NLopt is not available for this problem, '
                'as it currently only supports bound constraints '
                'and does not handle general inequality constraints.'
            )
        elif method == 'SLSQP':
            opt_method = nlopt.LD_SLSQP
        
        opt = nlopt.opt(opt_method, n_vars)
        opt.set_min_objective(make_obj())
        opt.add_inequality_mconstraint(
            make_lb_consts(), np.full(n_constrs, 1e-8)
        )
        opt.add_inequality_mconstraint(
            make_ub_consts(), np.full(n_constrs, 1e-8)
        )
        opt.set_xtol_rel(tol)
        opt.set_maxtime(maxtime)

        try:
            opt_vars = opt.optimize(u_ini)
        except nlopt.RoundoffLimited:
            # nlopt somehow may raise roundoff-limited exception,
            # but the results are still useful
            opt_vars = res_vars[-1]
        opt_obj = opt.last_optimum_value()
        
        return opt_vars, opt_obj
