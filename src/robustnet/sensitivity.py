'''Define class for robusness analysis.
'''


import platform
from functools import partial
import numpy as np
import pandas as pd
from scipy.linalg import eigvals, inv, pinv
from multiprocess import Pool
import logging
from .kinetics import Simulator


DEFAULT_FOLD_CHANGE = (0.2, 5)


class EnsembleSimulator(Simulator):
    '''
    Implement the continuation method to evaluate the effects of parameter
    perturbations on a metabolic system.

    Enzyme concentrations are treated as perturbation parameters in the
    robustness analysis.
    '''

    def __init__(self, model, exclude_metabs=None):
        '''
        Parameters
        ----------
        model : Model
            RobustNet model instance.
        exclude_metabs : list, optional
            Metabolites excluded from analysis.
        '''

        super().__init__(model, exclude_metabs, exclude_stoy_only_metabs=True)
        
        if len(self.metabs_only_in_rate) > 0:
            plural = self._get_plural(self.metabs_only_in_rate)

            logging.warning(
                f"{', '.join(self.metabs_only_in_rate)} appear{plural[5]} "
                'in rate expressions but not in the stoichiometric matrix. '
                f"This likely means {plural[0]} {plural[2]} excluded or "
                f"considered as unbalanced metabolite{plural[3]}. When "
                f"performing robustness analysis, {plural[0]} {plural[2]} "
                f"treated as kinetic parameter{plural[3]}, and {plural[4]} "
                f"concentration{plural[3]} will not be simulated."
            )

        if len(self.metabs_only_in_stoy) > 0:
                plural = self._get_plural(self.metabs_only_in_stoy)
                logging.warning(
                    f"{', '.join(self.metabs_only_in_stoy)} appear{plural[5]} "
                    'in the stoichiometric matrix but not in rate expressions. '
                    f'{plural[0].capitalize()} will be excluded from robustness '
                    'analysis to maintain numerical stability.'
                )
        
        (self.v_fun, 
         self.dvdx_fun, 
         self.dvde_fun) = self._lambdify_v(derivative=True)
        

    def _get_plural(self, test_list):
        if len(test_list) == 1:
            return ['it', 'has', 'is', '', 'its', 's']
        else:
            return ['they', 'have', 'are', 's', 'their', '']
        

    def _select_parameter_sets(self, n_models, rng_seed=None):
        '''
        Check whether sufficient parameter sets are available for ``n_models``.

        If enough parameter sets are available, randomly select ``n_models``
        sets. Otherwise, use all available parameter sets.
        '''

        sample_indices = self.model.kin_param_sets.index.to_list()
        if n_models <= len(sample_indices):
            choices = np.random.default_rng(rng_seed).choice(
                sample_indices, n_models, replace=False
            )
        else:
            logging.warning('Requested n_model exceeds the number of available '
                            f'parameter sets. At most {sample_indices} models can '
                            'be generated.')
            choices = sample_indices

        ini_x_sets = self.model.init_metab_sets.iloc[choices, :]
        e_sets = self.model.enz_conc_sets.iloc[choices, :]
        kparam_sets = self.model.kin_param_sets.iloc[choices, :]
        vin_sets = self.model.influx_sets.iloc[choices, :]
        vout_sets = self.model.efflux_sets.iloc[choices, :]
        
        return ini_x_sets, e_sets, kparam_sets, vin_sets, vout_sets


    def _check_metabolite_set_inclusion(self, ini_x_set):
        '''
        Check whether sampled initial concentrations are available for all 
        metabolites included in the analysis, and reorder them 
        according to ``self.stoy_mat.index``.
        '''

        missing = set(self.var_names['ini_xs']).difference(ini_x_set.columns)
        if missing:
            raise ValueError('No sampled data found for metabolite(s): '
                             f"{', '.join(sorted(missing))}.")
        else:
            return ini_x_set[self.var_names['ini_xs']]
    

    def _check_enzyme_set_inclusion(self, e_conc_set):
        '''
        Check whether sampled concentrations are available for all enzymes
        included in the analysis, and reorder them according to
        ``self.enz_vars``.
        '''

        missing = set(self.var_names['es']).difference(e_conc_set.columns)
        if missing:
            raise ValueError('No sampled data found for enzyme(s): '
                             f"{', '.join(sorted(missing))}")
        else:
            return e_conc_set[self.var_names['es']]
    

    def _check_kinetic_parameter_set_inclusion(self, kparam_set):
        '''
        Check whether sampled values are available for all kinetic parameters
        required in the analysis, and reorder them according to
        ``self.kparams``.
        '''

        missing = set(self.var_names['kparams']).difference(kparam_set.columns)
        if missing:
            raise ValueError('No sampled data found for kinetic parameter(s): '
                             f"{', '.join(sorted(missing))}")
        else:        
            return kparam_set[self.var_names['kparams']]
    

    def _check_influx_set_inclusion(self, vin_set):
        '''
        Check whether sampled influxes are available for all initial
        substrates, and reorder them according to ``self.ini_subs``.
        '''
        
        missing = set(self.var_names['vins']).difference(vin_set.columns)
        if missing:
            raise ValueError('No sampled data found for influx(es): '
                             f"{', '.join(sorted(missing))}")
        else:
            return vin_set[self.var_names['vins']]
    

    def _check_efflux_set_inclusion(self, vout_set):
        '''
        Check whether sampled effluxes are available for all final products,
        and reorder them according to ``self.fin_pros``.
        '''
        
        missing = set(self.var_names['vouts']).difference(vout_set.columns)
        if missing:
            raise ValueError('No sampled data for efflux(es): '
                             f"{', '.join(sorted(missing))}")
        else:
            return vout_set[self.var_names['vouts']]
        

    def _check_fold_change(self, bounds):
        if (1 - bounds[0])*(1 - bounds[1]) > 0:
            raise ValueError('Range specified by fold_change must span across 1.')


    def _get_perturbation_bounds(self, perturb_enzymes, fold_change):
        '''
        Get relative enzyme concentration levels corresponding to the left and
        right perturbation bounds.
        '''

        invalid_enzs = []

        if perturb_enzymes == 'all':
            pert_fc = {e: DEFAULT_FOLD_CHANGE for e in self.var_names['es']}
        elif isinstance(perturb_enzymes, list):
            pert_fc = {}    
            for e in perturb_enzymes:
                if e in self.var_names['es']:
                    pert_fc[e] = DEFAULT_FOLD_CHANGE
                else:
                    invalid_enzs.append(e)
        else:
            raise ValueError('perturb_enzymes must be "all" or a list of enzyme '
                             'IDs.')
        
        if isinstance(fold_change, tuple):
            self._check_fold_change(fold_change)
            pert_fc_update = {e: fold_change for e in pert_fc}
        elif isinstance(fold_change, dict):
            pert_fc_update = {}
            for e, bounds in fold_change.items():
                if e in self.var_names['es']:
                    self._check_fold_change(bounds)
                    pert_fc_update[e] = bounds
                else:
                    invalid_enzs.append(e)
        else:
            raise ValueError('fold_change must be a tuple or dictionary.')
            
        pert_fc.update(pert_fc_update)

        if len(invalid_enzs) > 0:
            logging.warning(f'Enzyme(s) {", ".join(sorted(set(invalid_enzs)))} '
                             'not found and will be ignored.')
        
        if len(pert_fc) == 0:
            raise ValueError('No valid enzymes were found for perturbation.')

        pert_fc_str = [f'{e} {bnd}' for e, bnd in pert_fc.items()]
        logging.info(f'Perturb enzyme {", ".join(pert_fc_str)}')
        
        lbs = [pert_fc[e][0] if e in pert_fc else 1. for e in self.var_names['es']]
        rbs = [pert_fc[e][1] if e in pert_fc else 1. for e in self.var_names['es']]

        return pert_fc, np.array(lbs), np.array(rbs)


    def _get_split_chunks(self, *all_sets, n_chunks):
        chunks = []
        for sets in all_sets:
            chunks.append(np.array_split(sets, n_chunks))
        
        return tuple(chunks)


    def _compute_v(self, ss_x, e, kparams, vins, vouts, v_fun):
        '''
        Compute the fluxes (including influxes and effluxes).

        Parameters
        ----------
        ss_x : numpy.array
            Steady-state metabolite concentrations ordered according to
            ``self.stoy_mat.index``.
            If two-dimensional, columns correspond to metabolites also
            in order of ``self.stoy_mat.index``.
        e : numpy.array
            Enzyme concentrations ordered according to ``self.enz_vars``.
            If two-dimensional, columns correspond to enzymes also
            in order of ``self.enz_vars``.
        kparams : numpy.array
            Kinetic parameters ordered according to ``self.kparams``.
        vins : numpy.array
            Influx rates for initial substrates ordered according to
            ``self.ini_subs``.
        vouts : numpy.array
            Efflux rates for final products ordered according to
            ``self.fin_pros``.
        v_fun : callable
            Callable function that computes the fluxes.
        '''

        return v_fun(*e, *ss_x, *kparams, *vins, *vouts).flatten()


    def _compute_dvdx(self, ss_x, e, kparams, vins, vouts, dvdx_fun):
        '''
        Compute the derivative of fluxes (including influxes and effluxes) with 
        respect to metabolite concentrations.

        Parameters
        ----------
        ss_x : numpy.array
            Steady-state metabolite concentrations ordered according to
            ``self.stoy_mat.index``.
        e : numpy.array
            Enzyme concentrations ordered according to ``self.enz_vars``.
        kparams : numpy.array
            Kinetic parameters ordered according to ``self.kparams``.
        vins : numpy.array
            Influx rates for initial substrates ordered according to
            ``self.ini_subs``.
        vouts : numpy.array
            Efflux rates for final products ordered according to
            ``self.fin_pros``.
        dvdx_fun : callable
            Callable function that computes derivatives of fluxes with
            respect to metabolite concentrations.
        '''

        return dvdx_fun(*e, *ss_x, *kparams, *vins, *vouts)
    

    def _compute_dvde(self, ss_x, e, kparams, vins, vouts, dvde_fun):
        '''
        Compute the derivative of fluxes (including influxes and effluxes) with 
        respect to enzyme concentrations.

        Parameters
        ----------
        ss_x : numpy.array
            Steady-state metabolite concentrations ordered according to
            ``self.stoy_mat.index``.
        e : numpy.array
            Enzyme concentrations ordered according to ``self.enz_vars``.
        kparams : numpy.array
            Kinetic parameters ordered according to ``self.kparams``.
        vins : numpy.array
            Influx rates for initial substrates ordered according to
            ``self.ini_subs``.
        vouts : numpy.array
            Efflux rates for final products ordered according to
            ``self.fin_pros``.
        dvde_fun : callable
            Callable function that computes derivatives of fluxes with
            respect to enzyme concentrations.
        '''

        return dvde_fun(*e, *ss_x, *kparams, *vins, *vouts)
    
    
    def _euler(self, x_ini, v_ini, e_ini, steps, Sr, L, 
               compute_dvdx, compute_dvde, compute_v, flux_sensitivity=True, 
               check_jacobian=True, tol=-1e-9, check_metabolite=True):
        '''
        Solve an IVP using the Euler method.

        Parameters
        ----------
        steps : 2-D numpy.array
            Perturbation step sizes for enzymes with shape
            ``(n_steps, n_enzymes)``.
        flux_sensitivity : bool, optional
            If ``True``, derivatives of steady-state fluxes with respect to
            enzyme concentrations are also computed.
        check_jacobian : bool, optional
            If ``True``, the maximum real parts of Jacobian eigenvalues are
            checked during perturbation, and matrix inversion is used. 
            If ``False``, the pseudoinverse is used when the Jacobian is
            singular or noninvertible.
        check_metabolite : bool, optional
            If ``True``, metabolite concentrations are checked to remain
            positive during perturbation. 
            
        Returns
        -------
        x_sol : 2-D numpy.array
            Simulated metabolite concentrations with shape
            ``(n_steps, self.stoy_mat.shape[0])``.
            Rows correspond to perturbation steps and columns correspond to
            metabolites.
            ``x_sol`` stores values for steps ``1`` through ``n_steps``.
            An empty array indicates an invalid model at the initial state.
        v_sol : 2-D numpy.array
            Simulated fluxes, including influxes and effluxes, with shape
            ``(n_steps, self.aug_stoy_mat.shape[1])``.
            Rows correspond to perturbation steps and columns correspond to
            reactions.
            ``v_sol`` stores values for steps ``1`` through ``n_steps``.
            An empty array indicates an invalid model at the initial state.
        max_reals : 1-D numpy.array
            Maximum real parts of Jacobian eigenvalues for each perturbation
            step, with shape ``(n_steps,)``.
            ``max_reals`` stores values for steps ``0`` through ``n_steps - 1``.
            An empty array indicates an invalid model at the initial state.
        '''

        x = x_ini
        e = e_ini
        if flux_sensitivity:
            v = v_ini
        
        x_sol = []
        v_sol = []
        max_reals = []

        dvdx = compute_dvdx(x, e)
        dvde = compute_dvde(x, e)
        jac = Sr@dvdx@L
        max_real = eigvals(jac).real.max()
        
        if max_real >= tol or (x <= 0.).any():
            return np.array([]), np.array([]), np.array([])

        for step in steps:
            dvdx = compute_dvdx(x, e)
            dvde = compute_dvde(x, e)

            jac = Sr@dvdx@L
            max_real = eigvals(jac).real.max()
            
            if check_jacobian:
                if max_real >= tol:
                    break
                jac_inv = inv(jac)
            else:
                try:
                    jac_inv = pinv(jac, check_finite=True)
                except:
                    break
            
            e = e + step

            x = x - L@jac_inv@Sr@dvde@step
            if check_metabolite:
                if (x <= 0.).any():
                    break
            x_sol.append(x)

            if flux_sensitivity:
                v = v + (-dvdx@L@jac_inv@Sr@dvde + dvde)@step
                v_sol.append(v)
            
            max_reals.append(max_real)
        
        return np.array(x_sol), np.array(v_sol), np.array(max_reals)


    def _solve_sensitivity(
            self, 
            ss_x_ini,
            e_ini, 
            kparams, 
            vins, 
            vouts,  
            rel_lbs, 
            rel_rbs, 
            n_steps=100,
            log_spacing=False,
            flux_sensitivity=True,
            check_jacobian=True,
            check_metabolite=True
        ):
        '''
        Solve sensitivity of steady state metabolite concentrations and fluxes with 
        respect to enzyme concentrations based on the continuation method.

        Parameters
        ----------
        ss_x_ini : numpy.array
            Initial steady-state metabolite concentrations ordered according
            to ``self.stoy_mat.index``.
        e_ini : numpy.array
            Initial enzyme concentrations ordered according to
            ``self.enz_vars``.
        kparams : numpy.array
            Kinetic parameters ordered according to ``self.kparams``.
        vins : numpy.array
            Influx rates for initial substrates ordered according to
            ``self.ini_subs``.
        vouts : numpy.array
            Efflux rates for final products ordered according to
            ``self.fin_pros``.
        rel_lbs, rel_rbs : numpy.array
            Relative left and right perturbation bounds for enzyme
            concentrations ordered according to ``self.enz_vars``. 
            If ``lb == rb``, the corresponding enzyme is not perturbed.
        n_steps : int
            Number of integration step.
        log_spacing : bool, optional
            If ``True``, perturbation points are logarithmically spaced.
            Otherwise, they are evenly spaced.
        flux_sensitivity : bool, optional
            If ``True``, derivatives of steady-state fluxes with respect to
            enzyme concentrations are also computed.
        check_jacobian : bool, optional
            If ``True``, verify that all Jacobian eigenvalues have negative
            real parts.
        check_metabolite : bool, optional
            If ``True``, enforce positive metabolite concentrations during
            perturbation.
        '''
        
        compute_v = partial(
            self._compute_v,
            kparams=kparams, vins=vins, vouts=vouts, v_fun=self.v_fun
        )
        compute_dvdx = partial(
            self._compute_dvdx,
            kparams=kparams, vins=vins, vouts=vouts, dvdx_fun=self.dvdx_fun
        )
        compute_dvde = partial(
            self._compute_dvde,
            kparams=kparams, vins=vins, vouts=vouts, dvde_fun=self.dvde_fun
        )
        
        ss_v_ini = compute_v(ss_x_ini, e_ini)

        if log_spacing:
            e_r = e_ini*np.logspace(np.log10(1), np.log10(rel_rbs), n_steps+1)
        else:
            e_r = e_ini*np.linspace(1, rel_rbs, n_steps+1)
        steps_r = np.diff(e_r, axis=0)
        x_sol_r, v_sol_r, max_reals_r = self._euler(
            ss_x_ini, ss_v_ini, e_ini, steps_r, 
            self.red_aug_stoy_mat, self.link_mat, 
            compute_dvdx, compute_dvde, compute_v,
            flux_sensitivity=flux_sensitivity, 
            check_jacobian=check_jacobian, 
            check_metabolite=check_metabolite
        )
        if x_sol_r.size == 0:
            return None
        else:
            x_sol_r = pd.DataFrame(x_sol_r, columns=self.var_names['ini_xs'])
            v_sol_r = pd.DataFrame(v_sol_r, columns=self.aug_stoy_mat.columns)
        
        if log_spacing:
            e_l = e_ini*np.logspace(np.log10(1), np.log10(rel_lbs), n_steps+1)
        else:
            e_l = e_ini*np.linspace(1, rel_lbs, n_steps+1)
        steps_l = np.diff(e_l, axis=0)
        x_sol_l, v_sol_l, max_reals_l = self._euler(
            ss_x_ini, ss_v_ini, e_ini, steps_l, 
            self.red_aug_stoy_mat, self.link_mat, 
            compute_dvdx, compute_dvde, compute_v,
            flux_sensitivity=flux_sensitivity, 
            check_jacobian=check_jacobian, 
            check_metabolite=check_metabolite
        )
        if x_sol_l.size == 0:
            return None
        else:
            x_sol_l = pd.DataFrame(x_sol_l, columns=self.var_names['ini_xs'])
            v_sol_l = pd.DataFrame(v_sol_l, columns=self.aug_stoy_mat.columns)

        sol = {
            'x_r': x_sol_r, 
            'x_l': x_sol_l,
            'v_ini': pd.Series(ss_v_ini, index=self.aug_stoy_mat.columns),
            'v_r': v_sol_r,
            'v_l': v_sol_l, 
            'steps_r': steps_r, 
            'steps_l': steps_l,
            'maxreals_r': max_reals_r,
            'maxreals_l': max_reals_l
        }
        
        return sol


    def _simulation_worker(
            self, 
            ini_x_set,
            e_conc_set,
            kparam_set, 
            vin_set,
            vout_set,
            rel_lbs,
            rel_rbs,
            n_steps=100,
            log_spacing=False,
            flux_sensitivity=True,
            check_jacobian=True,
            check_metabolite=True
        ):
        '''
        Simulation worker for sensitivity analysis in parallel.

        Parameters
        ----------
        ini_x_set : pandas.DataFrame
            Sampled sets of initial metabolite concentrations. 
        e_conc_set : pandas.DataFrame
            Sampled sets of enzyme concentrations.
        kparam_set : pandas.DataFrame
            Sampled sets of kinetic parameters.
        vin_set : pandas.DataFrame
            Sampled sets of influx for initial substrates.
        vout_set : pandas.DataFrame
            Sampled sets of efflux for final products.
        rel_lbs : numpy.array
            Left fold-change perturbation bounds for enzyme concentrations.
        rel_rbs : numpy.array
            Right fold-change perturbation bounds for enzyme concentrations.
        n_steps : int
            Number of integration steps in perturbations toward the left and
            right directions, respectively.
        log_spacing : bool, optional
            If ``True``, perturbation points are logarithmically spaced.
            Otherwise, they are evenly spaced. 
        flux_sensitivity : bool, optional
            If ``True``, derivatives of steady-state fluxes with respect to
            enzyme concentrations are also computed.
        check_jacobian : bool, optional
            If ``True``, verify that all Jacobian eigenvalues have negative
            real parts.
        check_metabolite : bool, optional
            If ``True``, enforce positive metabolite concentrations during
            perturbation.
        '''

        if platform.system() == 'Linux':
            import os
            os.sched_setaffinity(os.getpid(), range(os.cpu_count()))
        
        solutions = []
        for idx, ini_xs in ini_x_set.iterrows():
            es = e_conc_set.loc[idx,:]
            vins = vin_set.loc[idx,:]
            vouts = vout_set.loc[idx,:]
            kparams = kparam_set.loc[idx,:]
            
            ss_xs_ini = ini_xs
            
            sol = self._solve_sensitivity(
                ss_xs_ini.values,
                es.values, 
                kparams.values, 
                vins.values,
                vouts.values, 
                rel_lbs, 
                rel_rbs, 
                n_steps,
                log_spacing,
                flux_sensitivity,
                check_jacobian,
                check_metabolite
            )
            if sol is None:
                continue
            
            model_info = {
                'model_idx': idx,
                'x_ini': ss_xs_ini, 'e_ini': es, 
                'kparams': kparams, 'vins': vins, 'vouts': vouts, 
                'rel_lbs': pd.Series(rel_lbs, index=es.index), 
                'rel_rbs': pd.Series(rel_rbs, index=es.index), 
            }
            sol.update(model_info)
            solutions.append(sol)
        
        return solutions
    

    def simulate(
            self,
            perturb_enzymes, 
            fold_change, 
            n_steps,
            log_spacing,
            n_models,
            n_jobs,
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
            Enzymes perturbed during robustness analysis.
        fold_change : 2-tuple or dict
            Fold-change perturbation bounds for enzyme concentrations.
        n_steps : int
            Number of integration steps in perturbations toward the left and
            right directions, respectively.
        log_spacing : bool, optional
            If ``True``, perturbation points are logarithmically spaced.
            Otherwise, they are evenly spaced.
        n_models: int
            Number of sampled models used in the simulation. Increasing this
            value may improve consensus robustness estimates.
            Note that the maximum number of models is limited by the number of loaded
            parameter sets.
        n_jobs : int
            Number of parallel jobs.
        flux_sensitivity : bool, optional
            If ``True``, derivatives of steady-state fluxes with respect to
            enzyme concentrations are also computed.
        check_jacobian : bool, optional
            If ``True``, verify that all Jacobian eigenvalues have negative
            real parts.
        check_metabolite : bool, optional
            If ``True``, enforce positive metabolite concentrations during
            perturbation.
        '''

        pert_enzs, rel_lbs, rel_rbs = self._get_perturbation_bounds(
            perturb_enzymes, fold_change
        )
        
        (ini_x_sets, 
         e_sets, 
         kparam_sets, 
         vin_sets, 
         vout_sets) = self._select_parameter_sets(n_models)
        
        ini_x_sets = self._check_metabolite_set_inclusion(ini_x_sets)
        e_sets = self._check_enzyme_set_inclusion(e_sets)
        kparam_sets = self._check_kinetic_parameter_set_inclusion(kparam_sets)
        vin_sets = self._check_influx_set_inclusion(vin_sets)
        vout_sets = self._check_efflux_set_inclusion(vout_sets)
        
        all_chunks = self._get_split_chunks(
            *[ini_x_sets, e_sets, kparam_sets, vin_sets, vout_sets], 
            n_chunks=n_jobs
        )

        with Pool(n_jobs) as pool:
            async_res = []
            for chunks_indices in zip(*all_chunks):
                res = pool.apply_async(
                    func=self._simulation_worker,
                    args=(
                        *chunks_indices,
                        rel_lbs,
                        rel_rbs,
                        n_steps,
                        log_spacing,
                        flux_sensitivity,
                        check_jacobian,
                        check_metabolite
                    )
                )
                async_res.append(res)

            full_sim_res = []
            for res in async_res:
                sim_res = res.get()
                full_sim_res.extend(sim_res)

        return pert_enzs, full_sim_res, flux_sensitivity, self.stoy_mat, n_steps