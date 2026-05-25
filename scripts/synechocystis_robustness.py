'''This script demonstrates the use of RobustNet for metabolic robustness analysis with 
a Synechocystis model. The workflow consists of three main steps:
1. Estimate the reference-state flux distribution.
2. Sample model paramters using prior innformation from multi-omics data and enzyme 
   databases.
3. Evaluate metabolic robustness and predict metabolic responses to enzyme 
   perturbations.
'''


import os
import pandas as pd
from pathlib import Path
from robustnet import Model


BASE_DIR = Path(__file__).parent.parent
OUT_DIR = f'{BASE_DIR}/results/synechocystis'
MODEL_FILE = f'{BASE_DIR}/models/synechocystis/synechocystis_model.xlsx'
FLUXOMICS = f'{BASE_DIR}/models/synechocystis/measured_fluxes.xlsx'
FLUX_BOUNDS = f'{BASE_DIR}/models/synechocystis/flux_bounds.xlsx'
METABOLOMICS = f'{BASE_DIR}/models/synechocystis/measured_metabolites.xlsx'
PROTEOMICS = f'{BASE_DIR}/models/synechocystis/measured_enzymes.xlsx'
KINETIC_PARAMETERS = f'{BASE_DIR}/models/synechocystis/measured_kinetic_parameters.xlsx'

PERTURB_ENZYMES = [
    'RuBisCO', 'PGK', 'GAPD', 'TPI', 'ALD', 'TKT', 'TAL', 'FBPase', 'PGI', 'G6PD',
    'PGL', 'GND', 'FBA', 'SBPase', 'RPI', 'RPE', 'PRK', 'PGM', 'ENO', 'PYK', 'PDH', 
    'XFPK', 'PTA', 'CS', 'ACON', 'ICD', 'OGDC', 'SSADH', 'SDH', 'FUMS', 'MDH', 'ME', 
    'PPC', 'ATPSyn', 'NADPase'
]


def fit_reference_fluxes(model, out_dir):
    flux_data = pd.read_excel(FLUXOMICS, header=0, index_col=0)
    flux_bounds = pd.read_excel(FLUX_BOUNDS, header=0, index_col=0)
    
    model.load_priors(
        'fluxomics',
        data=flux_data.iloc[:,0],
        std=flux_data.iloc[:,1],
    )
    
    fit_res = model.estimate_reference_fluxes(
        bounds={rxn: tuple(row) for rxn, row in flux_bounds.iterrows()},
        exclude_metabolites=['CO2', 'NAD', 'NADH'],
        optimizer='scipy',
        method='SLSQP',   # faster, while less stable
        tol=1e-8
    )
    
    # output estimation results
    fit_fluxes = pd.DataFrame({
        'Mean': fit_res.estimated_fluxes, 
        'STD': fit_res.estimated_flux_errors 
               if fit_res.estimated_flux_errors is not None else 0.01
    })
    fit_fluxes.to_csv(
        f'{out_dir}/fitted_ref_fluxes.tsv', header=True, index=True, sep='\t'
    )

    # visualize the comparison between fitted and measured fluxes
    fit_res.plot_simulated_vs_measured_fluxes(out_dir)


def sample_parameters(model, out_dir, fit_res_dir):
    ref_flux_data = pd.read_csv(
        f'{fit_res_dir}/fitted_ref_fluxes.tsv', header=0, index_col=0, sep='\t'
    )
    metab_data = pd.read_excel(METABOLOMICS, header=0, index_col=0)
    enz_data = pd.read_excel(PROTEOMICS, header=0, index_col=0)
    kparam_data = pd.read_excel(KINETIC_PARAMETERS, header=0, index_col=0)

    model.load_priors(
        'reference_fluxes',
        data=ref_flux_data.iloc[:,0],
        std=ref_flux_data.iloc[:,1]
    )
    model.load_priors(
        'metabolomics', 
        data=metab_data.iloc[:,0], 
        std=metab_data.iloc[:,1],
    )
    model.load_priors(
        'proteomics', 
        data=enz_data.iloc[:,0], 
        std=enz_data.iloc[:,1],
    )
    model.load_priors(
        'kparameters',
        data=kparam_data.iloc[:,0],
        std=kparam_data.iloc[:,1]
    )

    samp_res = model.generate_parameter_sets(
        alpha=None,
        n_tunes=5000,
        n_samples=2000,
        n_chains=10,
        n_jobs=10
    )

    # output sampling trace
    samp_res.trace.to_netcdf(f'{out_dir}/trace.nc')
    
    # output sampled model parameters
    samp_res.sampled_kinetic_parameters.to_csv(
        f'{out_dir}/sampled_kparams.tsv', header=True, index=True, sep='\t'
    )
    samp_res.sampled_metabolite_concentrations.to_csv(
        f'{out_dir}/sampled_mconcs.tsv', header=True, index=True, sep='\t'
    )
    samp_res.sampled_enzyme_concentrations.to_csv(
        f'{out_dir}/sampled_econcs.tsv', header=True, index=True, sep='\t'
    )
    
    # visualize the comparison between sampled paramters and their priors
    samp_res.plot_sampled_vs_prior_kinetic_parameters(
        f'{out_dir}/sampled_kparams_plots', parameters='all'
    )
    samp_res.plot_sampled_vs_prior_metabolites(
        f'{out_dir}/sampled_mconcs_plots', metabolites='all'
    )
    samp_res.plot_sampled_vs_prior_enzymes(
        f'{out_dir}/sampled_econcs_plots', enzymes='all'
    )


def evaluate_robustness(model, out_dir, samp_res_dir):
    model.load_parameter_sets(
        mconc_set=f'{samp_res_dir}/sampled_mconcs.tsv',
        econc_set=f'{samp_res_dir}/sampled_econcs.tsv',
        kparam_set=pd.concat(
            (pd.read_csv(f'{samp_res_dir}/sampled_kparams.tsv', 
                         header=0, index_col=0, sep='\t'), 
             pd.read_csv(f'{samp_res_dir}/sampled_mconcs.tsv', 
                         header=0, index_col=0, sep='\t')), 
            axis=1
        ),
    )
    # Several metabolites, such as end metabolites, are treated as kinetic 
    # parameters. Therefore, sampled metabolites are combined with sampled kinetic 
    # parameters for kparam_set
    
    for enz in PERTURB_ENZYMES:
        rob_res = model.evaluate_robustness(
            perturb_enzymes=[enz],
            fold_change=(0.1, 10),
            exclude_metabolites=['CO2', 'NAD', 'NADH'],
            n_steps=300,
            n_models=1000,
            n_jobs=100,
            flux_sensitivity=True,
        )
        
        subout_dir = f'{out_dir}/{enz}'
        os.makedirs(subout_dir, exist_ok=True)
        # output robustness metrics and metabolic responses of individually 
        # perturbed enzymes
        print(f'Robustness index: {rob_res.robust_index}')
        rob_res.robust_model_probability(subout_dir)
        rob_res.metabolite_sensitivity(
            f'{subout_dir}/metab_sen', kind='stats', metabolites='all'
        )
        rob_res.metabolite_distribution(
            f'{subout_dir}/metab_dist', metabolites='all'
        )
        rob_res.flux_sensitivity(
            f'{subout_dir}/flux_sen', kind='stats', reactions='all'
        )
        rob_res.flux_distribution(
            f'{subout_dir}/flux_dist', reactions='all'
        )
        rob_res.eigreal_sensitivity(subout_dir)
        rob_res.eigreal_distribution(subout_dir)


def main():
    model = Model('syn')
    model.read_from_file(MODEL_FILE)
    
    # estimate reference fluxes
    fit_savepath = f'{OUT_DIR}/fit_res'
    os.makedirs(fit_savepath, exist_ok=True)
    fit_reference_fluxes(model, fit_savepath)

    # generate parameter sets
    samp_savepath = f'{OUT_DIR}/samp_res'
    os.makedirs(samp_savepath, exist_ok=True)
    sample_parameters(model, samp_savepath, fit_savepath)

    # evaluate robustness
    rob_savepath = f'{OUT_DIR}/rob_res'
    os.makedirs(rob_savepath, exist_ok=True)
    evaluate_robustness(model, rob_savepath, samp_savepath)

    


if __name__ == '__main__':
    main()
