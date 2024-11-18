import matplotlib.pyplot as plt
import torch
import time
import gc
import numpy as np
import SALib as sb

from ricker.visualisation_module import *
from ricker.settings import *
from ricker.models import *
from utils import *
from ricker.evaluation_module import EvaluationModule

set_seed(42)

#=======================#
# Ensemble  sensitivity #
#=======================#

config = load_config("configs/ricker.yaml")
observation_params = config['chaotic']['observation_params']
observation_noise = config['chaotic']['observation_noise']['stochastic']
initial_conditions = config['chaotic']['initial_conditions']
ensemble_size = config['simulation']['ensemble_size']

def run_experiment(initial_conditions, observation_params, observation_noise, ensemble_size):

    initial_conditions_distribution = {
        'species1': lambda: np.random.normal(initial_conditions['species1_mean'],
                                             initial_conditions['species1_std']),
        'species2': lambda: np.random.normal(initial_conditions['species2_mean'],
                                             initial_conditions['species2_std'])
    }

    observation_model = RickerPredation(initial_conditions_distribution,
                                        observation_params,
                                        forcing_params=config['forcing'])

    climatological_data = observation_model.create_observations(years=config['simulation']['climatology'])
    climatology = climatological_data.get('climatology').view((2, config['simulation']['climatology'] - 1 , config['forcing']['resolution']))
    climatological_variability_species1 = torch.std(climatology[0, :, :])
    climatological_variability_species2 = torch.std(climatology[1, :, :])

    observed_data = observation_model.create_observations(years=1)

    ensemble = observation_model.create_ensemble(ensemble_size,
                                                 forcing=observed_data.get('x_test'),
                                                 years=1)

    Evaluation = EvaluationModule(ensemble, climatology, config)
    species1_PPP = [
        Evaluation.compute_PPP(ensemble[:,0,:], t, climatological_variability_species1) for t in range(ensemble.shape[2])
                    ]
    species1_PPP = [tensor.detach().cpu().item() for tensor in species1_PPP]

    species2_PPP = [
        Evaluation.compute_PPP(ensemble[:,1,:], t, climatological_variability_species2) for t in range(ensemble.shape[2])
                    ]
    species2_PPP = [tensor.detach().cpu().item() for tensor in species2_PPP]

    dof_withingroups = ensemble_size - 1 # sample size minus degrees used for computation of mean
    dofs_climatology = Evaluation.degrees_of_freedom()

    PPP_threshold_species1 = Evaluation.PPP_threshold(df1=dof_withingroups,
                                             df2=dofs_climatology[0])
    PPP_threshold_species2 = Evaluation.PPP_threshold(df1=dof_withingroups,
                                             df2=dofs_climatology[1])

    horizon_species1 = np.argmin(~(species1_PPP < PPP_threshold_species1))
    print("Potential forecast horizon species 1: ", horizon_species1)
    horizon_species2 = np.argmin(~(species2_PPP < PPP_threshold_species2))
    print("Potential forecast horizon species 2: ", horizon_species2)

    del climatology, ensemble, observed_data
    gc.collect()

    return [horizon_species1, horizon_species2, PPP_threshold_species1, PPP_threshold_species2]

ensemble_sizes = list(range(5, 100, 1))
horizons1 = []
horizons2 = []
ppp_rho1 = []
ppp_rho2 = []
for ens_size in ensemble_sizes:

    horizon_species1, horizon_species2, PPP_threshold_species1, PPP_threshold_species2 = run_experiment(initial_conditions,
                                                                                                        observation_params,
                                                                                                        observation_noise,
                                                                                                        ens_size)
    horizons1.append(horizon_species1)
    horizons2.append(horizon_species2)
    ppp_rho1.append(PPP_threshold_species1)
    ppp_rho2.append(PPP_threshold_species2)


#=======================#
# Parameter sensitivity #
#=======================#


problem = {
    'num_vars': 2,
    'names': ['alpha1', 'alpha2'], # ,'bx1', 'cx1', 'bx2', 'cx1'
    'bounds': [[0.5, 2.0],
                [0.5, 2.0],
               ]
}

from SALib.sample import morris
param_values = sb.sample.morris.sample(problem, 64)

start_time = time.time()
horizons = []
for i in range(param_values.shape[0]):

    observation_params['alpha1'] = param_values[i,0]
    observation_params['alpha2'] = param_values[i,1]
    #initial_conditions['species1_std'] = param_values[i,2]
    #initial_conditions['species2_std'] = param_values[i,3]
    horizon_species1, horizon_species2 = run_experiment(initial_conditions, observation_params, observation_noise, ensemble_size)
    horizons.append([horizon_species1, horizon_species2])

    # Clear unused memory
    gc.collect()

horizons_array = np.array(horizons)
end_time = time.time()
print("Duration of experiment: ", (end_time - start_time)/60 , " minutes")

sizes = param_values[:, 2] * 300

plt.scatter(param_values[:,0], param_values[:,1], c=horizons_array[:,1],  cmap='viridis')
# Add a colorbar to show what the colors represent
plt.xlabel('$\\alpha_1$')
plt.ylabel('$\\alpha_2$')
plt.colorbar(label='Horizons')
plt.show()

sorted_indices = np.argsort(param_values[:, 1])
# Sort both arrays based on the indices from param_values[:,1]
sorted_horizons_array = horizons_array[sorted_indices]
sorted_param_values = param_values[sorted_indices]

plt.plot(sorted_param_values[:,1], sorted_horizons_array[:,1])
plt.show()

Y1 = horizons_array[:,0]
Y2 = horizons_array[:,1]

from SALib.analyze import sobol
SI1 = sobol.analyze(problem, Y1, print_to_console=True)
SI2 = sobol.analyze(problem, Y2, print_to_console=True)

SI1.plot()
plt.show()