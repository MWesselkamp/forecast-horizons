import matplotlib.pyplot as plt
import torch
import numpy as np

from ricker.visualisations import *
from ricker.settings import *
from ricker.models import *
from ricker.evaluation_module import EvaluationModule

config = load_config("configs/ricker.yaml")
observation_params = config['chaotic']['observation_params']
observation_noise = config['chaotic']['observation_noise']['stochastic']
initial_conditions = config['chaotic']['initial_conditions']

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
#climatological_variability_seasonal_species1 = torch.tensor([torch.std(climatology[0, :, t]) for t in range(climatology.shape[2])])
#climatological_variability_seasonal_species2 = torch.tensor([torch.std(climatology[1, :, t]) for t in range(climatology.shape[2])])
#plt.plot(climatological_variability_seasonal_species1, color = "blue")
#plt.plot(climatological_variability_seasonal_species2, color = "green")
#plt.show()

plt.plot(climatology[0,:,:].detach().numpy().transpose(), color = "blue")
plt.plot(climatology[1,:,:].detach().numpy().transpose(), color = "green")
plt.show()

lyapunovs = [observation_model.compute_lyapunov_exponent(t) for t in range(1, climatological_data['y_train'].shape[1])]
plt.plot(lyapunovs)
plt.show()

observed_data = observation_model.create_observations(years=1)
observation_model.plot_time_series(observed_data, 'y_test', climatology  = climatology.detach().numpy().reshape(-1, 200).transpose())

ensemble_size = config['simulation']['ensemble_size']
ensemble = observation_model.create_ensemble(ensemble_size,
                                             forcing=observed_data.get('x_test'),
                                             years=1)
plot_ensemble(ensemble)

Evaluation = EvaluationModule(ensemble, climatology, config)
species1_PPP = [
    Evaluation.compute_PPP(ensemble[:,0,:], t, climatological_variability_species1) for t in range(ensemble.shape[2])
                ]
species1_PPP = [tensor.item() for tensor in species1_PPP]

bootrap_container = Evaluation.bootstrap_ensemble()

species1_X = bootrap_container[..., 0, :]
species2_X = bootrap_container[..., 1, :]

species1_PPP = [
    Evaluation.compute_bootstrap_PPP(species1_X, t, climatological_variability_species1) for t in range(species1_X.shape[2])
                ]
species1_PPP = [tensor.item() for tensor in species1_PPP]

species2_PPP = [
    Evaluation.compute_bootstrap_PPP(species2_X, t, climatological_variability_species2) for t in range(species2_X.shape[2])
                ]
species2_PPP = [tensor.item() for tensor in species2_PPP]


dof_withingroups = config['simulation']['resampling_size'] - 1 # sample size minus degrees used for computation of mean
dofs_climatology = Evaluation.degrees_of_freedom()

PPP_threshold_species1 = Evaluation.PPP_threshold(df1=dof_withingroups,
                                         df2=dofs_climatology[0])
PPP_threshold_species2 = Evaluation.PPP_threshold(df1=dof_withingroups,
                                         df2=dofs_climatology[1])

plot_ppp(species1_PPP, species2_PPP, PPP_threshold_species1, PPP_threshold_species2)

plot_combined(species1_PPP, species2_PPP, PPP_threshold_species1, PPP_threshold_species2, ensemble)

species1_normalised_variances = np.array([
    Evaluation.bootstrap_normalised_variances(species1_X, t, climatological_variability_species1).detach().numpy() for t in range(species1_X.shape[2])])

species2_normalised_variances = np.array([
    Evaluation.bootstrap_normalised_variances(species2_X, t, climatological_variability_species2).detach().numpy() for t in range(species2_X.shape[2])])


plot_combined_2(species1_normalised_variances, species2_normalised_variances,
              species1_PPP, species2_PPP, PPP_threshold_species1, PPP_threshold_species2,
                time_horizon = 28)

plot_combined_3(species1_normalised_variances, species2_normalised_variances,
                species1_X[0,...].detach().numpy().transpose(), species2_X[0,...].detach().numpy().transpose(),
              PPP_threshold_species1, PPP_threshold_species2,
                time_horizon = 28)

# =========================================== #
# horizons from different initial conditions  #
# =========================================== #

x_test_rep = np.concat((observed_data.get('x_test'), observed_data.get('x_test')))

iterated_dynamics_species1 = []
iterated_dynamics_species2 = []
lyapunovs  = []

for i in range(0,80):

    ensemble = observation_model.create_ensemble(ensemble_size=50,
                                                 forcing=x_test_rep[i:(i + config['forcing']['resolution'])],
                                                 years=1)

    lyapunovs.append(observation_model.compute_lyapunov_exponent())

    Evaluation = EvaluationModule(ensemble)
    bootrap_container = Evaluation.bootstrap_ensemble(bootstrap_samples=50,
                                                      ensemble_samples=40)

    species1_X = bootrap_container[..., 0, :]
    species2_X = bootrap_container[..., 1, :]

    species1_PPP = [
        Evaluation.compute_bootstrap_PPP(species1_X, t, climatological_variability_species1) for t in range(species1_X.shape[2])
    ]
    species1_PPP = torch.tensor([tensor.item() for tensor in species1_PPP])
    iterated_dynamics_species1.append(species1_PPP)

    species2_PPP = [
        Evaluation.compute_bootstrap_PPP(species2_X, t, climatological_variability_species2) for t in range(species2_X.shape[2])
    ]
    species2_PPP = torch.tensor([tensor.item() for tensor in species2_PPP])

    iterated_dynamics_species2.append(species2_PPP)

iterated_dynamics_species1 = torch.stack(iterated_dynamics_species1).detach().numpy()
iterated_dynamics_species2 = torch.stack(iterated_dynamics_species2).detach().numpy()

print(np.array(lyapunovs).min())
print(np.array(lyapunovs).max())

horizons_species1 = (iterated_dynamics_species1 < PPP_threshold_species1).astype(int)
horizons_species2 = (iterated_dynamics_species2 < PPP_threshold_species2).astype(int)

plot_horizon_maps(iterated_dynamics_species1[:60,:60],
                  iterated_dynamics_species2[:60,:60])

plot_binary_horizon_maps(horizons_species1[:80,:80],
                  horizons_species2[:80,:80])
