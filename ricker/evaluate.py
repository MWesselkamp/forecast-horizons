import torch
import numpy as np

from ricker.visualisations import *
from ricker.settings import *
from ricker.models import *
from ricker.evaluation_module import EvaluationModule
from utils import compute_autocorrelation

config = load_config("configs/ricker.yaml")
observation_params = config['chaotic']['observation_params']
observation_noise = config['chaotic']['observation_noise']['stochastic']
initial_conditions = config['chaotic']['initial_conditions']

initial_conditions_distribution = {
    'species1': lambda: np.random.normal(initial_conditions['species1'], 0.1),
    'species2': lambda: np.random.normal(initial_conditions['species2'], 0.1)
}

observation_model = RickerPredation(initial_conditions_distribution, observation_params)

climatology = observation_model.create_observations(years=10, phase_shift= 0).get('climatology')
climatology = climatology.view((2, 9, 365))

observed_data = observation_model.create_observations(years=1, phase_shift= 0)
observation_model.plot_time_series(observed_data, 'y_test')

ensemble_size = 50
ensemble = observation_model.create_ensemble(ensemble_size, years=1, phase_shift=0)

plot_ensemble(ensemble)

Evaluation = EvaluationModule(ensemble)
bootrap_container = Evaluation.bootstrap_ensemble(bootstrap_samples=50, ensemble_samples=40)

species1_X = bootrap_container[..., 0, :]
species2_X = bootrap_container[..., 1, :]

species1_PPP = [
    Evaluation.compute_PPP(species1_X, t, torch.std(climatology[0, :, :])) for t in range(species1_X.shape[2])
                ]
species1_PPP = [tensor.item() for tensor in species1_PPP]

species2_PPP = [
    Evaluation.compute_PPP(species2_X, t, torch.std(climatology[1, :, :])) for t in range(species2_X.shape[2])
                ]
species2_PPP = [tensor.item() for tensor in species2_PPP]

autocorrelation_species1 = np.mean([compute_autocorrelation(climatology[0,i,:].detach().numpy()) for i in range(climatology.shape[1])])
print(f"Autocorrelation Species1: {autocorrelation_species1:.4f}")
autocorrelation_species2 = np.mean([compute_autocorrelation(climatology[1,i,:].detach().numpy()) for i in range(climatology.shape[1])])
print(f"Autocorrelation Species1: {autocorrelation_species1:.4f}")

decorrelation_coefficient = 0.2 # 1-lag temporal autocorrelation
df_withingroups = 39 # sample size minus degrees used for computation of mean
df_climatology = 365/((1-decorrelation_coefficient)/(1+decorrelation_coefficient)) - 1

PPP_threshold = Evaluation.PPP_threshold(df1=df_withingroups,
                                         df2=df_climatology)

plot_ppp(species1_PPP, species2_PPP, PPP_threshold)

plot_combined(species1_PPP, species2_PPP, PPP_threshold, ensemble)

x_test_rep = np.concat((observed_data.get('x_test'), observed_data.get('x_test')))

iterated_dynamics_species1 = []
iterated_dynamics_species2 = []

for i in range(120):

    ensemble = observation_model.iterate_ensemble(forcing=x_test_rep[i:(i + 365)], ensemble_size=50)

    Evaluation = EvaluationModule(ensemble)
    bootrap_container = Evaluation.bootstrap_ensemble(bootstrap_samples=50, ensemble_samples=40)

    species1_X = bootrap_container[..., 0, :]
    species2_X = bootrap_container[..., 1, :]

    species1_PPP = [
        Evaluation.compute_PPP(species1_X, t, torch.std(climatology[0, :, :])) for t in range(species1_X.shape[2])
    ]
    species1_PPP = torch.tensor([tensor.item() for tensor in species1_PPP])
    iterated_dynamics_species1.append(species1_PPP)

    species2_PPP = [
        Evaluation.compute_PPP(species2_X, t, torch.std(climatology[1, :, :])) for t in range(species2_X.shape[2])
    ]
    species2_PPP = torch.tensor([tensor.item() for tensor in species2_PPP])
    iterated_dynamics_species2.append(species2_PPP)

iterated_dynamics_species1 = torch.stack(iterated_dynamics_species1).detach().numpy()
iterated_dynamics_species2 = torch.stack(iterated_dynamics_species2).detach().numpy()

plot_horizon_maps(iterated_dynamics_species1, iterated_dynamics_species2)
