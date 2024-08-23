import os

import matplotlib.pyplot as plt

from ricker.settings import *
from ricker.models import *
from ricker.evaluation_module import EvaluationModule

config = load_config("configs/ricker.yaml")
observation_params = config['chaotic']['observation_params']
observation_noise = config['chaotic']['observation_noise']['stochastic']
initial_conditions = config['chaotic']['initial_conditions']

initial_conditions_distribution = {
    'species1': lambda: np.random.normal(initial_conditions['species1'], 0.1),
    'species2': lambda: np.random.normal(initial_conditions['species2'], 0.1)
}

observation_model = RickerPredation(initial_conditions_distribution, observation_params)
observed_data = observation_model.create_observations(years=1, phase_shift= 0)
observation_model.plot_time_series(observed_data, 'y_test')

ensemble_size = 50
ensemble = observation_model.create_ensemble(ensemble_size, years=1, phase_shift=0)

plt.figure(figsize=(10, 5))
plt.plot(ensemble[:,0,:].detach().numpy().squeeze().transpose())
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()

Evaluation = EvaluationModule(ensemble)
bootrap_container = Evaluation.bootstrap_ensemble(bootstrap_samples=50, ensemble_samples=40)

species1_X = bootrap_container[..., 0, :]
species2_X = bootrap_container[..., 1, :]

species1_PPP = [
    Evaluation.compute_PPP(species1_X, t, torch.std(ensemble[:,0,:])) for t in range(species1_X.shape[2])
                ]
species1_PPP = [tensor.item() for tensor in species1_PPP]

species2_PPP = [
    Evaluation.compute_PPP(species2_X, t, torch.std(ensemble[:,1,:])) for t in range(species2_X.shape[2])
                ]
species2_PPP = [tensor.item() for tensor in species2_PPP]

decorrelation_coefficient = 0.3 # random choice assumed as temporal autocorrelation
df_withingroups = 39 # sample size minus degrees used for computation of mean
df_climatology = 365/((1-decorrelation_coefficient)/(1+decorrelation_coefficient)) - 1

PPP_threshold = Evaluation.PPP_threshold(df1=df_withingroups,
                                         df2=df_climatology)

plt.figure(figsize=(7, 5))
plt.plot(species1_PPP, label = "Species 1")
plt.plot(species2_PPP, label = "Species 2")
plt.hlines(PPP_threshold, xmin=0, xmax=365, color = "black", linestyles='--')
plt.xlabel("Time")
plt.ylabel("Potential Prognostic Predictability")
plt.legend()
plt.show()