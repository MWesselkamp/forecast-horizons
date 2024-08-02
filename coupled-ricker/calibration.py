import torch
import numpy as np
import pandas as pd
import os
import os.path


from visualisations import plot_losses, plot_posterior
from torch.utils.data import DataLoader
from data import SimODEData
from models import Ricker_Predation, Ricker_Ensemble
from loss_functions import crps_loss
from neuralforecast.losses.pytorch import sCRPS, MQLoss, MAE, QuantileLoss

#===========================================#
# Fit the Ricker model with gradien descent #
#===========================================#

def train(y_train,sigma_train, x_train, model, epochs, loss_fun = 'mse', step_length = 2, fit_sigma = None):


    data = SimODEData(step_length=step_length, y=y_train, y_sigma=sigma_train, temp=x_train)
    trainloader = DataLoader(data, batch_size=1, shuffle=False, drop_last=True)

    optimizer = torch.optim.Adam([{'params':model.model_params}], lr=0.0001)

    criterion = torch.nn.MSELoss()
    criterion2 = sCRPS()
    criterion3 = MQLoss(quantiles = [0.4, 0.6])
    criterion4 = QuantileLoss(0.5) # Pinball Loss
    criterion5 = torch.nn.GaussianNLLLoss()

    losses = []
    for epoch in range(epochs):

        if epoch % 10 == 0:
            print('Epoch:', epoch)

        for batch in trainloader:

            target, var, temp = batch
            target = target.squeeze()

            target_upper = target + 2*torch.std(target).item()
            target_lower = target - 2*torch.std(target).item()

            initial_state = target.clone()[0]

            optimizer.zero_grad()

            output, output_sigma = model(initial_state, temp)

            if (loss_fun == 'mse') & (fit_sigma is not None):
                loss = criterion(output, target) + criterion(output_sigma[0], target_upper) + criterion(output_sigma[1], target_lower)
            elif (loss_fun == 'mse') & (fit_sigma is None):
                loss = criterion(output, target)
            elif loss_fun == 'crps':
                # loss = torch.zeros((1), requires_grad=True).clone()
                loss = torch.stack([crps_loss(output[:,i].squeeze(), target[i]) for i in range(step_length)])
            elif loss_fun == 'quantile':
                loss = torch.stack([criterion4(target[i], output[:,i].squeeze()) for i in range(step_length)])
            elif loss_fun == 'mquantile':
                pass
            elif loss_fun == 'gaussian':
                loss = criterion5(output, target, output_sigma)

            loss = torch.sum(loss) / step_length

            loss.backward()
            losses.append(loss.clone())
            optimizer.step()

    return losses


def model_fit(fit_model, dir, y_train, x_train, sigma_train, initial_params, initial_noise, **kwargs):

    if ('fitted_values' not in globals()) & (fit_model):

        fitted_values = []
        for i in range(kwargs['samples']):
            # Sample from prior
            ip = [np.random.normal(i, 0.1, 1)[0] for i in initial_params]
            model = Ricker_Ensemble(params=ip, noise=initial_noise, initial_uncertainty=None)

            losses = train(y_train, sigma_train, x_train, model,
                           epochs=kwargs['epochs'], loss_fun=kwargs['loss_fun'], step_length=kwargs['step_length'],
                           fit_sigma=initial_noise)

            fitted_values.append(model.get_fit())
            print(model.get_fit())

        fitted_values = pd.DataFrame(fitted_values)
        fitted_values.to_csv(os.path.join(dir, 'fitted_values.csv'))

        plot_posterior(fitted_values, saveto=dir)
        plot_losses(losses, loss_fun='mse', saveto=dir)

    else:
        print(f"Loading parameters from previous fit.")
        fitted_values = pd.read_csv(os.path.join(dir, 'fitted_values.csv'), index_col=False)
        fitted_values = fitted_values.drop(fitted_values.columns[0], axis=1)

    return fitted_values


