import torch
import pandas as pd
import os

from visualisations import plot_losses, plot_posterior
from torch.utils.data import DataLoader
from data import SimODEData
from neuralforecast.losses.pytorch import sCRPS, QuantileLoss
class Trainer:
    """
    Trainer class to fit the Ricker_Ensemble model to data generated with the RickerPredation class.
    """

    def __init__(self, y_train, sigma_train, x_train, dir):
        """
        Initialize the Trainer with data.

        Parameters:
        - y_train: Training target data.
        - sigma_train: Training sigma values.
        - x_train: Training input (forcing) data.
        - dir: Directory to save or load the fit results.
        """
        self.y_train = y_train
        self.sigma_train = sigma_train
        self.x_train = x_train
        self.dir = dir
        self.fitted_values_file = os.path.join(dir, 'fitted_values.csv')

    def train(self, model, epochs, loss_fun='mse', step_length=2, fit_sigma=None):
        """
        Train the model using gradient descent.

        Parameters:
        - model: The model to be trained.
        - epochs: Number of training epochs.
        - loss_fun: Loss function to use.
        - step_length: Length of the time step.
        - fit_sigma: Whether to fit sigma.

        Returns:
        - losses: List of loss values over epochs.
        """
        data = SimODEData(step_length=step_length, y=self.y_train, y_sigma=self.sigma_train, forcing=self.x_train)
        trainloader = DataLoader(data, batch_size=1, shuffle=False, drop_last=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        # Loss function selection
        loss_functions = {
            'mse': torch.nn.MSELoss(),
            'crps': sCRPS(),
            'quantile': QuantileLoss(0.5),
            'gaussian': torch.nn.GaussianNLLLoss()
        }
        criterion = loss_functions.get(loss_fun, torch.nn.MSELoss())

        losses = []
        for epoch in range(epochs):
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}')

            for target, target_variability, forcing in trainloader:
                target = target.squeeze()
                target_upper = target + 2 * torch.std(target).item()
                target_lower = target - 2 * torch.std(target).item()
                initial_state = target.clone()[0]

                optimizer.zero_grad()
                output, output_sigma = model(initial_state, forcing)

                if loss_fun == 'mse' and fit_sigma is not None:
                    loss = (criterion(output, target) +
                            criterion(output_sigma[0], target_upper) +
                            criterion(output_sigma[1], target_lower))
                elif loss_fun == 'mse':
                    loss = criterion(output, target)
                elif loss_fun == 'quantile':
                    loss = torch.stack([criterion(target[i], output[:, i].squeeze()) for i in range(step_length)])
                elif loss_fun == 'gaussian':
                    loss = criterion(output, target, output_sigma)

                loss = torch.sum(loss) / step_length
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        return losses

    def fit_model(self, model, fit_model=True, **kwargs):
        """
        Fit the model to the data and save the results.

        Parameters:
        - model: The model to be trained.
        - fit_model: Whether to fit the model or load previous fit.
        - kwargs: Additional arguments for training.

        Returns:
        - fitted_values: DataFrame of fitted values.
        """
        if fit_model or not os.path.exists(self.fitted_values_file):
            fitted_values = []

            for i in range(kwargs.get('samples', 10)):
                losses = self.train(model,
                                    epochs=kwargs.get('epochs', 100),
                                    loss_fun=kwargs.get('loss_fun', 'mse'),
                                    step_length=kwargs.get('step_length', 2),
                                    fit_sigma=kwargs.get('fit_sigma', False))

                fitted_values.append(model.get_fit())
                print(f"Sample {i+1}: {model.get_fit()}")

            fitted_values_df = pd.DataFrame(fitted_values)
            fitted_values_df.to_csv(self.fitted_values_file, index=False)

            plot_posterior(fitted_values_df, saveto=self.dir)
            plot_losses(losses, loss_fun=kwargs.get('loss_fun', 'mse'), saveto=self.dir)
        else:
            print("Loading parameters from previous fit.")
            fitted_values_df = pd.read_csv(self.fitted_values_file)

        return fitted_values_df


# Create a Trainer instance
#trainer = Trainer(y_train=y_train, sigma_train=sigma_train, x_train=x_train, dir=dir)

# Fit the model using the Trainer
#fitted_values = trainer.fit_model(model, fit_model=True, samples=10, epochs=100, loss_fun='mse', step_length=2)

