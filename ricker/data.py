from torch.utils.data import Dataset
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 18
import numpy as np
import random

random.seed(42)
np.random.seed(42)

class SimODEData(Dataset):
    """
        A very simple dataset class for simulating ODEs
    """

    def __init__(self,
                 step_length,  # List of time points as tensors
                 y,  # List of state values (tensor) at each time point
                 y_sigma, # List of state error (tensor) at each time point
                 forcing,# List of forcing (tensor) at each time point
                 ):
        self.step_length = step_length
        self.y = y
        self.y_sigma = y_sigma
        self.x = forcing

    def __len__(self) -> int:
        return len(self.y) - self.step_length

    def __getitem__(self, index: int): #  -> Tuple[torch.Tensor, torch.Tensor]
        return self.y[index:index+self.step_length], self.y_sigma[index:index+self.step_length], self.x[index:index+self.step_length]

class ForecastData(Dataset):
    """
        A very simple dataset class for generating forecast data sets of different lengths.
    """
    def __init__(self, y, forcing, climatology = None, forecast_days = 'all', lead_time = None):
        self.y = y
        self.x = forcing
        self.climatology = climatology
        self.forecast_days = forecast_days
        self.lead_time = lead_time

    def __len__(self) -> int:
        if self.forecast_days == 'all':
            return len(self.y)-1
        else:
            return self.forecast_days

    def __getitem__(self, index: int): #  -> Tuple[torch.Tensor, torch.Tensor]
        if self.forecast_days == 'all':
            if not self.climatology is None:
                return self.y[index:len(self.y)], self.x[index:len(self.x)], self.climatology[:,index:len(self.y)]
            else:
                return self.y[index:len(self.y)], self.x[index:len(self.x)]
        else:
            if not self.climatology is None:
                return self.y[index:(index+self.lead_time)], self.x[index:(index+self.lead_time)], self.climatology[:,index:(index+self.lead_time)]
            else:
                return self.y[index:(index+self.lead_time)], self.x[index:(index+self.lead_time)]


