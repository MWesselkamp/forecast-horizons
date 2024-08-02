import torch
import torch.nn as nn
class Ricker_Predation(nn.Module):
    """

    """

    def __init__(self, params, noise = None):
        super().__init__()
        if not noise is None:
            self.model_params = torch.nn.Parameter(torch.tensor(params + [noise], requires_grad=True, dtype=torch.double))
            self.noise = noise
        else:
            self.model_params = torch.nn.Parameter(torch.tensor(params, requires_grad=True, dtype=torch.double))
            self.noise = noise

    def forward(self, Temp):

        if not self.noise is None:
            alpha1, beta1, gamma1, bx1, cx1, alpha2, beta2, gamma2, bx2, cx2, sigma = self.model_params
        else:
            alpha1, beta1, gamma1, bx1, cx1, alpha2, beta2, gamma2, bx2, cx2 = self.model_params

        Temp = Temp.squeeze()
        out = torch.ones((2,len(Temp)), dtype=torch.double)

        if not self.noise is None:
            for i in range(len(Temp) - 1):
                out[0, i + 1] = out.clone()[0, i] * torch.exp(alpha1*(1 - beta1*out.clone()[0, i] - gamma1*out.clone()[1, i] + bx1 * Temp[i] + cx1 * Temp[i]**2)) \
                                + sigma*torch.normal(mean=torch.tensor([0.0,]), std=torch.tensor([0.1]))
                out[1, i + 1] = out.clone()[1, i] * torch.exp(alpha2*(1 - beta2*out.clone()[1, i] - gamma2*out.clone()[0, i] + bx2 * Temp[i] + cx2 * Temp[i]**2)) \
                                + sigma*torch.normal(mean=torch.tensor([0.0,]), std=torch.tensor([0.1]))
        else:
            for i in range(len(Temp) - 1):
                out[0, i + 1] = out.clone()[0, i] * torch.exp(alpha1*(1 - beta1*out.clone()[0, i] - gamma1*out.clone()[1, i] + bx1 * Temp[i] + cx1 * Temp[i]**2))
                out[1, i + 1] = out.clone()[1, i] * torch.exp(alpha2*(1 - beta2*out.clone()[1, i] - gamma2*out.clone()[0, i] + bx2 * Temp[i] + cx2 * Temp[i]**2))

        return out

    def __repr__(self):
        return f" alpha1: {self.model_params[0].item()}, \
            beta1: {self.model_params[1].item()}, \
                gamma1: {self.model_params[2].item()}, \
        bx1: {self.model_params[3].item()}, \
                    cx1: {self.model_params[4].item()}, \
                        alpha2: {self.model_params[5].item()}, \
        beta2: {self.model_params[6].item()}, \
                    gamma2: {self.model_params[7].item()}, \
                        bx2: {self.model_params[8].item()}, \
                    cx2: {self.model_params[9].item()},\
               sigma:{self.noise} "


class Ricker(nn.Module):
    """
     Single species extended Ricker Model
    """

    def __init__(self, params):

        super().__init__()

        self.model_params = torch.nn.Parameter(torch.tensor(params, requires_grad=True, dtype=torch.double))

    def forward(self, N0, Temp, sigma):

        alpha, beta, bx, cx = self.model_params

        Temp = Temp.squeeze()

        out = torch.zeros_like(Temp, dtype=torch.double)
        out[0] = N0 # initial value

        for i in range(len(Temp)-1):
            # Introduce randomness
            epsilon = sigma * torch.normal(mean=torch.tensor([0.0]), std=torch.tensor([1.0]))

            out[i+1] = out.clone()[i] * torch.exp(alpha * (1 - beta * out.clone()[i] + bx * Temp[i] + cx * Temp[i] ** 2)) \
                           + epsilon

        return out

    def __repr__(self):
        return f" alpha: {self.model_params[0].item()}, \
            beta: {self.model_params[1].item()}, \
                bx: {self.model_params[2].item()}, \
                    cx: {self.model_params[3].item()}, \
               sigma: {self.noise}"


class Ricker_Ensemble(nn.Module):
    """
     Single-Species extended Ricker with Ensemble prediction.
    """

    def __init__(self, params, noise=None, initial_uncertainty = None):

        super().__init__()

        if (not noise is None) & (not initial_uncertainty is None):
            self.model_params = torch.nn.Parameter(torch.tensor(params + [noise] + [initial_uncertainty], requires_grad=True, dtype=torch.double))
            self.initial_uncertainty = initial_uncertainty
            self.noise = noise
        elif (not noise is None) & (initial_uncertainty is None):
            self.model_params = torch.nn.Parameter(torch.tensor(params + [noise], requires_grad=True, dtype=torch.double))
            self.initial_uncertainty = initial_uncertainty
            self.noise = noise
        elif (noise is None) & (not initial_uncertainty is None):
            self.model_params = torch.nn.Parameter(torch.tensor(params + [initial_uncertainty], requires_grad=True, dtype=torch.double))
            self.initial_uncertainty = initial_uncertainty
            self.noise = noise
        elif (noise is None) & (initial_uncertainty is None):
            self.model_params = torch.nn.Parameter(torch.tensor(params, requires_grad=True, dtype=torch.double))
            self.noise = noise
            self.initial_uncertainty = initial_uncertainty

    def forward(self, N0, Temp, ensemble_size=15):

        if (not self.noise is None) & (not self.initial_uncertainty is None):
            alpha, beta, bx, cx, sigma, phi = self.model_params
        elif (not self.noise is None) & (self.initial_uncertainty is None):
            alpha, beta, bx, cx, sigma = self.model_params
        elif (self.noise is None) & (not self.initial_uncertainty is None):
            alpha, beta, bx, cx, phi = self.model_params
        else:
            alpha, beta, bx, cx = self.model_params

        Temp = Temp.squeeze()

        if not self.initial_uncertainty is None:
            initial = N0 + phi * torch.normal(torch.zeros((ensemble_size)), torch.repeat_interleave(torch.tensor([.1, ]), ensemble_size))
            out = torch.zeros((len(initial), len(Temp)), dtype=torch.double)
        else:
            initial = N0
            out = torch.zeros((1, len(Temp)), dtype=torch.double)

        out[:,0] = initial  # initial value

        if not self.noise is None:
            for i in range(len(Temp) - 1):
                out[:,i + 1] = out.clone()[:,i] * torch.exp(
                    alpha * (1 - beta * out.clone()[:,i] + bx * Temp[i] + cx * Temp[i] ** 2))# \
                          #   + sigma * torch.normal(mean=torch.tensor([0.0, ]), std=torch.tensor([1.0, ]))
            var = sigma * 2#torch.normal(mean=torch.tensor([0.0, ]), std=torch.tensor([1.0, ]))
            out_upper = out + torch.repeat_interleave(var, len(Temp)) #+ torch.full_like(out, var.item())
            out_lower = out - torch.repeat_interleave(var, len(Temp))  #- torch.full_like(out, var.item())

            return out, [out_upper, out_lower]

        else:
            for i in range(len(Temp) - 1):
                out[:,i + 1] = out.clone()[:,i] * torch.exp(
                    alpha * (1 - beta * out.clone()[:,i] + bx * Temp[i] + cx * Temp[i] ** 2))

            return out, None

    def get_fit(self):

        return {"alpha": self.model_params[0].item(), \
            "beta": self.model_params[1].item(), \
                "bx": self.model_params[2].item(), \
                    "cx": self.model_params[3].item(), \
               "sigma": self.noise if self.noise is None else self.model_params[4].item(), \
               "phi": self.initial_uncertainty if self.initial_uncertainty is None else (self.model_params[5].item() if self.noise is not None else self.model_params[4].item())
                }

    def forecast(self, N0, Temp, ensemble_size=15):

        if (not self.noise is None) & (not self.initial_uncertainty is None):
            alpha, beta, bx, cx, sigma, phi = self.model_params
        elif (not self.noise is None) & (self.initial_uncertainty is None):
            alpha, beta, bx, cx, sigma = self.model_params
        elif (self.noise is None) & (not self.initial_uncertainty is None):
            alpha, beta, bx, cx, phi = self.model_params
        else:
            alpha, beta, bx, cx = self.model_params

        Temp = Temp.squeeze()

        if not self.initial_uncertainty is None:
            initial = N0 + phi * torch.normal(torch.zeros((ensemble_size)),
                                              torch.repeat_interleave(torch.tensor([.1, ]), ensemble_size))
            out = torch.zeros((len(initial), len(Temp)), dtype=torch.double)
        else:
            initial = N0
            out = torch.zeros((1, len(Temp)), dtype=torch.double)

        out[:, 0] = initial  # initial value

        if not self.noise is None:
            for i in range(len(Temp) - 1):
                out[:, i + 1] = out.clone()[:, i] * torch.exp(
                    alpha * (1 - beta * out.clone()[:, i] + bx * Temp[i] + cx * Temp[i] ** 2))  \
                    + sigma * torch.normal(mean=torch.tensor([0.0, ]), std=torch.tensor([0.1, ]))

            #out = out + sigma * torch.normal(mean=torch.tensor([0.0, ]), std=torch.tensor([1.0, ]))

            return out

        else:
            for i in range(len(Temp) - 1):
                out[:, i + 1] = out.clone()[:, i] * torch.exp(
                    alpha * (1 - beta * out.clone()[:, i] + bx * Temp[i] + cx * Temp[i] ** 2))

            return out
