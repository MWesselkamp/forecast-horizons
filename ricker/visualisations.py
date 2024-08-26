import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_losses(losses, loss_fun, log=True, saveto=''):
    if log:
        ll = np.log(torch.stack(losses).detach().numpy())
    else:
        ll = torch.stack(losses).detach().numpy()
    plt.plot(ll)
    plt.ylabel(f'{loss_fun} loss')
    plt.xlabel(f'Epoch')
    plt.savefig(os.path.join(saveto, 'losses.pdf'))
    plt.close()

def plot_posterior(df, saveto=''):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    axes[0, 0].hist(df["alpha"], bins=10, edgecolor='black')
    axes[0, 0].set_title("Histogram of alpha")
    axes[0, 1].hist(df["beta"], bins=10, edgecolor='black')
    axes[0, 1].set_title("Histogram of beta")
    axes[1, 0].hist(df["bx"], bins=10, edgecolor='black')
    axes[1, 0].set_title("Histogram of bx")
    axes[1, 1].hist(df["cx"], bins=10, edgecolor='black')
    axes[1, 1].set_title("Histogram of cx")
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(saveto, 'posterior.pdf'))
    plt.close()

def plot_ppp(species1_PPP, species2_PPP, PPP_threshold):
    plt.figure(figsize=(7, 5))
    plt.plot(species1_PPP, label="Species 1", color="blue")
    plt.plot(species2_PPP, label="Species 2", color="green")
    plt.hlines(PPP_threshold, xmin=0, xmax=365, color="black", linestyles='--')

    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Potential Prognostic Predictability", fontsize=16)
    # Set font size for tick labels to 16
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=16)

    # Save the plot to a file
    plt.savefig("ricker/plots/ricker_forecast_horizon.pdf")
    #plt.tight_layout()
    plt.tight_layout()
    plt.show()


def plot_ensemble(ensemble):
    plt.figure(figsize=(7, 5))
    plt.plot(ensemble[:,1,:].detach().numpy().squeeze().transpose(),
             alpha = 0.3, color = "lightgreen")
    plt.plot(ensemble[:, 0, :].detach().numpy().squeeze().transpose(),
         alpha=0.3, color="lightblue")
    plt.plot(ensemble[:, 1, :].detach().numpy().squeeze().transpose().mean(axis=1),
             color="green", label="Species 2")
    plt.plot(ensemble[:, 0, :].detach().numpy().squeeze().transpose().mean(axis=1),
             color="blue", label="Species 1")
    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Relative Population Size", fontsize=16)
    # Set font size for tick labels to 16
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=16)

    # Save the plot to a file
    plt.savefig("ricker/plots/ensemble.pdf")
    #plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
    plt.legend()
    plt.show()
def plot_combined(species1_PPP, species2_PPP, PPP_threshold, ensemble):
    # Create a figure with two subplots arranged in two rows
    fig, ax = plt.subplots(2, 1, figsize=(8, 8),
                           sharex= True)  # 2 rows, 1 column

    # Plot the ensemble in the second subplot
    ax[0].plot(ensemble[:, 1, :].detach().numpy().squeeze().transpose(),
               alpha=0.3, color="lightgreen")
    ax[0].plot(ensemble[:, 0, :].detach().numpy().squeeze().transpose(),
               alpha=0.3, color="lightblue")
    ax[0].plot(ensemble[:, 1, :].detach().numpy().squeeze().transpose().mean(axis=1),
               color="green", label="Species 2")
    ax[0].plot(ensemble[:, 0, :].detach().numpy().squeeze().transpose().mean(axis=1),
               color="blue", label="Species 1")

    #ax[0].set_xlabel("Time", fontsize=16)
    ax[0].set_ylabel("Relative Population Size", fontsize=16)
    ax[0].tick_params(axis='both', which='major', labelsize=16)

    handles, labels = ax[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax[0].legend(by_label.values(), by_label.keys(), fontsize=16)

    # Plot PPP for both species in the first subplot
    ax[1].plot(species1_PPP, label="Species 1", color="blue")
    ax[1].plot(species2_PPP, label="Species 2", color="green")
    ax[1].hlines(PPP_threshold, xmin=0, xmax=365, color="black", linestyles='--')

    ax[1].set_xlabel("Time", fontsize=16)
    ax[1].set_ylabel("Potential Prognostic Predictability", fontsize=16)
    ax[1].tick_params(axis='both', which='major', labelsize=16)

    handles, labels = ax[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #ax[1].legend(by_label.values(), by_label.keys(), fontsize=16)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the combined plot to a file
    plt.savefig("ricker/plots/combined_plot.pdf")

    # Show the combined plot
    plt.show()

def plot_horizon_maps(iterated_dynamics_species1, iterated_dynamics_species2):
    fig, ax = plt.subplots(1, 2, figsize=(9, 9), sharey=True)  # Two subplots side by side, sharing y-axis

    # Plot CRPS heatmap for iterated_dynamics_species1
    heatmap1 = ax[0].imshow(iterated_dynamics_species1.transpose(),
                            vmin=0, origin='lower', cmap='OrRd_r')
    ax[0].set_xlabel('Initial time', fontsize=18)
    ax[0].set_ylabel('Lead time', fontsize=18)
    ax[0].set_title('Species 1', fontsize=18)

    # Plot CRPS heatmap for iterated_dynamics_species2
    heatmap2 = ax[1].imshow(iterated_dynamics_species2.transpose(),
                            vmin=0, origin='lower', cmap='OrRd_r')
    ax[1].set_xlabel('Initial time', fontsize=18)
    ax[1].set_title('Species 2', fontsize=18)

    # Reduce horizontal space between the subplots
    plt.subplots_adjust(wspace=-0.6)

    # Share a single colorbar for both subplots
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="9%", pad=0.2)
    cb = plt.colorbar(heatmap2, cax=cax)
    cb.set_label('Potential Prognostic Predictability', fontsize=18)
    cb.ax.tick_params(labelsize=18)

    # Adjust tick label fontsize for both subplots
    plt.setp(ax[0].get_xticklabels(), fontsize=18)
    plt.setp(ax[0].get_yticklabels(), fontsize=18)
    plt.setp(ax[1].get_xticklabels(), fontsize=18)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig('ricker/plots/horizon_maps.pdf')
    plt.show()

