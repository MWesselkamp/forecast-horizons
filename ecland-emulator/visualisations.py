import matplotlib.pyplot as plt

def plot_station_data(y_prog, station_data_transformed, matching_indices, save_to):

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True) 

    ax[0].plot(y_prog[:,:,matching_indices[0]], color="blue", label="ecland")
    ax[0].plot(station_data_transformed[:,:,0], color="red", label="station data")
    ax[0].set_title("Layer 1")
    ax[0].legend()

    ax[1].plot(y_prog[:,:,matching_indices[1]], color="blue", label="ecland")
    ax[1].plot(station_data_transformed[:,:,1], color="red", label="station data")
    ax[1].set_title("Layer 2")
    ax[1].legend()

    ax[2].plot(y_prog[:,:,matching_indices[2]], color="blue", label="ecland")
    ax[2].plot(station_data_transformed[:,:,2], color="red", label="station data")  
    ax[2].set_title("Layer 3")
    ax[2].legend()

    plt.tight_layout()
    plt.savefig(save_to)
    plt.show()