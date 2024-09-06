import numpy as np
import matplotlib.pyplot as plt

# Parameters
phi = 0.5  # Autoregressive parameter (can be adjusted between -1 and 1)
sigma = 0.1  # Standard deviation of the white noise
n = 200  # Number of steps in the process
initial_value = 0  # Initial value of the process

# Initialize the array for the process
red_noise = np.zeros(n)
red_noise[0] = initial_value  # Set the initial value

# Simulate the red noise process
for t in range(1, n):
    red_noise[t] = phi * red_noise[t-1] + np.random.normal(0, sigma)

# Plot the red noise process
plt.figure(figsize=(10, 5))
plt.plot(red_noise, label='Red Noise Process', color='red')
plt.title('Simulated Red Noise (AR(1) Process)')
plt.xlabel('Time step')
plt.ylabel('Value')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()
plt.show()
