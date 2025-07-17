import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
S_0 = 1.20  # Spot price in dollars
r = 0.02  # Risk-free rate (2%)
sigma = 0.25  # Volatility (25%)
T = 0.5  # Time to maturity in years
num_sim = 100  # Number of simulations
num_step = 252  # Number of steps (daily)

# Time increment
dt = T / num_step

# Simulating price paths
np.random.seed(42)  # For reproducibility remove for random simulations arbitrary
price_paths = np.zeros((num_step, num_sim))
price_paths[0] = S_0

for t in range(1, num_step):
    z = np.random.standard_normal(num_sim)
    price_paths[t] = price_paths[t-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

# Plot random walks
plt.figure(figsize=(10, 5))
for i in range(num_sim):
    plt.plot(price_paths[:, i], alpha=0.7)

 # Calculating the average simulated price at maturity
average_simulated_price = np.mean(price_paths[-1])
print(f"The average simulated price of the coffee futures contract at maturity is ${average_simulated_price:.3f}.")


plt.xlabel("Time Steps (Days)")
plt.ylabel("Futures Price ($)")
plt.title("Simulated Random Walks of Futures Prices")
plt.grid(True)
plt.show()
