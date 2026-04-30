import matplotlib.pyplot as plt
import numpy as np

num_cores = [1, 5, 10, 15, 25, 35, 45, 55, 56]
serial_time = [3597.34, 3619.59, 3617.07, 3617.74, 3622.22, 3640.81, 3654.53, 3777.12, 3771.99]
mpi_time = [3597.34, 759.17, 382.22, 265.89, 153.76, 167.70, 112.06, 98.37, 227.67]

serial_baseline = np.mean(serial_time)
linear_speedup = [serial_baseline / (serial_baseline / r) for r in num_cores]
actual_speedup = [serial_baseline / t for t in mpi_time]
efficiency_pct = [(s / r) * 100 for s, r in zip(actual_speedup, num_cores)]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("JONSWAP MPI Scaling: $1000\\times1000$ Grid, $100$ Waves")

# timing
ax = axes[0]
ax.plot(num_cores, mpi_time, label='MPI elapsed time')
ax.axhline(serial_baseline, label=f'Serial baseline', color='orange')
ax.set_xlabel("Number of cores for MPI")
ax.set_ylabel("Time elapsed (ms)")
ax.set_title("Time Elapsed vs. Number of Cores Used")
ax.legend()
ax.grid(True, alpha=0.5)

# speedup data
ax = axes[1]
ax.plot(num_cores, actual_speedup, label='Actual speedup')
ax.plot(num_cores, linear_speedup, label='Theoretical linear speedup', color='orange')
ax.set_xlabel("Number of cores for MPI")
ax.set_ylabel("Speedup ($T_{serial} / T_{parallel}$)")
ax.set_title("Speedup vs. Number of Cores Used")
ax.legend()
ax.grid(True, alpha=0.5)

# efficiency data
ax = axes[2]
ax.plot(num_cores, efficiency_pct)
ax.axhline(100, label='$100\\%$ efficiency', color='orange')
ax.set_xlabel("Number of cores for MPI")
ax.set_ylabel("Parallel Efficiency (%)")
ax.set_title("Parallel Efficiency vs. Number of Cores Used")
ax.legend()
ax.grid(True, alpha=0.5)
ax.set_ylim(0, 120)

plt.tight_layout()
plt.savefig("results/jonswap_wave_gen_perf_graphs.png")
plt.show()
