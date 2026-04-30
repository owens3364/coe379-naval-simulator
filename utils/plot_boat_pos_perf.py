import matplotlib.pyplot as plt

threads = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
static_time_total = [34.64, 20.30, 15.73, 14.02, 13.48, 13.33, 14.00, 10.29, 10.74, 17.07]
guided_time_total = guided_ms  = [33.57, 19.31, 15.43, 13.59, 22.95, 18.21, 19.74, 20.97, 16.33, 25.40]

time_total = guided_time_total # change this to plot different results

speedup = [time_total[0] / t for t in time_total]
linear_speedup = threads
efficiency = [(s / n) * 100 for s, n in zip(speedup, threads)]

fig, axes = plt.subplots(1, 3)
fig.suptitle("OpenMP Ship Motion with Guided Scheduling: $1200$ steps of $0.05$ s each, $100$ waves)")

# time
ax = axes[0]
ax.plot(threads, time_total)
ax.set_xlabel("Number of threads for OpenMP")
ax.set_ylabel("Time elapsed (ms)")
ax.set_title("Time Elapsed vs. Number of Threads Used")
ax.set_xticks(threads)
ax.grid(True, alpha=0.5)

# speedup
ax = axes[1]
ax.plot(threads, speedup, label='Actual Speedup')
ax.plot(threads, linear_speedup, label='Theoretical Linear Speedup', color='orange')
ax.set_xlabel("Number of threads for OpenMP")
ax.set_ylabel("Speedup ($T_{single-threaded} / T_{multi-threaded}$)")
ax.set_title("Speedup vs. Number of Threads Used")
ax.set_xticks(threads)
ax.legend()
ax.grid(True, alpha=0.5)

# efficiency
ax = axes[2]
ax.plot(threads, efficiency)
ax.axhline(100, label='100\\% efficiency', color='orange')
ax.set_xlabel("Number of threads for OpenMP")
ax.set_ylabel("Parallel Efficiency (%)")
ax.set_title("Parallel Efficiency vs. Number of Threads Used")
ax.set_xticks(threads)
ax.set_ylim(0, 120)
ax.legend()
ax.grid(True, alpha=0.5)

plt.tight_layout()
plt.savefig("results/boat_pos_guided_perf_graphs.png")
plt.show()
