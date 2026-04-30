import numpy as np
import matplotlib.pyplot as plt
import jonswap_py as jp

GRID = 1000
EXTENT = 50
T = 0.0

min_wavelength = (2 * EXTENT) / (GRID / 2)
waves = jp.generate_waves(jp.JonswapConditions.STORMY)
waves = [w for w in waves if w.wavelength > min_wavelength]

x = np.linspace(-EXTENT, EXTENT, GRID, dtype=np.float32)
y = np.linspace(-EXTENT, EXTENT, GRID, dtype=np.float32)
X, Y = np.meshgrid(x, y)

Z = np.array(
    jp.height_grid(waves, X.ravel().tolist(), Y.ravel().tolist(), T)
).reshape(GRID, GRID)

fig = plt.figure(figsize=(12, 7))
ax  = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='ocean', linewidth=0, antialiased=True, alpha=0.9)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Height (m)')
ax.set_zlim(np.min(Z), np.max(Z))
ax.set_title(f'JONSWAP Ocean Surface — Stormy, t={T}s')
plt.tight_layout()
plt.show()