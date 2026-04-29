import capytaine as cpt
import json
import numpy as np
import xarray as xr
from series_60_offsets import STATIONS, MAX_STATION, WATERLINES, MAX_WATERLINE, CB_VALUES, OFFSETS

# TODO: dofs and center of rotation and mass and such should be appropriately defined(?)

# constants

L = 100.0 # length (m)
B = 15.0 # beam (m)
T = 10.0 # draft (m)

# helper function definitions

def vertex_idx(station_idx: int, num_waterlines: int, waterline_idx: int, port: bool):
    return (station_idx * num_waterlines + waterline_idx) * 2 + (1 if port else 0)

def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, xr.DataArray):
        return {
            "data": obj.values.tolist(),
            "dims": obj.dims,
            "coords": {k: make_serializable(v.values) for k, v in obj.coords.items()},
            "name": obj.name
        }
    else:
        try:
            return obj.item()
        except:
            return obj

# get cb value and offsets

cb = float(input("Enter the Cb value to use: "))

if cb not in CB_VALUES:
    print("This Cb value is unavailable.")
    exit()

offsets = OFFSETS[np.where(CB_VALUES == cb)[0][0]]

# create points for mesh

x_coords = STATIONS / MAX_STATION * L
z_coords = (WATERLINES[WATERLINES <= 1.0] * T) - T # shift waterplane to z=0
offsets = offsets[:, WATERLINES <= 1.0] # prevent waterline above deck

mesh_points = []
for station_idx, x in enumerate(x_coords):
    for waterline_idx, z in enumerate(z_coords):
        y = offsets[station_idx, waterline_idx] * (B / 2)
        mesh_points.append([x, y, z])
        mesh_points.append([x, -y, z])

# build mesh TODO: interpolation for finer mesh

# mesh points are vertices
faces = []

num_stations = len(x_coords)
num_waterlines = len(z_coords)
for i in range(num_stations - 1):
    # keel strip
    v1 = vertex_idx(i, num_waterlines, 0, False)
    v2 = vertex_idx(i+1, num_waterlines, 0, False)
    v3 = vertex_idx(i+1, num_waterlines, 0, True)
    v4 = vertex_idx(i, num_waterlines, 0, True)
    faces.append([v1, v2, v3, v4])

    # regular faces
    for j in range(num_waterlines - 1):
        for port in [True, False]:
            v1 = vertex_idx(i, num_waterlines, j, port)
            v2 = vertex_idx(i+1, num_waterlines, j, port)
            v3 = vertex_idx(i+1, num_waterlines, j+1, port)
            v4 = vertex_idx(i, num_waterlines, j+1, port)
            # faces.append([v1, v2, v3])
            # faces.append([v1, v3, v4])
            if port:
                faces.append([v4, v3, v2, v1])
            else:
                faces.append([v1, v2, v3, v4])

# close bow and stern
for i in [0, num_stations-1]:
    for j in range(num_waterlines - 1):
        v1 = vertex_idx(i, num_waterlines, j, False)
        v2 = vertex_idx(i, num_waterlines, j+1, False)
        v3 = vertex_idx(i, num_waterlines, j+1, True)
        v4 = vertex_idx(i, num_waterlines, j, True)
        faces.append([v1, v2, v3, v4])

# capytaine computations

mesh = cpt.Mesh(vertices=np.array(mesh_points), faces=np.array(faces))
body = cpt.FloatingBody( # we may need more parameters
    mesh=mesh,
    center_of_mass=(L/2, 0.0, -T/2) # TODO: check
)
body.add_all_rigid_body_dofs()

hydrostatics = body.compute_hydrostatics() # we may need more parameters

print("\nGenerated Hydrostatic Coefficients")

for k, v in hydrostatics.items():
    print(f"{k}: {v}")

added_masses = []
radiation_dampings = []

# om = np.linspace(0.2, 2.0, 10)
om = [1.0] # TODO: replace all this stuff with JONSWAP wave spectra

solver = cpt.BEMSolver()
for o in om:
    added_masses.append([])
    radiation_dampings.append([])
    for dof in body.dofs:
        rp = cpt.RadiationProblem(body=body, omega=o, radiating_dof=dof) # TODO: specify radiating dof
        res = solver.solve(rp)
        added_masses[-1].append(res.added_mass)
        radiation_dampings[-1].append(res.radiation_damping)

print(added_masses)
print(radiation_dampings)

# dump

output = {
    "params": {
        "L": L,
        "B": B,
        "T": T,
        "Cb": cb
    },
    "hydrostatic_coeffs": hydrostatics,
    "dofs": body.dofs,
    "radiation_data": [
        {
            "omega": o,
            "added_masses": added_masses[i],
            "radiation_dampings": radiation_dampings[i]
        }
        for i, o in enumerate(om)
    ]
}

with open("output.json", "w") as f:
    json.dump(make_serializable(output), f)
