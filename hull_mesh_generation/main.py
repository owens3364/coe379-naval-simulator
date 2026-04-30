import capytaine as cpt
from capytaine.post_pro import rao
import json
import numpy as np
import xarray as xr
import trimesh
from series_60_offsets import STATIONS, MAX_STATION, WATERLINES, CB_VALUES, OFFSETS

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
        if np.iscomplexobj(obj):
            return {"real": obj.real.tolist(), "imag": obj.imag.tolist()}
        return obj.tolist()
    elif isinstance(obj, xr.DataArray):
        vals = obj.values
        data = {"real": vals.real.tolist(), "imag": vals.imag.tolist()} if np.iscomplexobj(vals) else vals.tolist()
        return {
            "data": data,
            "dims": obj.dims,
            "coords": {k: make_serializable(v.values) for k, v in obj.coords.items()},
            "name": obj.name
        }
    elif isinstance(obj, xr.Dataset):
        return {var: make_serializable(obj[var]) for var in obj.data_vars}
    elif isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    elif isinstance(obj, float):
        if np.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        elif np.isnan(obj):
            return None
        return obj
    else:
        try:
            return obj.item()
        except:
            return obj

# get cb value and offsets

cb = float(input("Enter the Cb value to use: "))
export_mesh = input("Export mesh? (y/N): ").lower() == "y"

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

wave_spec_filepath = input('Enter the filepath of wave specifications that you would like to run the BEM solver for: ')
waves = []
with open(wave_spec_filepath) as f:
    n = int(f.readline())
    for _ in range(n):
        waves.append(list(map(float, f.readline().split())))


# capytaine computations

mesh = cpt.Mesh(vertices=np.array(mesh_points), faces=np.array(faces))

if export_mesh:
    tri_faces = []
    for face in faces:
        tri_faces.append([face[0], face[1], face[2]])
        tri_faces.append([face[0], face[2], face[3]])
    trimesh.Trimesh(vertices=mesh.vertices, faces=np.array(tri_faces)).export("mesh.obj")

body = cpt.FloatingBody( # we may need more parameters
    mesh=mesh,
    center_of_mass=(L/2, 0.0, -T/2) # TODO: check
)
body.add_all_rigid_body_dofs()

hydrostatics = body.compute_hydrostatics() # we may need more parameters

print("\nGenerated Hydrostatic Coefficients")

for k, v in hydrostatics.items():
    print(f"{k}: {v}")

frequencies = np.logspace(np.log10(0.05), np.log10(2.0), 50)
omega_vals = 2 * np.pi * frequencies

wave_directions = list(set(
    np.degrees(np.arctan2(w[5], w[4])) * np.pi / 180
    for w in waves
))

test_matrix = xr.Dataset(coords={
    'omega': omega_vals,
    'wave_direction': wave_directions,
    'radiating_dof': list(body.dofs.keys()),
    'rho': 1025.0,
    'water_depth': np.inf,
})

body.inertia_matrix = body.compute_rigid_body_inertia(rho=500.0)  # kg/m3
body.hydrostatic_stiffness = body.compute_hydrostatic_stiffness()

solver = cpt.BEMSolver()
dataset = solver.fill_dataset(test_matrix, body, hydrostatics=True)
RAO = rao(dataset)

output = {
    "params": {"L": L, "B": B, "T": T, "Cb": cb},
    "hydrostatic_coeffs": hydrostatics,
    "dofs": list(body.dofs.keys()),
    "rao": make_serializable(RAO),
}

with open("output.json", "w") as f:
    json.dump(make_serializable(output), f, indent=2)
