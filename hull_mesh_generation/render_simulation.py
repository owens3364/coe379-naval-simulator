"""
Renderer for precomputed ocean simulation data.

Usage:
    python render_simulation.py ocean.bin boat_motion.txt mesh.obj

Controls:
    Space       pause / resume
    Left/Right  step one frame
    R           reset to frame 0
    Mouse       rotate / zoom (turntable camera)
"""

import sys
import struct
import numpy as np
from numpy.typing import NDArray
from typing import Any
import trimesh
from vispy import app, scene

HEADER_MAGIC = 0x4E45434F
HEADER_FORMAT = '<IIIIIff36x'
HEADER_SIZE = 64

def parse_header(path: str) -> dict:
  with open(path, 'rb') as f:
    raw = f.read(HEADER_SIZE)
  magic, _, rows, cols, n_frames, fps, extent = struct.unpack(HEADER_FORMAT, raw)
  assert magic == HEADER_MAGIC, f"Bad magic: {magic:#010x}"
  return dict(rows=rows, cols=cols, n_frames=n_frames, fps=fps, extent=extent)

def load_frame(mm: np.memmap, frame_idx: int, rows: int, cols: int) -> tuple[float, NDArray]:
  frame_floats = 1 + rows * cols
  offset = frame_idx * frame_floats
  ts = float(mm[offset])
  Z = mm[offset + 1 : offset + 1 + rows * cols].reshape(rows, cols)
  return ts, Z

def load_boat_motion(path: str) -> NDArray:
    data = np.loadtxt(path, skiprows=1)
    # remove DC offset so ship starts near origin visually
    data[:, 1:] -= data[0, 1:]  # subtract t=0 values from all DOFs
    return data

def compute_ocean_faces(rows: int, cols: int) -> NDArray[np.int32]:
  faces = []
  for i in range(rows - 1):
    for j in range(cols - 1):
      idx = i * cols + j
      tl, tr, bl, br = idx, idx + 1, idx + cols, idx + cols + 1
      faces.append([tl, bl, br])
      faces.append([tl, br, tr])
  return np.array(faces, dtype=np.int32)

def rotation_matrix(roll: float, pitch: float, yaw: float) -> NDArray:
  cx, sx = np.cos(roll),  np.sin(roll)
  cy, sy = np.cos(pitch), np.sin(pitch)
  cz, sz = np.cos(yaw),   np.sin(yaw)
  Rx = np.array([[1,  0,   0 ], [0,  cx, -sx], [0,  sx,  cx]])
  Ry = np.array([[cy, 0,   sy], [0,  1,   0 ], [-sy, 0,  cy]])
  Rz = np.array([[cz, -sz, 0 ], [sz, cz,  0 ], [0,   0,  1 ]])
  return Rz @ Ry @ Rx

def transform_boat(verts_rest: NDArray, motion_row: NDArray) -> NDArray:
  _, surge, sway, heave, roll, pitch, yaw = motion_row
  R = rotation_matrix(roll, pitch, yaw)
  center = verts_rest.mean(axis=0)
  verts = (verts_rest - center) @ R.T + center
  verts += np.array([surge, sway, heave + 5.0])
  dominant_angle = np.arctan2(-1, -1)  # = -3π/4
  cos_a, sin_a = np.cos(dominant_angle), np.sin(dominant_angle)
  R_align = np.array([[cos_a, -sin_a, 0],
                      [sin_a,  cos_a, 0],
                      [0,      0,     1]], dtype=np.float32)
  boat_verts_rest = boat_verts_rest @ R_align.T
  return verts.astype(np.float32)

def main(ocean_path: str, motion_path: str, obj_path: str) -> None:
  hdr = parse_header(ocean_path)
  rows, cols, n_frames, fps, extent = (hdr['rows'], hdr['cols'], hdr['n_frames'], hdr['fps'], hdr['extent'])
  print(f"Ocean: {rows}x{cols}, {n_frames} frames @ {fps}fps, extent=±{extent}m")

  mm = np.memmap(ocean_path, dtype=np.float32, mode='r', offset=HEADER_SIZE)

  motion = load_boat_motion(motion_path)
  print(f"Boat motion: {len(motion)} timesteps")

  boat_mesh = trimesh.load(obj_path)
  boat_verts_rest = np.array(boat_mesh.vertices, dtype=np.float32)
  boat_verts_rest -= boat_verts_rest.mean(axis=0)
  boat_faces      = np.array(boat_mesh.faces,    dtype=np.int32)
  print(f"Boat mesh: {len(boat_verts_rest)} verts, {len(boat_faces)} faces")

  x = np.linspace(-extent, extent, cols, dtype=np.float32)
  y = np.linspace(-extent, extent, rows, dtype=np.float32)
  X, Y = np.meshgrid(x, y)
  ocean_faces = compute_ocean_faces(rows, cols)

  canvas = scene.SceneCanvas(
      title='Ocean Simulation', keys='interactive', show=True, bgcolor='#0a0f1a')
  view = canvas.central_widget.add_view()
  view.camera = 'turntable'
  view.camera.set_range(
      x=(-extent, extent), y=(-extent, extent), z=(-5, 5))

  _, Z = load_frame(mm, 0, rows, cols)
  Z_flat = Z.ravel()

  ocean_verts = np.c_[X.ravel(), Y.ravel(), Z_flat].astype(np.float32)
  norm_Z = (Z_flat - Z_flat.min()) / (np.ptp(Z_flat) + 1e-8)
  ocean_colors = np.c_[
      0.05 + 0.15 * norm_Z,
      0.25 + 0.30 * norm_Z,
      0.60 + 0.30 * norm_Z,
      np.ones_like(norm_Z)
  ].astype(np.float32)

  ocean = scene.visuals.Mesh(
      vertices=ocean_verts, faces=ocean_faces,
      vertex_colors=ocean_colors, shading='smooth')
  ocean.shading_filter.ambient_light       = (0.2, 0.5, 0.9, 0.3)
  ocean.shading_filter.ambient_coefficient = (0.1, 0.3, 0.7, 1.0)
  ocean.shading_filter.shininess           = 150
  ocean.update_gl_state(blend=True, depth_test=True)
  view.add(ocean)

  boat_verts_0 = transform_boat(boat_verts_rest, motion[0])
  boat_color = np.array([[0.85, 0.82, 0.75, 1.0]] * len(boat_verts_rest),
                            dtype=np.float32)
  boat = scene.visuals.Mesh(
      vertices=boat_verts_0, faces=boat_faces,
      vertex_colors=boat_color, shading='smooth')
  boat.update_gl_state(blend=True, depth_test=True)
  view.add(boat)

  info = scene.visuals.Text(
      '', color='white', font_size=10, pos=(10, 20),
      anchor_x='left', anchor_y='top',
      parent=canvas.scene)

  state = {'frame': 0, 'playing': True}

  def go_to_frame(frame_idx: int) -> None:
    frame_idx = max(0, min(n_frames - 1, frame_idx))
    state['frame'] = frame_idx

    ts, Z = load_frame(mm, frame_idx, rows, cols)
    Z_flat = Z.ravel()
    ocean_verts[:, 2] = Z_flat
    norm_Z = (Z_flat - Z_flat.min()) / (np.ptp(Z_flat) + 1e-8)
    ocean_colors[:, 0] = 0.05 + 0.15 * norm_Z
    ocean_colors[:, 1] = 0.25 + 0.30 * norm_Z
    ocean.set_data(vertices=ocean_verts, faces=ocean_faces, vertex_colors=ocean_colors)

    motion_idx = np.searchsorted(motion[:, 0], ts)
    motion_idx = min(motion_idx, len(motion) - 1)
    boat_verts = transform_boat(boat_verts_rest, motion[motion_idx])
    boat.set_data(vertices=boat_verts, faces=boat_faces, vertex_colors=boat_color)

    info.text = (f"Frame {frame_idx}/{n_frames-1}  t={ts:.2f}s  "
                  f"{'▶' if state['playing'] else '⏸'}  "
                  f"[Space=pause  ←→=step  R=reset]")

  go_to_frame(0)

  def update(_: Any) -> None:
    if not state['playing']:
      return
    next_frame = state['frame'] + 1
    if next_frame >= n_frames:
      next_frame = 0
    go_to_frame(next_frame)

  _ = app.Timer(interval=1.0 / fps, connect=update, start=True)

  @canvas.connect
  def on_key_press(event: Any) -> None:
    if event.key == 'Space':
      state['playing'] = not state['playing']
      go_to_frame(state['frame'])
    elif event.key == 'Right':
      state['playing'] = False
      go_to_frame(state['frame'] + 1)
    elif event.key == 'Left':
      state['playing'] = False
      go_to_frame(state['frame'] - 1)
    elif event.key == 'R':
      go_to_frame(0)

  app.run()

if __name__ == '__main__':
  if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} ocean.bin boat_motion.txt mesh.obj")
    sys.exit(1)
  main(sys.argv[1], sys.argv[2], sys.argv[3])
