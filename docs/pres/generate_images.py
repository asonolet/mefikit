"""Generate the figures used by the `mefikit_version_courte.md` slide deck.

Produces three PNGs in `docs/images/`:

- `field_visualization.png`  — a 2D QUAD4 mesh with a computed scalar field and
  the associated "liquid" selection group.
- `polyhedral_mesh.png`      — the 2000-cell polyhedral mesh `mesh_36.med`.
- `polyhedral_remap.png`     — a conservative P0/P0 remap: source mesh_36 ->
  target mesh_27.

Inspired by `docs/python_examples/*.ipynb`, but with a light theme suited to the
presentation (the notebooks use the dark mdbook theme).

Run from the repository root:

    uv run python docs/pres/generate_images.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image

import mefikit as mf

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "tests" / "data"
OUT = Path(__file__).resolve().parents[1] / "images"

SIZE = (800, 900)
SIZE_SINGLE = (1600, 900)
GAP = 8

pv.set_plot_theme("document")
pv.set_jupyter_backend("static")


def save(img: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / name)
    print(f"  -> {OUT / name}")


def render_scene(setup, size: tuple[int, int] = SIZE) -> Image.Image:
    """Render a single panel using a fresh Plotter + reset_camera + zoom."""
    pt = pv.Plotter(off_screen=True, window_size=size)
    setup(pt)
    return Image.fromarray(pt.screenshot(None, return_img=True))


def hstack(*panels: Image.Image) -> Image.Image:
    sep = np.full((SIZE[1], GAP, 3), 255, dtype=np.uint8)
    parts = []
    for i, p in enumerate(panels):
        if i:
            parts.append(sep)
        parts.append(np.asarray(p))
    return Image.fromarray(np.concatenate(parts, axis=1))


def field_visualization() -> None:
    x = np.logspace(-3, 0.0, 40)
    mesh = mf.build_cmesh(x, x)
    mesh.fields["T"] = 1.0 + mf.X**2 + 0.5 * mf.Y

    T = mf.Field("T")
    liquid = mesh.select((T > 1.25) & (T < 1.95))
    liquid_ids = np.concatenate(list(liquid.ids().values()))

    all_pv = mesh.to_pyvista()
    all_pv["T"] = mesh.fields["T"].numpy()
    region = np.zeros(mesh.num_elements())
    region[liquid_ids] = 1.0
    all_pv["In"] = region

    def left(pt: pv.Plotter) -> None:
        pt.add_mesh(
            all_pv,
            scalars="T",
            cmap="viridis",
            show_edges=True,
            line_width=0.5,
            lighting=False,
        )
        pt.add_scalar_bar(title="T", fmt="%.2f")
        pt.camera_position = "xy"

    def right(pt: pv.Plotter) -> None:
        pt.add_mesh(
            all_pv,
            scalars="In",
            cmap=["#cfcfcf", "#e74c3c"],
            clim=[0, 1],
            show_edges=True,
            line_width=0.5,
            lighting=False,
            show_scalar_bar=False,
        )
        pt.camera_position = "xy"

    save(hstack(render_scene(left), render_scene(right)), "field_visualization.png")


def polyhedral_mesh() -> None:
    mesh = mf.UMesh.read(str(DATA / "mesh_36.med"))
    boundary = mesh.boundaries().to_pyvista()

    def setup(pt: pv.Plotter) -> None:
        pt.add_mesh(
            boundary,
            color="#9bb8d3",
            show_edges=True,
            edge_color="#2c3e50",
            line_width=0.6,
        )
        pt.camera_position = "iso"
        pt.reset_camera()
        pt.camera.zoom(0.95)

    save(render_scene(setup, SIZE_SINGLE), "polyhedral_mesh.png")


def polyhedral_remap() -> None:
    src = mf.UMesh.read(str(DATA / "mesh_36.med"))
    tgt = mf.UMesh.read(str(DATA / "mesh_27.med"))

    src_faces = src.boundaries().to_pyvista()
    tgt_faces = tgt.boundaries().to_pyvista()

    def source(pt: pv.Plotter) -> None:
        pt.add_text("Source : mesh_36.med", font_size=22, position=(20, 840))
        pt.add_mesh(
            src_faces,
            color="#9bb8d3",
            show_edges=True,
            edge_color="#2c3e50",
            line_width=0.6,
        )
        pt.camera_position = "iso"
        pt.reset_camera()
        pt.camera.zoom(0.9)

    def target(pt: pv.Plotter) -> None:
        pt.add_text("Target : mesh_27.med", font_size=22, position=(20, 840))
        pt.add_mesh(
            tgt_faces,
            color="#d3a99b",
            show_edges=True,
            edge_color="#7f4a3a",
            line_width=0.6,
        )
        pt.camera_position = "iso"
        pt.reset_camera()
        pt.camera.zoom(0.9)

    save(hstack(render_scene(source), render_scene(target)), "polyhedral_remap.png")


if __name__ == "__main__":
    print("Generating presentation figures...")
    field_visualization()
    polyhedral_mesh()
    polyhedral_remap()
    print("Done.")
