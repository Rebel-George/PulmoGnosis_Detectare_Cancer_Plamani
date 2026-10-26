"""Matplotlib figures for the app: slices with PET overlay, crosshairs, patch boxes."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from pulmognosis.preprocessing.intensity import window

PLANES = {"axial": 0, "coronal": 1, "sagittal": 2}


def take_slice(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    """2D slice of a (z, y, x) volume. Coronal/sagittal rows run inferior → superior."""
    axis = PLANES[plane]
    return np.take(volume, int(np.clip(index, 0, volume.shape[axis] - 1)), axis=axis)


def _in_plane(plane: str, zyx) -> tuple:
    """(column, row) components of a (z, y, x) triple for the given plane."""
    z, y, x = zyx
    return {"axial": (x, y), "coronal": (x, z), "sagittal": (y, z)}[plane]


def slice_figure(
    ct_hu: np.ndarray,
    plane: str,
    index: int,
    ct_window: tuple[float, float],
    pet_suv: np.ndarray | None = None,
    pet_opacity: float = 0.5,
    pet_min_suv: float = 2.0,
    pet_max_suv: float = 10.0,
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    center_zyx: tuple[int, int, int] | None = None,
    box_mm: float | None = None,
    title: str | None = None,
    size: float = 5.0,
):
    """One slice at native resolution, displayed with true physical proportions."""
    origin = "upper" if plane == "axial" else "lower"
    col_mm, row_mm = _in_plane(plane, spacing_xyz[::-1])
    aspect = row_mm / col_mm

    fig, ax = plt.subplots(figsize=(size, size))
    fig.patch.set_alpha(0)
    ax.imshow(window(take_slice(ct_hu, plane, index), *ct_window), cmap="gray", vmin=0, vmax=1,
              origin=origin, aspect=aspect, interpolation="none")

    if pet_suv is not None and pet_opacity > 0:
        pet = take_slice(pet_suv, plane, index)
        masked = np.ma.masked_less(pet, pet_min_suv)
        ax.imshow(masked, cmap="hot", vmin=pet_min_suv, vmax=pet_max_suv, alpha=pet_opacity,
                  origin=origin, aspect=aspect, interpolation="none")

    if center_zyx is not None:
        col, row = _in_plane(plane, center_zyx)
        ax.axhline(row, color="#00e5ff", lw=0.8, alpha=0.7)
        ax.axvline(col, color="#00e5ff", lw=0.8, alpha=0.7)
        if box_mm:
            w, h = box_mm / col_mm, box_mm / row_mm
            ax.add_patch(Rectangle((col - w / 2, row - h / 2), w, h, fill=False, ec="#00e5ff", lw=1.5))

    if title:
        ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    return fig


def patch_figure(ct_patch: np.ndarray, pet_patch: np.ndarray | None, ct_window: tuple[float, float]):
    """Central axial, coronal and sagittal slices of the model's input cube (CT row, PET row)."""
    rows = 2 if pet_patch is not None else 1
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3.1 * rows), squeeze=False)
    fig.patch.set_alpha(0)
    mid = ct_patch.shape[0] // 2
    for j, plane in enumerate(PLANES):
        origin = "upper" if plane == "axial" else "lower"
        axes[0, j].imshow(window(take_slice(ct_patch, plane, mid), *ct_window), cmap="gray", vmin=0, vmax=1,
                          origin=origin, interpolation="none")
        axes[0, j].set_title(f"CT · {plane}", fontsize=9)
        if pet_patch is not None:
            axes[1, j].imshow(take_slice(pet_patch, plane, mid), cmap="hot", origin=origin, interpolation="none")
            axes[1, j].set_title(f"PET · {plane}", fontsize=9)
    for ax in axes.flat:
        ax.set_axis_off()
    fig.tight_layout(pad=0.3)
    return fig
