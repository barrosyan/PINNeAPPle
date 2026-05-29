"""Mesh and geometry visualisation utilities."""
from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.figure import Figure

from .style import get_cmap, make_figure, DEFAULT_CMAP


def _to_np(t):
    try:
        return t.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(t)


def plot_mesh(
    nodes: "ArrayLike",
    elems: "ArrayLike",
    *,
    node_values: Optional["ArrayLike"] = None,
    cmap: str = DEFAULT_CMAP,
    edge_color: str = "#4c9be8",
    edge_lw: float = 0.5,
    node_size: float = 3.0,
    title: str = "Mesh",
    show: bool = False,
) -> Figure:
    """
    Plot a 2-D mesh (triangles or quads).

    nodes      : (N, 2) node coordinates
    elems      : (E, 3) or (E, 4) element connectivity
    node_values: (N,) scalar to colour nodes
    """
    n = _to_np(nodes)
    e = _to_np(elems)

    fig, ax = make_figure(figsize=(8, 7))

    # Draw edges
    segments = []
    for el in e:
        for k in range(len(el)):
            i0, i1 = el[k], el[(k + 1) % len(el)]
            segments.append([n[i0], n[i1]])
    lc = mc.LineCollection(segments, colors=edge_color, linewidths=edge_lw, alpha=0.6)
    ax.add_collection(lc)

    # Draw nodes, optionally coloured
    if node_values is not None:
        nv = _to_np(node_values)
        sc = ax.scatter(n[:, 0], n[:, 1], c=nv, cmap=get_cmap(cmap), s=node_size, zorder=3)
        fig.colorbar(sc, ax=ax, label="node value")
    else:
        ax.scatter(n[:, 0], n[:, 1], c=edge_color, s=node_size, zorder=3, alpha=0.7)

    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(f"{title}  |  {n.shape[0]} nodes, {e.shape[0]} elements")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if show:
        plt.show()
    return fig


def plot_boundary(
    nodes: "ArrayLike",
    boundary_groups: dict,
    *,
    title: str = "Boundary groups",
    show: bool = False,
) -> Figure:
    """
    Visualise labelled boundary groups.

    boundary_groups : {"left": tensor_of_node_ids, "right": ...}
    """
    n = _to_np(nodes)
    fig, ax = make_figure(figsize=(7, 6))
    ax.scatter(n[:, 0], n[:, 1], c="gray", s=2, alpha=0.3, label="interior")

    palette = plt.cm.Set1(np.linspace(0, 1, max(len(boundary_groups), 1)))
    for i, (name, idx) in enumerate(boundary_groups.items()):
        idx_ = _to_np(idx).ravel().astype(int)
        ax.scatter(n[idx_, 0], n[idx_, 1], c=[palette[i]], s=20, label=name, zorder=3)

    ax.legend(fontsize=9, markerscale=2)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if show:
        plt.show()
    return fig


def plot_point_cloud(
    points: "ArrayLike",
    values: Optional["ArrayLike"] = None,
    *,
    cmap: str = "plasma",
    s: float = 3.0,
    title: str = "Point cloud",
    show: bool = False,
    dim3: bool = False,
) -> Figure:
    """
    2-D (or 3-D) scatter of a point cloud, optionally coloured by values.
    """
    pts = _to_np(points)
    if pts.shape[1] >= 3 and dim3:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                        c=_to_np(values) if values is not None else "#4c9be8",
                        cmap=get_cmap(cmap), s=s, alpha=0.7)
        if values is not None:
            fig.colorbar(sc, ax=ax, pad=0.1)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    else:
        fig, ax = make_figure(figsize=(7, 6))
        sc = ax.scatter(pts[:, 0], pts[:, 1],
                        c=_to_np(values) if values is not None else "#4c9be8",
                        cmap=get_cmap(cmap), s=s, alpha=0.7)
        if values is not None:
            fig.colorbar(sc, ax=ax, label="value")
        ax.set_aspect("equal")
        ax.set_xlabel("x"); ax.set_ylabel("y")

    plt.title(f"{title}  |  {pts.shape[0]} points")
    if show:
        plt.show()
    return fig
