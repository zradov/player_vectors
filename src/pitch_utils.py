import math
import mplsoccer
import numpy as np
from mplsoccer import Pitch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from highlight_text import ax_text
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter


def create_pitch() -> mplsoccer.Pitch:
    """
    Creates and returns mplsoccer.Pitch object.
    """
    pitch = Pitch(pitch_type="statsbomb", line_zorder=2, pitch_color="#22312b",
                  line_color="#efefef", linewidth=0.8)

    return pitch


def create_pitch_grid(pitch: Pitch, num_rows: int, num_cols: int) -> tuple[Figure, dict[str, Axes]]:
    """
    Creates a pitch grid and returns the top level container for all the plot elements and a
    dictionary mapping the labels to the Axes objects.

    Parameters
    ----------
    pitch: The football pitch abstraction object.
    num_rows: The number of rows in the pitch grid.
    num_cols: The number of columns in the pitch grid.
    """
    fig, axes = pitch.grid(nrows=num_rows, ncols=num_cols, figheight=30,
                          endnote_height=0.03, endnote_space=0,
                          axis=False, title_height=0.08,
                          grid_width=0.85, grid_height=0.85)
    fig.set_facecolor("white")

    return fig, axes

def plot_pitch_data(pitch: Pitch, data: np.array, ax: plt.Axes,
                    football_pitch_size: tuple[int, int], title: str) -> None:
    """
    Smoothes the data, calculates the data heatmap and plots it onto the pitch.
    It is used in a conjunction with the grid method of mplsoccer.Pitch object.

    Parameters
    ----------
    pitch: The football pitch abstraction object.
    data: The data that needs to be plotted onto the pitch.
    ax: The specific subplot in a figure.
    football_pitch_size: The height and width of the football pitch.
    title: The title of the pitch plot.
    """
    y, x = data.shape
    x_grid = np.linspace(0, football_pitch_size[0], x + 1)
    y_grid = np.linspace(0, football_pitch_size[1], y + 1)
    cx = x_grid[:-1] + 0.5 * (x_grid[1] - x_grid[0])
    cy = y_grid[:-1] + 0.5 * (y_grid[1] - y_grid[0])
    smoothed_data = gaussian_filter(data, 0.8)
    stats = dict(statistic=smoothed_data, x_grid=x_grid, y_grid=y_grid, cx=cx, cy=cy)
    _ = pitch.heatmap(stats, ax=ax, cmap="hot", edgecolors="#22312b")

    ax_text(0, -10, f"<{title}>", ha='left', va='center', fontsize=10,
            highlight_textprops=[{"color": '#000000'}], ax=ax)


def get_pitch_grid_dim(max_features: int, total_features_num: int,
                       max_rows: int = 3, max_cols: int = 5) -> tuple[int, int]:
    """
    Returns the number of rows and columns for the grid of mplsoccer.Pitch objects.

    Parameters
    ----------
    max_features: Limits the number of NMF model features that can be displayed. Useful when the total number of
                  features exceeds the number of features that can be displayed. This is done to improve features
                  visualization.
    total_features_num: Refers to the total number of components in the NMF model for particular action type.
    max_rows: The maximum number of rows in the grid.
    max_cols: The maximum number of columns per row.
    """
    if total_features_num >= max_features:
        return max_rows, max_cols
    cols = rows = 1
    while cols * rows < total_features_num:
        cols = math.ceil(total_features_num / rows)
        if cols > max_cols:
            rows += 1
            # Preferred number of columns per row
            cols = 3
        elif rows > max_rows:
            cols += 1

    return rows, cols
