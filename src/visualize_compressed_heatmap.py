import os
import consts
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pitch_utils import (
    create_pitch,
    create_pitch_grid,
    plot_pitch_data,
    get_pitch_grid_dim
)

MAX_PITCHES_TO_DISPLAY = 10


def get_script_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", help="Target player ID", required=True,
                        nargs=1, type=int)
    parser.add_argument("-a", "--action", help="Type of player's action (Shot|Dribble|Pass|Cross)",
                        required=True, nargs=1, type=str)
    parser.add_argument("-p", "--pitches", help="Maximum number of pitches to display",
                        nargs="?", type=int, default=MAX_PITCHES_TO_DISPLAY)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_script_args()
    action_heatmap_W = np.load(os.path.join(consts.OUTPUT_DIR_PATH, f"compressed_heatmap_{args.action[0]}_W.npy"))
    action_heatmap_H = np.load(os.path.join(consts.OUTPUT_DIR_PATH, f"compressed_heatmap_{args.action[0]}_H.npy"))
    actions_grid_smoothed = pd.read_csv(consts.ACTIONS_GRID_SMOOTHED_START_FILE_PATH)
    player_ids = actions_grid_smoothed["player_id"].unique()
    player_col_index = np.where(player_ids == args.target[0])[0][0]
    total_features_num = action_heatmap_W.shape[1]
    # When determining the index of the column representing the dominant feature, only the data about the starting
    # tile of the action is considered. The "H" matrix for the action types containing information about the position
    # where the action ended will be doubled in size regarding the number of columns.
    dominant_feature_index = np.argsort(action_heatmap_H[:, player_col_index])[::-1][0]
    pitch_grid_dim = get_pitch_grid_dim(args.pitches, total_features_num)

    pitch = create_pitch()
    fig, axs = create_pitch_grid(pitch, pitch_grid_dim[0], pitch_grid_dim[1])

    for idx, ax in enumerate(axs['pitch'].flat):
        if idx >= total_features_num:
            break
        data = action_heatmap_W[:, idx].reshape(consts.ACTIONS_GRID_SHAPE)
        title = f"feature_{idx+1}" + (" - Dominant" if idx == dominant_feature_index else "")
        plot_pitch_data(pitch, data, ax, consts.FOOTBALL_PITCH_SIZE, title)

    plt.show()
