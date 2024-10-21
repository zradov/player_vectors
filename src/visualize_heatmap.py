import os
import consts
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pitch_utils import plot_pitch_data, create_pitch, create_pitch_grid


def get_script_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--player", help="Player ID", required=True,
                        nargs=1, type=int)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_script_args()
    action_types = [ "Shot", "Dribble", "Pass", "Cross" ]
    pitch = create_pitch()
    fig, axs = create_pitch_grid(pitch, 2, 4)
    actions_grid_smoothed = pd.read_csv(consts.ACTIONS_GRID_SMOOTHED_START_FILE_PATH)
    players_count = len(actions_grid_smoothed["player_id"].unique())

    for idx, ax in enumerate(axs['pitch'].flat):
        action_type = action_types[idx // 2]
        # Only indices of grid tiles where the action started are considered when visualizing heatmaps.
        if idx % 2 == 0:
            action_heatmap = np.load(os.path.join(consts.OUTPUT_DIR_PATH, f"actions_heatmap_{action_type}.npy"))
            action_heatmap_H = np.load(os.path.join(consts.OUTPUT_DIR_PATH, f"compressed_heatmap_{action_type}_H.npy"))
            action_heatmap_W = np.load(os.path.join(consts.OUTPUT_DIR_PATH, f"compressed_heatmap_{action_type}_W.npy"))
            reconstructed_action_heatmap = np.matmul(action_heatmap_W, action_heatmap_H)
            heatmap_column_index = np.where(actions_grid_smoothed["player_id"].unique() == args.player[0])[0]
            player_action_heatmap = reconstructed_action_heatmap[:, heatmap_column_index]#action_heatmap[:, heatmap_column_index]
            title = f"{action_type} Compressed Heatmap"
        else:
            player_action_heatmap = actions_grid_smoothed.loc[
                actions_grid_smoothed["player_id"] == args.player[0]][action_type].to_numpy()
            title = f"{action_type} Heatmap"
        player_action_heatmap = player_action_heatmap.reshape(consts.FOOTBALL_PITCH_TILES)
        plot_pitch_data(pitch, player_action_heatmap, ax, consts.FOOTBALL_PITCH_SIZE, title)

    plt.show()
