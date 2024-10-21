import os
import consts
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score


if __name__ == "__main__":
    for action_type in ["Shot", "Dribble", "Pass", "Cross"]:
        action_heatmap = np.load(os.path.join(consts.OUTPUT_DIR_PATH, f"actions_heatmap_{action_type}.npy"))
        action_heatmap = action_heatmap.reshape(-1)
        players_count = len(action_heatmap)//consts.MAX_TILES
        action_heatmap_H = np.load(os.path.join(consts.OUTPUT_DIR_PATH, f"compressed_heatmap_{action_type}_H.npy"))
        action_heatmap_W = np.load(os.path.join(consts.OUTPUT_DIR_PATH, f"compressed_heatmap_{action_type}_W.npy"))
        if action_type in ["Pass"]:
            action_heatmap_H = action_heatmap_H[:, :players_count]
        reconstructed_action_heatmap = np.matmul(action_heatmap_W, action_heatmap_H).reshape(-1)
        score = explained_variance_score(action_heatmap, reconstructed_action_heatmap)
        mse = mean_squared_error(action_heatmap, reconstructed_action_heatmap)

        print(f"Explained variance score for action type '{action_type} is {score}.")
        print(f"MSE for action type '{action_type} is {mse}.")

