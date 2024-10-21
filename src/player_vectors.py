import os
import math
import time
import consts
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF


def load_data(actions_grid_file_path: str, players_total_played_time_file_path: str, action_grid_shape: tuple[int, int]):
    df_actions_grid = pd.read_csv(actions_grid_file_path, converters={
        "player_id": int, "Shot": float, "Pass": float, "Cross": float, "Dribble": float,
        "grid_index": int, "end_grid_index": int})
    df_players_total_played_time = pd.read_csv(players_total_played_time_file_path)
    df_actions_grid = df_actions_grid.astype({"Pass": float, "Shot": float, "Dribble": float})
    # Remove players from the actions grid that are not found in the players total played time dataframe
    # See extract_players_played_time.py for the details why is this necessary.
    unique_player_ids = df_players_total_played_time["player_id"].unique()
    df_actions_grid = df_actions_grid.loc[df_actions_grid["player_id"].isin(unique_player_ids)]

    return df_actions_grid, df_players_total_played_time


def get_gaussian_kernel(kernel_radius=1):
    # Function implementation found at https://stackoverflow.com/questions/8204645/implementing-gaussian-blur-how-to-calculate-convolution-matrix-kernel
    gaussian = lambda x, mu, sigma: math.exp(-(((x - mu) / (sigma)) ** 2) / 2.0)

    sigma = kernel_radius / 2.  # for [-2*sigma, 2*sigma]

    # compute the actual kernel elements
    h_kernel = [gaussian(x, kernel_radius, sigma) for x in range(2 * kernel_radius + 1)]
    v_kernel = [x for x in h_kernel]
    kernel2d = [[xh * xv for xh in h_kernel] for xv in v_kernel]

    # normalize the kernel elements
    kernel_sum = sum([sum(row) for row in kernel2d])
    kernel2d = np.array([[x / kernel_sum for x in row] for row in kernel2d])

    return kernel2d


def smooth_action(df_player_action_grid: pd.DataFrame, action_type: str,
                  actions_grid_matrix_shape: tuple, smoothing_kernel: np.array,
                  grid_index_column = "grid_index") -> pd.DataFrame:
    max_tiles_num = actions_grid_matrix_shape[0] * actions_grid_matrix_shape[1]
    kernel_half_width = int(smoothing_kernel.shape[0]/2)
    kernel_half_height = int(smoothing_kernel.shape[1]/2)
    df = df_player_action_grid.loc[df_player_action_grid[action_type] > 0][[action_type, grid_index_column]]
    temp_df = pd.DataFrame(range(0, max_tiles_num), columns=[grid_index_column])

    temp_df[action_type] = 0.0
    df = df.merge(temp_df, how="right", on=[grid_index_column], suffixes=(None, "_r"))
    df.drop(columns=[f"{action_type}_r"], inplace=True)
    df[action_type] = df[action_type].fillna(0.0)
    df = df.groupby(by=grid_index_column).sum().reset_index()
    actions_grid_matrix = df[action_type].values.reshape(
        actions_grid_matrix_shape[0], actions_grid_matrix_shape[1])
    temp_matrix = np.zeros([actions_grid_matrix.shape[0] + kernel_half_width*2,
                           actions_grid_matrix.shape[1] + kernel_half_height*2])
    temp_matrix[kernel_half_width:temp_matrix.shape[0] - kernel_half_width,
                kernel_half_height:temp_matrix.shape[1] - kernel_half_height] = actions_grid_matrix
    smoothed_shots_matrix = np.zeros(actions_grid_matrix.shape)

    for i in range(kernel_half_width, temp_matrix.shape[0] - kernel_half_width):
        for j in range(kernel_half_height, temp_matrix.shape[1] - kernel_half_height):
            smoothed_shots_matrix[i - kernel_half_width, j - kernel_half_height] += (
                (temp_matrix[i - kernel_half_width:i + kernel_half_width + 1,
                 j - kernel_half_height:j + kernel_half_height + 1] * smoothing_kernel).sum())
    smoothed_data = np.reshape(smoothed_shots_matrix, max_tiles_num)

    #non_null_indices = np.where(smoothed_data > 0)
    #non_null_data = smoothed_data[non_null_indices]
    #df_smoothed_data = pd.DataFrame(zip(non_null_indices[0], non_null_data), columns=[grid_index_column, action_type])
    df_smoothed_data = pd.DataFrame(zip(range(0, max_tiles_num), smoothed_data),
                                    columns=[grid_index_column, action_type])
    df_smoothed_data = df_smoothed_data.astype(dtype=df.dtypes)

    return df_smoothed_data


def concat_actions_data(actions_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    max_tiles = consts.FOOTBALL_PITCH_TILES[0] * consts.FOOTBALL_PITCH_TILES[1]
    df_temp = pd.DataFrame(range(0, max_tiles), columns=["grid_index"], dtype=np.int64)
    df = None

    for action_type, data in actions_data.items():
        if df is None:
            df = data.merge(df_temp, how="right", on=["grid_index"])
        else:
            df[action_type] = data.merge(df_temp, how="right", on=["grid_index"])[action_type]
        df[action_type] = df[action_type].fillna(0.0)

    return df

def smooth_actions_data(df_actions_grid: pd.DataFrame, actions_grid_matrix_shape: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    kernel_matrix = get_gaussian_kernel()
    player_ids = df_actions_grid["player_id"].unique()
    df_actions_start_grid_smoothed = pd.DataFrame(columns=df_actions_grid.columns).astype(dtype=df_actions_grid.dtypes)
    df_actions_end_grid_smoothed = pd.DataFrame(columns=df_actions_grid.columns).astype(dtype=df_actions_grid.dtypes)
    df_actions_start_grid_smoothed.drop(columns=["end_grid_index", "player_name", "player_nickname", "country"],
                                        inplace=True)
    df_actions_end_grid_smoothed.drop(columns=["Shot", "Dribble", "player_name", "player_nickname",
                                               "country", "end_grid_index"],
                                      inplace=True)

    for player_id in player_ids:
        df_player_action_grid = df_actions_grid.loc[df_actions_grid["player_id"] == player_id]#.copy()
        start_smoothed_data_columns = {}
        end_smoothed_data_columns = {}
        for action_type in ["Shot", "Pass", "Cross", "Dribble"]:
            smoothed_data = smooth_action(df_player_action_grid, action_type,
                                          actions_grid_matrix_shape, kernel_matrix)
            start_smoothed_data_columns[action_type] = smoothed_data
            if action_type in ["Pass"]:
                end_smoothed_data_columns[action_type] = smooth_action(df_player_action_grid, action_type,
                                                                       actions_grid_matrix_shape, kernel_matrix,
                                                                       grid_index_column="end_grid_index")
                end_smoothed_data_columns[action_type] = end_smoothed_data_columns[action_type].rename(
                    columns={"end_grid_index": "grid_index"})

        concatenated_start_data = concat_actions_data(start_smoothed_data_columns)
        concatenated_end_data = concat_actions_data(end_smoothed_data_columns)
        concatenated_start_data["player_id"] = player_id
        concatenated_end_data["player_id"] = player_id
        df_actions_start_grid_smoothed = pd.concat([df_actions_start_grid_smoothed, concatenated_start_data])
        df_actions_end_grid_smoothed = pd.concat([df_actions_end_grid_smoothed, concatenated_end_data])

    return df_actions_start_grid_smoothed, df_actions_end_grid_smoothed

def compress_heatmap(df_actions_grid_start: pd.DataFrame, df_actions_grid_end: pd.DataFrame,
                     components_per_action: dict, max_iter: int=1000):
    actions_heatmap = {}
    action_types =  ["Shot", "Pass", "Cross", "Dribble"]
    players_count = len(df_actions_grid_start["player_id"].unique())

    df_start_action = df_actions_grid_start.groupby(by=["player_id", "grid_index"]).sum().reset_index()
    df_end_action = df_actions_grid_end.groupby(by=["player_id", "grid_index"]).sum().reset_index()

    for action_type in action_types:
        action_array = df_start_action[action_type].to_numpy().reshape(players_count, consts.MAX_TILES).T
        if action_type in ["Pass"]:
            end_action_array = df_end_action[action_type].to_numpy().reshape(players_count, consts.MAX_TILES).T
            action_array = np.hstack([action_array, end_action_array])
        actions_heatmap[action_type] = action_array

    compressed_actions_heatmap = {}
    for action_type, heatmap in actions_heatmap.items():
        model = NMF(n_components=components_per_action[action_type], init='random', random_state=0, max_iter=max_iter)
        W = model.fit_transform(heatmap)
        H = model.components_
        compressed_actions_heatmap[action_type] = (W, H)
        print(f"NMF reconstruction error for action '{action_type}: {model.reconstruction_err_}")

    return actions_heatmap, compressed_actions_heatmap


def save_heatmaps(action_heatmap: dict[str, np.array],
                  compressed_actions_heatmap: dict[str, np.array],
                  output_dir_path: str) -> None:
    for action_type, (W, H) in compressed_actions_heatmap.items():
        np.save(os.path.join(output_dir_path, f"compressed_heatmap_{action_type}_W"), W)
        np.save(os.path.join(output_dir_path, f"compressed_heatmap_{action_type}_H"), H)
    for action_type, heatmap in action_heatmap.items():
        np.save(os.path.join(output_dir_path, f"actions_heatmap_{action_type}"), heatmap)


if __name__ == "__main__":
    start_time = time.time()
    df_actions_grid, df_players_total_played_time = load_data(consts.ACTIONS_GRID_FILE_PATH,
                                                              consts.PLAYERS_TOTAL_PLAYED_TIME_FILE_PATH,
                                                              consts.ACTIONS_GRID_SHAPE)
    df_actions_grid_normalized = df_actions_grid.merge(df_players_total_played_time, on="player_id", how="inner", validate="m:1")
    df_actions_grid_normalized["Shot"] /= df_actions_grid_normalized["play_duration"]
    df_actions_grid_normalized["Pass"] /= df_actions_grid_normalized["play_duration"]
    df_actions_grid_normalized["Cross"] /= df_actions_grid_normalized["play_duration"]
    df_actions_grid_normalized["Dribble"] /= df_actions_grid_normalized["play_duration"]
    df_actions_grid_normalized.drop(columns=["play_duration"], inplace=True)
    df_actions_grid_normalized.to_csv(consts.ACTIONS_GRID_NORMALIZED_FILE_PATH, index=False)
    df_actions_start_grid_smoothed, df_actions_end_grid_smoothed = smooth_actions_data(
        df_actions_grid_normalized, actions_grid_matrix_shape=consts.ACTIONS_GRID_SHAPE)
    df_actions_start_grid_smoothed.to_csv(consts.ACTIONS_GRID_SMOOTHED_START_FILE_PATH, index=False)
    df_actions_end_grid_smoothed.to_csv(consts.ACTIONS_GRID_SMOOTHED_END_FILE_PATH, index=False)
    actions_heatmap, compressed_actions_heatmap = compress_heatmap(df_actions_start_grid_smoothed,
                                                                   df_actions_end_grid_smoothed,
                                                                   consts.COMPONENTS_PER_ACTION)
    save_heatmaps(actions_heatmap, compressed_actions_heatmap, consts.OUTPUT_DIR_PATH)
    end_time = time.time()
    print(f"Player vectors generator, elapsed time: {end_time - start_time}s")
