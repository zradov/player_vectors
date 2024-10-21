import consts
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_selection import r_regression


def load_data(actions_grid_file_path: str, total_played_time_file_path: str) -> pd.DataFrame:

    df_actions_grid = pd.read_csv(actions_grid_file_path, converters={
        "player_id": int, "Shot": float, "Pass": float, "Cross": float, "Dribble": float,
        "grid_index": int, "end_grid_index": int})
    df_players_total_played_time = pd.read_csv(total_played_time_file_path)
    df_actions_grid = df_actions_grid.astype({"Pass": float, "Shot": float, "Dribble": float})
    # Remove players from the actions grid that are not found in the players total played time dataframe
    # See extract_players_played_time.py for the details why is this necessary.
    unique_player_ids = df_players_total_played_time["player_id"].unique()
    df_actions_grid = df_actions_grid.loc[df_actions_grid["player_id"].isin(unique_player_ids)]
    df_actions_grid = df_actions_grid.merge(df_players_total_played_time, on=["player_id"])
    df_actions_grid["Shot"] /= df_actions_grid["play_duration"]
    df_actions_grid["Pass"] /= df_actions_grid["play_duration"]
    df_actions_grid["Cross"] /= df_actions_grid["play_duration"]
    df_actions_grid["Dribble"] /= df_actions_grid["play_duration"]

    return df_actions_grid


def get_player_actions(actions_grid: pd.DataFrame, player_id: int, max_tiles=consts.MAX_TILES) -> pd.DataFrame:
    df_start_grid = pd.DataFrame(range(0, max_tiles), columns=["grid_index"])
    df_end_grid = pd.DataFrame(range(0, max_tiles), columns=["end_grid_index"])
    df_player_actions = actions_grid.loc[actions_grid["player_id"] == player_id][[
        "player_id", "Shot", "Pass", "Cross", "Dribble", "grid_index", "end_grid_index"]]
    df_player_end_actions = df_player_actions[["player_id", "Pass", "end_grid_index"]]
    df_player_actions.drop(columns=["end_grid_index"], inplace=True)
    df_player_actions = df_player_actions.merge(df_start_grid, how="right")
    df_player_end_actions = df_player_end_actions.merge(df_end_grid, how="right")
    df_player_actions["player_id"] = player_id
    df_player_end_actions["player_id"] = player_id
    df_player_actions[["Shot", "Pass", "Cross", "Dribble"]] = (
        df_player_actions[["Shot", "Pass", "Cross", "Dribble"]].fillna(0))
    df_player_end_actions["Pass"] = df_player_end_actions["Pass"].fillna(0)
    df_player_actions = df_player_actions.groupby(by=["player_id", "grid_index"]).sum().reset_index()
    df_player_end_actions = df_player_end_actions.groupby(by=["player_id", "end_grid_index"]).sum().reset_index()
    df_player_actions["Pass_end"] = df_player_end_actions["Pass"]
    df_player_actions["end_grid_index"] = df_player_end_actions["end_grid_index"]

    return df_player_actions


def get_manhattan_distance(a, b):
    return np.abs(a - b).sum()


def get_script_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", help="Target player ID",
                        required=True, nargs=1, type=int)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_script_args()
    target_player_id = args.target[0]
    df_actions_grid = load_data(consts.ACTIONS_GRID_FILE_PATH, consts.PLAYERS_TOTAL_PLAYED_TIME_FILE_PATH)
    corr_coefs_matrix = None
    df_target_player_actions = get_player_actions(df_actions_grid, player_id=target_player_id)
    other_players_ids = df_actions_grid.loc[df_actions_grid["player_id"] != target_player_id]["player_id"].unique()
    target_player_actions_vector = [1, 1, 1, 1]
    target_player_actions = {
        "Cross": df_target_player_actions["Cross"].to_numpy(),
        "Pass": np.append(df_target_player_actions["Pass"].to_numpy(), df_target_player_actions["Pass_end"].to_numpy()),
        "Dribble": df_target_player_actions["Dribble"].to_numpy(),
        "Shot": df_target_player_actions["Shot"].to_numpy(),
    }

    for player_id in other_players_ids:
        df_other_player_actions = get_player_actions(df_actions_grid, player_id=player_id)
        corr_coefs_arr = [target_player_id, player_id]
        for action_type in ["Pass", "Cross", "Shot", "Dribble"]:
            action_array = df_other_player_actions[action_type].to_numpy()
            if action_type in ["Pass"]:
                action_array = np.append(action_array, df_other_player_actions["Pass_end"].to_numpy())
            action_array = action_array.reshape((action_array.shape[0], 1))
            corr_coef = r_regression(action_array, target_player_actions[action_type])[0]
            corr_coefs_arr.append(corr_coef)
        manhattan_distance = get_manhattan_distance(np.array(corr_coefs_arr[2:]), np.array(target_player_actions_vector))
        manhattan_distance = manhattan_distance.item()
        corr_coefs_arr.append(manhattan_distance)
        if corr_coefs_matrix is None:
            corr_coefs_matrix = corr_coefs_arr
        else:
            corr_coefs_matrix = np.vstack([corr_coefs_matrix, corr_coefs_arr])

    df_pearson_rs = pd.DataFrame(corr_coefs_matrix, columns=["target_player_id", "player_id", "pass_corr", "cross_corr", "shot_corr", "dribble_corr", "manhattan_distance" ])
    df_pearson_rs = df_pearson_rs.astype({ "target_player_id": int, "player_id": int })
    output_file_path = consts.PEARSON_CORRELATION_COEFS_FILE_PATH_FORMAT.format(str(target_player_id))
    df_pearson_rs.to_csv(output_file_path, index=False)






