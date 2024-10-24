import os
import time
import consts
import logging
import numpy as np
import pandas as pd


def get_action_vector(shot_heatmap: np.array, pass_heatmap: np.array, dribble_heatmap: np.array,
                      cross_heatmap: np.array, player_column_index: int, players_count: int) -> np.array:
    shot_vector = shot_heatmap[:, player_column_index]
    pass_vector = pass_heatmap[:, player_column_index]
    pass_end_vector = pass_heatmap[:, player_column_index + players_count]
    dribble_vector = dribble_heatmap[:, player_column_index]
    cross_vector = cross_heatmap[:, player_column_index]

    return np.concat([shot_vector, pass_vector, pass_end_vector, cross_vector, dribble_vector])


def get_manhattan_distance(a, b):
    return np.abs(a - b).sum()


def calculate_mahattan_distance(row, unique_player_ids, players_count, shot_action_heatmap_H,
                                pass_action_heatmap_H, dribble_action_heatmap_H,
                                cross_action_heatmap_H):
    row_index = np.where(unique_player_ids == row["player_id"])[0]
    player_vector = get_action_vector(shot_action_heatmap_H, pass_action_heatmap_H,
                                      dribble_action_heatmap_H, cross_action_heatmap_H,
                                      row_index, players_count)
    row_index2 = np.where(unique_player_ids == row["player_id_right"])[0]
    player2_vector = get_action_vector(shot_action_heatmap_H, pass_action_heatmap_H,
                                       dribble_action_heatmap_H, cross_action_heatmap_H,
                                       row_index2, players_count)
    manhattan_distance = get_manhattan_distance(player_vector, player2_vector)
    return manhattan_distance


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=consts.APP_LOG_PATH, level=logging.INFO)
    start_time = time.time()
    logger.info(f"Start time: {start_time}")

    df_squads = pd.read_csv(consts.SQUADS_FILE_PATH)
    df_players_total_played_time = pd.read_csv(consts.PLAYERS_TOTAL_PLAYED_TIME_FILE_PATH)
    shot_action_heatmap_H = np.load(os.path.join(consts.OUTPUT_DIR_PATH, f"compressed_heatmap_Shot_H.npy"))
    pass_action_heatmap_H = np.load(os.path.join(consts.OUTPUT_DIR_PATH, f"compressed_heatmap_Pass_H.npy"))
    dribble_action_heatmap_H = np.load(os.path.join(consts.OUTPUT_DIR_PATH, f"compressed_heatmap_Dribble_H.npy"))
    cross_action_heatmap_H = np.load(os.path.join(consts.OUTPUT_DIR_PATH, f"compressed_heatmap_Cross_H.npy"))
    actions_grid = pd.read_csv(os.path.join(consts.OUTPUT_DIR_PATH, "actions_grid.csv"))
    actions_grid_unique_player_ids = actions_grid["player_id"].unique()
    unique_player_ids = actions_grid["player_id"].unique()
    euro24_player_names = df_squads["Player"].unique()
    # Select only Euro 24 players and players that have recorded actions such as shots, passes and
    # dribbles in the actions grid.
    df_european_players = df_players_total_played_time.loc[
        (df_players_total_played_time["player_name"].isin(euro24_player_names) |
         df_players_total_played_time["player_nickname"].isin(euro24_player_names)) &
         df_players_total_played_time["player_id"].isin(actions_grid_unique_player_ids)]
    df_manhattan_distance = df_european_players[["player_id"]].merge(df_european_players[["player_id"]], how="cross",
                                                      suffixes=(None, "_right"))
    df_manhattan_distance = df_manhattan_distance.loc[df_manhattan_distance["player_id"] != df_manhattan_distance["player_id_right"]]
    df_manhattan_distance["distance"] = df_manhattan_distance.apply(calculate_mahattan_distance, axis=1, args=(
        unique_player_ids, len(unique_player_ids), shot_action_heatmap_H, pass_action_heatmap_H,
        dribble_action_heatmap_H, cross_action_heatmap_H))
    # Remove rows where Manhattan distance is 0.0 because it is a strong indication that there are not enough
    # data stored in the actions grid.
    df_manhattan_distance = df_manhattan_distance.loc[df_manhattan_distance["distance"] != 0.0]
    df_manhattan_distance.to_csv(consts.PLAYER_STYLE_MANHATTAN_DISTANCE_FILE_PATH, index=False)

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time}")