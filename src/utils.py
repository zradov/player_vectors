import pandas as pd


"""
def expand_player_data(df_player_actions: pd.DataFrame, action_type: str, target_matrix_shape: tuple, grid_index_column: str):
    df_temp = df_player_actions[[action_type, grid_index_column]].drop_duplicates(subset=grid_index_column, ignore_index=True)
    max_tiles = target_matrix_shape[0] * target_matrix_shape[1]
    df_grids = pd.DataFrame(range(max_tiles), columns=[grid_index_column])
    df_temp = df_temp.merge(df_grids, how="right", on=[grid_index_column])
    player_id = df_player_actions["player_id"].iloc[0]
    df_temp["player_id"] = player_id
    df_temp[action_type] = df_temp[action_type].fillna(0)

    return df_temp
"""