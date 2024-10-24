import consts
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def get_script_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--player", help="Player ID", required=True,
                        nargs=1, type=int)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_script_args()
    df_manhattan_distances = pd.read_csv(consts.PLAYER_STYLE_MANHATTAN_DISTANCE_FILE_PATH)
    df_players_total_played_time = pd.read_csv(consts.PLAYERS_TOTAL_PLAYED_TIME_FILE_PATH)
    df_similiar_players = df_manhattan_distances.loc[
        df_manhattan_distances["player_id"] == args.player[0]][["player_id", "player_id_right", "distance"]]
    df_similiar_players = df_similiar_players.merge(df_players_total_played_time, on="player_id", how="left")
    df_similiar_players = df_similiar_players.merge(df_players_total_played_time, left_on="player_id_right",
                                                    right_on="player_id", how="left", suffixes=(None, "_right"))
    df_similiar_players.sort_values(by=["distance"], inplace=True)
    df_top_similar_players = df_similiar_players.head(10)
    print(df_top_similar_players[["player_name", "player_name_right", "distance"]])
    distances = df_top_similar_players["distance"].to_list()
    player_names = df_top_similar_players["player_name_right"].to_list()

    plt.pie(distances, labels=player_names)
    plt.legend(title="Player style similarity")
    plt.show()