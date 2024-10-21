import consts
import argparse
import pandas as pd


# Threshold for Pearson's correlation coefficient.
MIN_R_COEF = 0.30


def get_script_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", help="Target player ID", required=True,
                        nargs=1, type=int)
    parser.add_argument("-r", "--coef", nargs='?', default=MIN_R_COEF,
                        help="Minimum R coefficient", type=float)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_script_args()
    file_path = consts.PEARSON_CORRELATION_COEFS_FILE_PATH_FORMAT.format(args.target[0])
    df = pd.read_csv(file_path)
    df["target_player_id"] = df["target_player_id"].astype(int)
    df["player_id"] = df["player_id"].astype(int)
    df_players_total_played_time = pd.read_csv(consts.PLAYERS_TOTAL_PLAYED_TIME_FILE_PATH)
    df_most_correlated_players = df.loc[(df["shot_corr"] > args.coef) & (df["cross_corr"] > args.coef) &
                                        (df["pass_corr"] > args.coef) & (df["dribble_corr"] > args.coef)]
    df_closest_players_regarding_manhattan_distance = df.sort_values(by=["manhattan_distance"])

    print(f"There are {len(df_most_correlated_players)} with Pearson correlation coefficient " +
          f"greater than {args.coef} for each action.")
    df_most_correlated_players = df_most_correlated_players.sort_values(by=[
        "shot_corr", "cross_corr", "pass_corr", "dribble_corr"], ascending=False, ignore_index=True)
    print(f"Top 5 player with the highest Pearson correlation coefficient")
    for index, player_id in enumerate(df_most_correlated_players["player_id"][:5]):
        player_name = df_players_total_played_time.loc[
            df_players_total_played_time["player_id"] == player_id]["player_name"].to_list()[0]
        print(f"{index+1}. {player_name}")
    print("\nClosest players regarding the Manhattan distance metric:")
    for index, player_id in enumerate(df_closest_players_regarding_manhattan_distance["player_id"][:5]):
        player_name = df_players_total_played_time.loc[
            df_players_total_played_time["player_id"] == player_id]["player_name"].to_list()[0]
        print(f"{index + 1}. {player_name}")