import os
import queue
import consts
import logging
import numpy as np
import pandas as pd
from time import time
import multiprocessing as mp
from statsbombpy_local import sb
from concurrent.futures import ProcessPoolExecutor


pd.options.mode.copy_on_write = True
logger = logging.getLogger(__name__)


def get_matches_ids(data_path: str, max_events=-1):
    matches_ids = []

    for idx, data_file in enumerate(os.listdir(data_path)):
        match_id = int(data_file.split(".")[0])
        matches_ids.append(match_id)
        if max_events != -1 and idx+1 == max_events:
            break

    return matches_ids


def calculate_player_total_play_duration(row):
    total_play_time_in_seconds = 0

    for pos in row["positions"]:
        minutes_from, seconds_from = pos["from"].split(":")
        minutes_to, seconds_to = (pos["to"] if pos["to"] else "90:00").split(":")
        total_seconds_from = int(minutes_from) * 60 + int(seconds_from)
        total_seconds_to = int(minutes_to) * 60 + int(seconds_to)
        total_play_time_in_seconds += total_seconds_to - total_seconds_from

    total_play_time_in_minutes = round(total_play_time_in_seconds / 60, 2)

    return total_play_time_in_minutes


def find_player_played_duration(matches_queue: queue.Queue,
                                    players_total_play_duration_queue: queue.Queue):
    df_all_players_total_play_duration = pd.DataFrame(
        np.array([], dtype=[("match_id", np.int32), ("player_id", np.int32), ("player_name", str),
                            ("player_nickname", str), ("country", str), ("play_duration", np.int32)]),
        columns=["match_id", "player_id", "player_name", "player_nickname", "country", "play_duration"])
    matches_ids = matches_queue.get()

    for match_id in matches_ids:
        match_lineups = sb.lineups(match_id=match_id)
        for team_name, df_team_lineup in match_lineups.items():
            df_team_lineup["play_duration"] = df_team_lineup.apply(calculate_player_total_play_duration, axis=1)
            df_team_lineup["match_id"] = match_id
            df_all_players_total_play_duration = pd.concat([df_all_players_total_play_duration,
                                                            df_team_lineup[["match_id", "player_id", "player_name",
                                                                            "player_nickname", "country", "play_duration"]]],
                                                           ignore_index=True)

    players_total_play_duration_queue.put(df_all_players_total_play_duration)
    matches_queue.task_done()


def get_players_played_time(matches_ids: list[int], parallel_processes_count=os.cpu_count()):
    matches_slice_size = round(len(matches_ids) / parallel_processes_count)
    df_all_players_total_play = None
    logger.debug(f"matches_slice_size: {matches_slice_size}")

    with mp.Manager() as manager:
        players_total_play_duration_queue = manager.Queue()
        matches_queue = manager.Queue()
        for i in range(parallel_processes_count):
            start = i * matches_slice_size
            end = (i + 1) * matches_slice_size
            logger.debug(f"Start slice: {start}, end slice: {end}")
            matches_queue.put(matches_ids[start:end])
        with ProcessPoolExecutor(max_workers=parallel_processes_count) as executor:
            executor.map(find_player_played_duration,
                         [matches_queue] * parallel_processes_count,
                         [players_total_play_duration_queue] * parallel_processes_count)
        matches_queue.join()

        while not players_total_play_duration_queue.empty():
            df_players_total_play_duration = players_total_play_duration_queue.get()
            if df_all_players_total_play is None:
                df_all_players_total_play = df_players_total_play_duration
            else:
                df_all_players_total_play = pd.concat([df_all_players_total_play,
                                                       df_players_total_play_duration])

    return df_all_players_total_play


if __name__ == "__main__":
    logging.basicConfig(filename=consts.APP_LOG_PATH, level=logging.DEBUG)
    start_time = time()
    matches_ids = get_matches_ids(os.path.join(consts.STATSBOMB_OPEN_DATA_LOCAL_PATH, "data", "events"))
    df_players_played_time = get_players_played_time(matches_ids)
    # Remove rows with negative play_duration values.
    # Negative play_duration value means that the player entered the match as a substitute in the overtime.
    # This is done because there are no data fields in the dataset containing exact values for the match duration.
    df_players_played_time = df_players_played_time[df_players_played_time["play_duration"] > 0]
    df_players_played_time["player_nickname"] = df_players_played_time["player_nickname"].fillna("NoNickname")
    df_players_played_time.to_csv(consts.PLAYERS_PLAYED_TIME_FILE_PATH, index=False)
    df_all_players_total_played_time = (df_players_played_time[[
        "player_id", "player_name","player_nickname", "country", "play_duration"]].groupby([
        "player_id","player_name","player_nickname", "country"]).sum().reset_index())
    # Remove all rows that have the duplicate player_id values.
    # This occurs when the same player has a different nickname and country in the dataset.
    duplicates = df_all_players_total_played_time.duplicated(subset="player_id", keep=False)
    df_unique_players_total_played_time = df_all_players_total_played_time[~duplicates]
    df_unique_players_total_played_time.to_csv(consts.PLAYERS_TOTAL_PLAYED_TIME_FILE_PATH, index=False)
    logger.info(f"Elapsed time: {time() - start_time}s")