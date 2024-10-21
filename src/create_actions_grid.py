import os
import time
import queue
import consts
import logging
import argparse
import pandas as pd
import numpy as np
import multiprocessing as mp
from statsbombpy_local import sb
from concurrent.futures import ProcessPoolExecutor

pd.options.mode.copy_on_write = True
logger = logging.getLogger(__name__)
STATSBOMB_OPEN_DATA_LOCAL_PATH = consts.STATSBOMB_OPEN_DATA_LOCAL_PATH


def load_events_worker(matches_queue: queue.Queue, events_queue: queue.Queue):
    df_all_events = pd.DataFrame(np.array([], dtype=[
        ("player_id", np.int32), ("type", str), ("pass_cross", bool),
        ("location", object), ("pass_end_location", object)]),
        columns=["player_id", "type", "pass_cross", "location", "pass_end_location"])
    matches_ids = matches_queue.get()

    for match_id in matches_ids:
        print(f"Loading events data for match with id: {match_id}")
        df_events = sb.events(match_id=match_id)
        # Filter out starting events such as kick off event.
        df_events = df_events.loc[df_events["player_id"].notna()]
        df_events["player_id"] = df_events["player_id"].astype(int)
        if df_all_events is None:
            df_all_events = df_events[["player_id", "type", "pass_cross", "location", "pass_end_location"]]
        else:
            df_all_events = pd.concat([df_all_events, df_events[["player_id", "type", "pass_cross", "location", "pass_end_location"]]])

    events_queue.put(df_all_events)
    matches_queue.task_done()


def load_events_worker_single_proc(matches_ids):
    df_all_events = pd.DataFrame(np.array([], dtype=[
        ("player_id", np.int32), ("type", str), ("pass_cross", bool),
        ("location", object), ("pass_end_location", object)]),
         columns=["player_id", "type", "pass_cross", "location", "pass_end_location"])
    for match_id in matches_ids:
        print(f"Loading events data for match with id: {match_id}")
        df_events = sb.events(match_id=match_id)
        # Filter out starting events such as kick off event.
        df_events = df_events.loc[df_events["player_id"].notna()]
        df_events["player_id"] = df_events["player_id"].astype(int)
        if df_all_events is None:
            df_all_events = df_events[["player_id", "type", "pass_cross", "location", "pass_end_location"]]
        else:
            df_all_events = pd.concat([df_all_events, df_events[
                ["player_id", "type", "pass_cross", "location", "pass_end_location"]]])

    return df_all_events


def load_events(matches_ids: list[int], parallel_processes_count=os.cpu_count()):
    logger.debug(f"matches count: {len(matches_ids)}")
    matches_slice_size = round(len(matches_ids)/parallel_processes_count)
    logger.debug(f"matches_slice_size: {matches_slice_size}")

    with mp.Manager() as manager:
        events_queue = manager.Queue()
        matches_queue = manager.Queue()
        for i in range(parallel_processes_count):
            start = i * matches_slice_size
            end = (i + 1) * matches_slice_size
            logger.debug(f"Start slice: {start}, end slice: {end}")
            matches_queue.put(matches_ids[start:end])
        logger.debug("Starting with events loading ...")
        with ProcessPoolExecutor(max_workers=parallel_processes_count) as executor:
            executor.map(load_events_worker, [matches_queue]*parallel_processes_count, [events_queue]*parallel_processes_count)
        matches_queue.join()

        df_all_events = None
        while not events_queue.empty():
            df_events = events_queue.get()
            if df_all_events is None:
                df_all_events = df_events
            else:
                df_all_events = pd.concat([df_all_events, df_events])

    return df_all_events


def load_events_single_proc(matches_ids: list[int]):
    logger.debug(f"matches count: {len(matches_ids)}")
    matches_slice_size = round(len(matches_ids))
    logger.debug(f"matches_slice_size: {matches_slice_size}")

    df_all_events = load_events_worker_single_proc(matches_ids)

    return df_all_events


def get_matches_ids(data_path: str, max_events=-1):
    matches_ids = []

    for idx, data_file in enumerate(os.listdir(data_path)):
        match_id = int(data_file.split(".")[0])
        matches_ids.append(match_id)
        if max_events != -1 and idx+1 == max_events:
            break

    return matches_ids


def get_actions(df_events: pd.DataFrame, max_x_pos: int, max_y_pos: int):
    action_types_filter = ((df_events.type == "Dribble") | (df_events.type == "Shot") |
                           (df_events.type == "Pass"))
    df_actions = df_events.loc[action_types_filter].reset_index()
    # For the "Shot" and "Dribble" actions the index of the tile where the action ended is not
    # important, so we set it to 0 anyway.
    df_actions[["grid_index", "end_grid_index"]] = 0
    df_actions[np.array(df_actions.type.unique())] = 0
    df_actions["Cross"] = 0
    df_actions[["loc_x", "loc_y"]] = df_actions["location"].tolist()
    # By default, for the actions that do not have coordinates for the position where certain action ended
    # the position will be set to [0, 0].
    df_actions["end_loc"] = df_actions["pass_end_location"].apply(lambda v: v if isinstance(v, list) else [0, 0])
    df_actions[["end_loc_x", "end_loc_y"]] = df_actions["end_loc"].tolist()
    df_actions.drop(columns=["end_loc"], inplace=True)
    df_actions.loc[df_actions["loc_x"] > max_x_pos, "loc_x"] = max_x_pos
    df_actions.loc[df_actions["loc_y"] > max_y_pos, "loc_y"] = max_y_pos
    df_actions.loc[df_actions["end_loc_x"] > max_x_pos, "end_loc_x"] = max_x_pos
    df_actions.loc[df_actions["end_loc_y"] > max_y_pos, "end_loc_y"] = max_y_pos

    return df_actions


def create_actions_grid(df_actions: pd.DataFrame, football_pitch_tiles, heatmap_tile_size):
    df_actions["grid_index"] = ((df_actions["loc_x"] / heatmap_tile_size[0]).astype(int) +
                                (df_actions["loc_y"] / heatmap_tile_size[1]).astype(int) * football_pitch_tiles[0])
    df_actions.loc[(df_actions["end_loc_x"] > 0) | (df_actions["end_loc_y"] > 0), "end_grid_index"] = \
        ((df_actions["end_loc_x"] / heatmap_tile_size[0]).astype(int) +
         (df_actions["end_loc_y"] / heatmap_tile_size[1]).astype(int) * football_pitch_tiles[0])
    df_actions.loc[df_actions["type"] == "Shot", "Shot"] = 1
    df_actions.loc[df_actions["type"] == "Pass", "Pass"] = 1
    df_actions.loc[df_actions["type"] == "Dribble", "Dribble"] = 1
    df_actions.loc[((df_actions["type"] == "Pass") & df_actions["pass_cross"]), "Cross"] = 1

    df_actions_grid = df_actions[["player_id", "Shot", "Pass", "Dribble", "Cross", "grid_index", "end_grid_index"]]
    df_actions_grid = df_actions_grid.groupby(["player_id", "grid_index", "end_grid_index"], as_index=False).sum()

    return df_actions_grid


def get_script_args():
    parser = argparse.ArgumentParser();
    parser.add_argument("-p", "--processes", help="Max parallel processes when processing matches events",
                        nargs="?", default=os.cpu_count(), type=int)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_script_args()
    logging.basicConfig(filename=consts.APP_LOG_PATH, level=logging.DEBUG)
    matches_ids = get_matches_ids(os.path.join(STATSBOMB_OPEN_DATA_LOCAL_PATH, "data", "events"))
    start_time = time.time()
    df_events = load_events(matches_ids, parallel_processes_count=args.processes)
    #df_events = load_events_single_proc(matches_ids)
    end_time = time.time()
    print(f"Load events elapsed time: {end_time-start_time}s")
    df_actions = get_actions(df_events, consts.FOOTBALL_PITCH_SIZE[0]-1, consts.FOOTBALL_PITCH_SIZE[1]-1)
    df_actions_grid = create_actions_grid(df_actions, consts.FOOTBALL_PITCH_TILES, consts.HEATMAP_TILE_SIZE)
    df_actions_grid.to_csv(consts.ACTIONS_GRID_FILE_PATH, index=False)