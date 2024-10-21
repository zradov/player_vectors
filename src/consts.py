import os
from pathlib import Path


SQUADS_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "Datasets", "euro2024_squads.csv")
OUTPUT_DIR_PATH = os.path.join(os.path.dirname(__file__), "..", "output")
APP_LOG_PATH = os.path.join(OUTPUT_DIR_PATH, "app.log")
PLAYER_STYLE_MANHATTAN_DISTANCE_FILE_PATH = os.path.join(OUTPUT_DIR_PATH, "manhattan_distance.csv")
ACTIONS_GRID_FILE_PATH = os.path.join(OUTPUT_DIR_PATH, "actions_grid.csv")
PLAYERS_PLAYED_TIME_FILE_PATH = os.path.join(OUTPUT_DIR_PATH, "players_played_time.csv")
PLAYERS_TOTAL_PLAYED_TIME_FILE_PATH = os.path.join(OUTPUT_DIR_PATH, "players_total_played_time.csv")
PEARSON_CORRELATION_COEFS_FILE_PATH_FORMAT = os.path.join(OUTPUT_DIR_PATH, "{0}_correlation_coefficients.csv")
ACTIONS_GRID_NORMALIZED_FILE_PATH = os.path.join(OUTPUT_DIR_PATH, "actions_grid_normalized.csv")
ACTIONS_GRID_SMOOTHED_START_FILE_PATH = os.path.join(OUTPUT_DIR_PATH, "actions_grid_smoothed_start.csv")
ACTIONS_GRID_SMOOTHED_END_FILE_PATH = os.path.join(OUTPUT_DIR_PATH, "actions_grid_smoothed_end.csv")
START_ACTIONS_GRID_SMOOTHED_FILE_PATH = os.path.join(OUTPUT_DIR_PATH, "start_actions_grid_smoothed.csv")
END_ACTIONS_GRID_SMOOTHED_FILE_PATH = os.path.join(OUTPUT_DIR_PATH, "end_actions_grid_smoothed.csv")
STATSBOMB_OPEN_DATA_LOCAL_PATH = os.environ["OPEN_DATA_REPO_PATH"]
FOOTBALL_PITCH_SIZE = (120, 80)
HEATMAP_TILE_SIZE = (6, 4)
FOOTBALL_PITCH_TILES = (FOOTBALL_PITCH_SIZE[0]//HEATMAP_TILE_SIZE[0], FOOTBALL_PITCH_SIZE[1]//HEATMAP_TILE_SIZE[1])
MAX_TILES = FOOTBALL_PITCH_TILES[0] * FOOTBALL_PITCH_TILES[1]
ACTIONS_GRID_SHAPE = (int(FOOTBALL_PITCH_SIZE[0]/HEATMAP_TILE_SIZE[0]), int(FOOTBALL_PITCH_SIZE[1]/HEATMAP_TILE_SIZE[1]))
COMPONENTS_PER_ACTION = { "Shot": 6, "Pass": 15, "Cross": 4, "Dribble": 15 }


Path(OUTPUT_DIR_PATH).mkdir(parents=True, exist_ok=True)