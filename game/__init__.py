from .prototypes import load_soldier_prototypes
from .enums import SoldierType
from .tabletactics import TableTactics
from .random_sim import RandomSim
from .replay import Replay, load_replays, count_replays, final_states
from .taclib import *
from .board_setups import get_board_setup, copy_board_setup, set_board_setup, save_board_setups, load_board_setups, fix_board_setup, \
    random_obstacles, random_soldiers, random_placement_space, random_board_setup
from .stats import total_scores
