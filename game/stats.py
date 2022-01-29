

import numpy as np
from .replay import Replay


def total_scores(games):
    return sum(
        np.array(game.get_score())
        for game in games
    ).tolist()

