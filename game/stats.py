

import numpy as np

from .tabletactics import TableTactics
from .replay import Replay
from .taclib import get_adjacent_position


def total_scores(games):
	return sum(
		np.array(game.get_score())
		for game in games
	).tolist()


def replay_stats(replays):
	'''
	Compute some statistics on a replay or a list of replays.  Assumes exactly two armies exist in each replay.
	'''
	if isinstance(replays, Replay):
		replays = [replays]
	victim_frequency = [{}, {}] # For each army, what do its soldiers attack?
	attacker_frequency = [{}, {}] # For each army, which soldiers attack?
	for replay in replays:
		for game, (func, args) in replay.step():
			if func is not None:
				if func.__name__ == 'attack_soldier':
					x, y, d = args
					nx, ny = get_adjacent_position(x, y, d)
					attacker_army = board[y,x,TableTactics.ARMY_INDEX]
					attacker_type = board[y,x,TableTactics.SOLDIER_TYPE_INDEX]
					if attacker_type not in attacker_frequency[attacker_army]:
						attacker_frequency[attacker_army][attacker_type] = 1
					else:
						attacker_frequency[attacker_army][attacker_type] += 1
					victim_army = board[ny,nx,TableTactics.ARMY_INDEX]
					victim_type = board[ny,nx,TableTactics.SOLDIER_TYPE_INDEX]
					if victim_type not in victim_frequency[victim_army]:
						victim_frequency[victim_army][victim_type] = 1
					else:
						victim_frequency[victim_army][victim_type] += 1
			board = game.board.copy()
	return {
		''
	}
