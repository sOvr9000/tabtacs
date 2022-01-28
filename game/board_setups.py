
import json
import numpy as np
from copy import deepcopy
from .enums import SoldierType
from .taclib import obstacles_to_str



def get_board_setup(name):
	'''
	Get a predefined board setup by its name.
	The returned setup is a copy of the predefined one, so it can be modified freely.
	'''
	if name not in board_setups:
		raise ValueError(f'Unrecognized board setup name: {name}')
	return copy_board_setup(board_setups[name])

def copy_board_setup(setup):
	'''
	Return a copy of the given board setup.
	'''
	return deepcopy(setup)

def set_board_setup(name, setup):
	'''
	Define a board setup under a given name to be accessible via get_board_setup(name).
	'''
	if name in board_setups:
		raise ValueError(f'A board setup already exists under the name \'{name}\'')
	board_setups[name] = setup

def random_obstacles(board_size, num_obstacles, symmetric = False):
	'''
	Randomly generate obstacles for a given board size.
	board_size is a tuple of the form (width, height).
	Obstacles are generated in such a way that a valid path (which a soldier can take) exists between any two non-obstructed positions on the board.
	'''
	w,h = board_size
	wh = w*h
	obstacles = [[False]*w for _ in range(h)]
	open_set = [(x,y) for y in range(h) for x in range(w)]
	total = 0

	def is_connected():
		for y in range(h):
			for x in range(w):
				if not obstacles[y][x]:
					arr = deepcopy(obstacles)
					flood_fill(x,y,arr)
					return sum(map(sum,arr)) == wh
	
	def flood_fill(x,y,arr):
		if not arr[y][x]:
			arr[y][x] = True
			if x+1 < w:
				flood_fill(x+1,y,arr)
			if x > 0:
				flood_fill(x-1,y,arr)
			if y+1 < h:
				flood_fill(x,y+1,arr)
			if y > 0:
				flood_fill(x,y-1,arr)

	def try_position(p):
		open_set.remove(p)
		obstacles[p[1]][p[0]] = True
		if not is_connected():
			obstacles[p[1]][p[0]] = False
			return False
		return True

	while len(open_set) > 0 and (total+2 < num_obstacles or not symmetric and total < num_obstacles):
		p = open_set[np.random.randint(len(open_set))]
		if not try_position(p):
			continue
		if symmetric:
			# TODO: refactor so as to not repeat any code
			rp = (board_size[0]-1-p[0], board_size[1]-1-p[1])
			if not try_position(rp):
				obstacles[p[1]][p[0]] = False
				continue
			total += 2
		else:
			total += 1

	return obstacles

def random_soldiers(soldiers_per_player, symmetric = False):
	'''
	Randomly generate a soldier configuration for each player.
	soldiers_per_player must be a list of integers each no less than one.
	If symmetric is True, then the same randomly generated configuration is used for all players.
	'''

	soldiers = []
	num_types = len(list(iter(SoldierType)))
	for num_soldiers in soldiers_per_player:
		v = np.abs(np.random.normal(size=num_types-1))
		v = (v * (num_soldiers - 1) / v.sum() + 0.5).astype(int)
		while v.sum() > num_soldiers:
			m = v.max()
			indices = [i for i,_v in enumerate(v) if _v == m]
			v[indices[np.random.randint(len(indices))]] -= 1
		_soldiers = {
			SoldierType(i+1): v
			for i,v in enumerate(v)
		}
		if symmetric:
			return [deepcopy(_soldiers) for _ in range(len(soldiers_per_player))]
		soldiers.append(_soldiers)
	return soldiers

def random_placement_space(obstacles, positions_per_player):
	'''
	Given a board's obstacles, randomly generate a placement space (a list of positions) for each player.
	
	positions_per_player must be a list of integers, each representing the number of positions available for a unique player.
	'''
	valid_positions = [(x,y) for y in range(len(obstacles)) for x in range(len(obstacles[0])) if not obstacles[y][x]]
	if len(valid_positions) < sum(positions_per_player):
		raise ValueError(f'The number of open positions in the given board is less than the total number of placement positions.\nBoard obstacles:\n{obstacles_to_str(obstacles)}\nPositions per player: {positions_per_player}')
	return [
		[
			valid_positions.pop(np.random.randint(len(valid_positions)))
			for _ in range(num_positions)
		]
		for num_positions in positions_per_player
	]

def random_board_setup(
	board_size,
	num_obstacles = 6,
	num_players = 2,
	positions_per_player = 4,
	symmetric_obstacles = False,
	soldiers_per_player = 4,
	symmetric_soldiers = False
):
	'''
	Randomly generate a board setup with the specified parameters.
	'''
	obstacles = random_obstacles(board_size, num_obstacles, symmetric=symmetric_obstacles)
	placement_space = random_placement_space(
		obstacles,
		[positions_per_player]*num_players
		if isinstance(positions_per_player, int) else
		positions_per_player
	)
	soldiers = random_soldiers(
		[soldiers_per_player]*num_players
		if isinstance(soldiers_per_player, int) else
		soldiers_per_player,
		symmetric=symmetric_soldiers
	)
	return {
		'board_size': board_size,
		'placement_space': placement_space,
		'soldiers': soldiers,
		'obstacles': obstacles,
	}

def fix_board_setup(setup):
	'''
	Saving and loading of setups with JSON does not retain the original objects in the setup, such as tuples inside placement_space lists.
	This function ensures that objects are of correct type, modifying the setup in place.
	'''
	for i, placement_space in enumerate(setup['placement_space']):
		setup['placement_space'][i] = list(map(tuple, placement_space))
	for i, soldiers in enumerate(setup['soldiers']):
		fixed_soldiers = {}
		for k, v in soldiers.items():
			fixed_soldiers[SoldierType(int(k))] = v
		setup['soldiers'][i] = fixed_soldiers

def save_board_setups(fpath):
	'''
	Save the currently defined board_setups to file.
	'''
	json.dump(board_setups,open(fpath,'w'))

def load_board_setups(fpath):
	'''
	Load board setups from file.
	'''
	global board_setups
	board_setups = json.load(open(fpath,'r'))
	map(fix_board_setup,board_setups)


board_setups = {
	'standard': {
		'board_size': (6,6),
		'placement_space': [
			[(0,0),(1,0),(0,1),(1,1)],
			[(5,5),(4,5),(5,4),(4,4)],
		],
		'soldiers': [
			{
				SoldierType.Fighter: 2,
				SoldierType.Thief: 1,
			},
			{
				SoldierType.Fighter: 2,
				SoldierType.Thief: 1,
			},
		]
	},
}

board_setups['tweaked'] = get_board_setup('standard')
board_setups['tweaked']['soldiers'][1][SoldierType.Fighter] = 1
board_setups['tweaked']['soldiers'][1][SoldierType.Thief] = 2

