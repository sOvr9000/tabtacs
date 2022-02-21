
import numpy as np
from ...game import SoldierType, TableTactics



ESTIMATED_PIECE_VALUES = {
	SoldierType.Noble: 18,
	SoldierType.Fighter: 15,
	SoldierType.Thief: 14,
}

ACTION_INDICES_SYMMETRIES = [
	[
		[
			[
				[
					[y,x,z] # initialize as identity for now
					for z in range(11)
				]
				for x in range(6)
			]
			for y in range(6)
		]
		for k in range(4)
	]
	for flip in range(2)
]

z_flip_map = [0,3,2,1,4,7,6,5,8,9,10]
z_rot_map = [3,0,1,2,7,4,5,6,8,9,10]

for flip in range(2):
	for k in range(4):
		for y in range(6):
			for x in range(6):
				for z in range(11):
					if flip:
						nz = z_flip_map[z]
						ACTION_INDICES_SYMMETRIES[flip][k][y][x][z] = [5-y,x,nz]
					if k > 0:
						ry, rx, rz = ACTION_INDICES_SYMMETRIES[flip][k-1][y][x][z]
						nz = z_rot_map[rz]
						ACTION_INDICES_SYMMETRIES[flip][k][y][x][z] = [5-rx,ry,nz]

ACTION_INDICES_SYMMETRIES = np.array(ACTION_INDICES_SYMMETRIES, dtype=int)

def state_flip_symmetry(state, flip):
	if flip:
		return np.flip(state, 0)
	return state.copy()

def state_rot_symmetry(state, k):
	return np.rot90(state, k, (0,1))

def action_symmetry(action_indices, k, flip):
	if action_indices is None:
		return None
	y,x,z = action_indices # normally using y,x,k to denote model output array indices
	return ACTION_INDICES_SYMMETRIES[flip,k,y,x,z]

def actions_symmetry(actions_indices, k, flip):
	'''
	Vectorized action_symmetry().
	'''
	if actions_indices is None:
		return None
	Y,X,K = actions_indices.T
	return ACTION_INDICES_SYMMETRIES[flip,k,Y,X,K]



SOLDIER_TYPE_SPACE = np.array([-1, 0.25, 0.75])

def get_state(game, turn=None):
	# TODO: BECAUSE THERE ARE ONLY TWO ARMIES, IT IS BETTER TO LET -1 CORRESPOND TO PLAYER 0, 1 CORRESPOND TO PLAYER 1, AND 0 CORRESPOND TO NO PLAYER
	# ... to avoid exploding gradients

	if turn is None:
		turn = game.turn
	obstacle = game.board[:,:,[0]]
	army = game.board[:,:,[1]]
	soldier_type = game.board[:,:,[2]]
	soldier_hitpoints = game.board[:,:,[3]]
	soldier_actions = game.board[:,:,[4]]
	placement_mask = game.placement_mask[:,:,np.newaxis]
	lal = np.zeros((6,6,1))
	lal[game.last_action_location_y, game.last_action_location_x] = game.current_steps_remaining ** 0.5 * 0.5
	return np.concatenate((
		obstacle,
		np.where(army == -1, 0, 1-2*game.army_cycle[turn][army.astype(int)]),
		np.where(soldier_type == -1, 0, SOLDIER_TYPE_SPACE[soldier_type.astype(int)]),
		np.where(soldier_hitpoints == -1, 0, soldier_hitpoints*0.3),
		np.where(soldier_actions == -1, 0, 3-2*soldier_actions),
		np.where(placement_mask == -1, 0, 1-2*game.army_cycle[turn][placement_mask.astype(int)]),
		lal
	), axis=2)
	# conc = np.concatenate((game.board, game.placement_mask[:,:,np.newaxis]), axis=2)
	# conc_army = conc[:,:,[1,5]]
	# ac = game.army_cycle[turn][conc_army.astype(int)]
	# lal = np.zeros((6,6,1))
	# lal[game.last_action_location_y, game.last_action_location_x] = game.current_steps_remaining ** 0.5 * 0.5
	# return np.concatenate((
	# 	conc[:,:,[0]],
	# 	np.where(conc_army == -1, conc_army, ac),
	# 	np.interp(conc[:,:,[2,4]], (-1, 2), (-1, 1)),
	# 	np.interp(conc[:,:,[3]], (-1, 4), (-1, 1)),
	# 	lal,#lol
	# ), axis=2)

def games_to_input(games, turn=None):
	'''
	Convert an iterable of TableTactics instances to a NumPy array, suitable to be directly given to a Keras model.
	'''
	inp = np.zeros((len(games), 6, 6, 7))
	if turn is None or isinstance(turn, int):
		for i,game in enumerate(games):
			inp[i] = get_state(game, turn=turn)
	else:
		for i,(game,t) in enumerate(zip(games,turn)):
			inp[i] = get_state(game, turn=t)
	return inp

def game_valid_actions(game, include_end_turn = False):
	'''
	Return a list of actions that are valid in the game's current state.
	'''
	return list(game.valid_actions(include_end_turn=include_end_turn))

def games_valid_actions(games, include_end_turn = False):
	'''
	For each game in games, return a list of actions that are valid in the game's current state.

	This is a vectorized version of game_valid_actions().
	'''
	return [
		game_valid_actions(game, include_end_turn=include_end_turn)
		for game in games
	]

def heuristic_score(game, limit=np.inf):
	'''
	Compute a heuristic value of the score imbalance in the first player's favor, assuming there are only two players.
	For example, a negative score means the second player is likely in a better position than the first player.

	Refer to heuristics.md.
	'''
	if game.setup_phase:
		return 0.
	if not any(game.soldiers_of_army(0)):
		return -limit
	if not any(game.soldiers_of_army(1)):
		return limit
	s = np.log(
		np.square(
			sum(
				ESTIMATED_PIECE_VALUES[game.get_soldier_type(x, y)] * game.get_soldier_hitpoints_remaining(x, y)
				for x, y in game.soldiers_of_army(0)
			) /
			sum(
				ESTIMATED_PIECE_VALUES[game.get_soldier_type(x, y)] * game.get_soldier_hitpoints_remaining(x, y)
				for x, y in game.soldiers_of_army(1)
			)
		) *
		sum(
			np.square(game.get_soldier_hitpoints_remaining(x, y))
			for x, y in game.soldiers_of_army(0)
		) /
		sum(
			np.square(game.get_soldier_hitpoints_remaining(x, y))
			for x, y in game.soldiers_of_army(1)
		)
	)
	return min(max(s,-limit),limit)

def heuristic_scores(games, limit=np.inf):
	'''
	Vectorized heuristic_score()
	'''
	return np.array([
		heuristic_score(game, limit=limit)
		for game in games
	])

def simulate(actions):
	for a_func,a_args in actions:
		a_func(*a_args)

def random_action(game, valid_actions=None):
	# valid_actions can be supplied if it's already been calculated
	if valid_actions is None:
		valid_actions = list(game.valid_actions(include_end_turn=False))
	return valid_actions[np.random.randint(len(valid_actions))]

def random_actions(games, valid_actions=None):
	# vectorized random_action()
	# valid_actions should be None or a list of lists
	if valid_actions is None:
		return [
			random_action(game)
			for game in games
		]
	return [
		random_action(game, valid_actions=vas)
		for game, vas in zip(games, valid_actions)
	]


def action_to_indices(action):
	'''
	Take a single action and return a tuple representing the array indices to which the action maps.

	This is the inverse of indices_to_action().
	'''
	if not isinstance(action, tuple):
		raise TypeError(f'Incorrect type for action.  Received: {type(action)}')
	# Model output is array of shape (6,6,11).
	# 11 = 4 + 4 + 3, one layer for each movement/attack direction (4+4) or soldier type to place (3) on any given tile (6x6).
	# print(f'Action: {action}')
	a_func, a_args = action
	afn = a_func.__name__
	if afn == 'end_turn':
		raise ValueError('end_turn is not supported; end the turn automatically for the agent when it is the only remaining action.')
	if afn == 'move_soldier':
		return a_args[1],a_args[0],a_args[2]
	if afn == 'attack_soldier':
		return a_args[1],a_args[0],a_args[2]+4
	if afn == 'add_soldier':
		return a_args[1],a_args[0],int(a_args[2])+8

def indices_to_action(game, indices):
	'''
	Return the action in game to which indices map.

	This is the inverse of action_to_indices().
	'''
	if not isinstance(indices, (tuple, list, np.ndarray)):
		raise TypeError(f'Incorrect type for indices.  Received: {type(indices)}')
	y,x,k = indices
	if k >= 8:
		return game.add_soldier,(int(x),int(y),SoldierType(int(k)-8))
	if k >= 4:
		return game.attack_soldier,(int(x),int(y),int(k)-4)
	return game.move_soldier,(int(x),int(y),int(k))

def actions_to_indices(actions):
	'''
	Take a list of actions and return a list of tuples each representing the array indices to which the corresponding action maps.

	This is a vectorized version of action_to_indices().
	'''
	return np.array([
		action_to_indices(action)
		for action in actions
	], dtype=int)

def indices_to_actions(games, indices):
	'''
	Return the action for each game in games to which each set of indices in indices map.

	This is a vectorized version of indices_to_action().
	'''
	if isinstance(games, TableTactics):
		games = [games]*len(indices)
	return [
		indices_to_action(g, i)
		for g, i in zip(games, indices)
	]

def pred_argmax(prediction, valid_actions_indices):
	# A model will predict on game states.  The indices of the maximum values of its predictions are used in the deep double Q-learning update rule.
	# valid_actions_indices is necessary to filter out the predictions for actions that aren't possible.
	# valid_actions_indices should be a list of NumPy arrays
	argmax = np.zeros((prediction.shape[0],3), dtype=int)
	for i, (pred, indices) in enumerate(zip(prediction, valid_actions_indices)):
		if indices is None:
			argmax[i] = np.unravel_index(0, pred.shape)
			continue
		arr = np.zeros_like(pred)
		arr[:,:,:] = -np.inf
		for y,x,k in indices:
			arr[y,x,k] = pred[y,x,k]
		argmax[i] = np.unravel_index(np.argmax(arr), arr.shape)
	return argmax

