
import numpy as np
from ..game import SoldierType



ESTIMATED_PIECE_VALUES = {
	SoldierType.Noble: 18,
	SoldierType.Fighter: 15,
	SoldierType.Thief: 14,
}

def get_state(game):
	'''
	Return two-tuple of arrays.  First array is the board state, second array is the extraneous information defining the overall game state.
	'''
	return \
		np.interp(
			np.concatenate((game.board, game.placement_mask[:,:,np.newaxis]), axis=2),
			(-1, (1, 1, 2, 4, 2, 1)),
			(-1, 1)
		), \
		np.interp(
			[game.last_action_position_x, game.last_action_position_y, game.current_steps_remaining],
			((-1, -1, 0), (5, 5, 4)),
			(-1, 1)
		)

def games_to_input(games):
	'''
	Convert an iterable of TableTactics instances to a list of two inputs, suitable to be directly given to a Keras model.
	'''
	inp_board = np.zeros((len(games), 6, 6, 6))
	inp_extra = np.zeros((len(games), 3))
	for i,game in enumerate(games):
		inp_board[i], inp_extra[i] = get_state(game)
	return [inp_board, inp_extra]

def heuristic_score(game):
	'''
	Compute a heuristic value of the score imbalance in the first player's favor, assuming there are only two players.
	For example, a negative score means the second player is likely in a better position than the first player.

	Refer to heuristics.md.
	'''
	if game.setup_phase:
		return 0.
	if not any(game.soldiers_of_army(0)):
		return -np.inf
	if not any(game.soldiers_of_army(1)):
		return np.inf
	return np.log(
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

def simulate(games, actions):
	scores = map(heuristic_score,games)
	for i,(game,(a_func,a_args)) in enumerate(zip(games,actions)):
		a_func(*a_args)
		scores[i] = heuristic_score(game) - scores[i]
	return scores # change in score

def random_action(game):
	va = list(game.valid_actions())
	return va[np.random.randint(len(va))]


def action_to_indices(action):
	# Model output 0 is array of shape (6,6,11).
	# 11 = 4 + 4 + 3, one layer for each movement/attack direction (4+4) or soldier type to place (3) on any given tile (6x6).
	# Model output 1 is a float value indicating the Q-value of ending the turn.
	a_func, a_args = action
	afn = a_func.__name__
	if afn == 'end_turn':
		return 1,0
	if afn == 'move_soldier':
		return 0,(a_args[1],a_args[0],a_args[2])
	if afn == 'attack_soldier':
		return 0,(a_args[1],a_args[0],a_args[2]+4)
	if afn == 'add_soldier':
		return 0,(a_args[1],a_args[0],int(a_args[2])+8)

def indices_to_action(game, indices):
	# inverse of function above (game must be provided to return bound methods)
	i,t = indices
	if i == 1:
		return game.end_turn,()
	y,x,k = t
	if k >= 8:
		return game.add_soldier,(x,y,SoldierType(k-8))
	if k >= 4:
		return game.attack_soldier,(x,y,k-4)
	return game.move_soldier,(x,y,k)

def actions_to_indices(actions):
	# vectorized action_to_indices()
	dim0_indices, dim1_indices = zip(*(action_to_indices(action) for action in actions))
	print(dim0_indices)
	print(dim1_indices)
	return np.array(dim0_indices, dtype=int), np.array(dim1_indices, dtype=int)

def indices_to_actions(game, indices):
	# vectorized indices_to_action()
	dim0_indices, dim1_indices = indices
	return np.array([
		indices_to_action(game, i)
		for i in zip(dim0_indices, dim1_indices)
	], dtype=int)

def pred_argmax(prediction, valid_actions):
	# A model will predict on game states.  The indices of the maximum values of its predictions are used in the deep double Q-learning update rule.
	# valid_actions is necessary to filter out the predictions for actions that aren't possible.
	valid_indices = actions_to_indices(valid_actions)
	arr1 = np.zeros_like(prediction[0])
	arr2 = np.zeros_like(prediction[1])
	arr1 = -np.inf
	arr2 = -np.inf
	for z, ((i, j), k) in enumerate(valid_indices):
		arr1[z,i,j] = prediction[0][z,i,j]
		arr2[z,k] = prediction[1][z,k]
	argmax = []
	for pred_board, pred_extra in zip(arr1, arr2):
		if np.max(pred_board) > np.max(pred_extra):
			argmax.append(np.argmax(pred_board))
		else:
			argmax.append(np.argmax(pred_extra))
	return argmax

