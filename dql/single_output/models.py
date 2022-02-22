
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, TimeDistributed, GlobalAveragePooling2D, multiply

from ...game import action_to_str, TableTactics, random_board_setup, total_scores_to_str
from .data import game_valid_actions, games_to_input, games_valid_actions, indices_to_action, pred_argmax, actions_to_indices, random_action, simulate


def build_model(blocks = 10, filters = 128, dense_layers = 3, dense_units = 32, learning_rate = 0.001):
	# Following the architecture of LCZero: https://lczero.org/dev/backend/nn/
	# ... with additional layers.
	def squeeze_and_excitation_layer(inp_convolution, num_channels):
		x = GlobalAveragePooling2D()(inp_convolution)
		x = Dense(num_channels // 16, 'relu')(x)
		x = Dense(num_channels, 'sigmoid')(x)
		return multiply([inp_convolution, x])
	def block(inp_convolution):
		x = Conv2D(filters, (3,3), (1,1), 'same', activation='relu')(inp_convolution)
		return squeeze_and_excitation_layer(x, filters)

	inp_board = Input((6,6,7))
	x = inp_board

	for _ in range(dense_layers):
		x = TimeDistributed(TimeDistributed(Dense(dense_units, 'relu')))(x)

	for _ in range(blocks):
		x = block(x)
	
	x = Conv2D(11, (3,3), (1,1), 'same', activation='linear')(x)

	model = tf.keras.Model(inputs=inp_board, outputs=x)
	model.compile('adam', 'mse')
	model.optimizer.learning_rate = learning_rate
	return model


def load_model(fpath):
	return tf.keras.models.load_model(fpath)


def copy_model(model):
	copy = tf.keras.models.clone_model(model)
	copy.set_weights(model.get_weights())
	return copy


def model_predict(model, games):
	return model.predict(games_to_input(games))


def predict_actions(model, games, epsilon):
	pred = model_predict(model, games)
	valid_actions = games_valid_actions(games)
	valid_actions_indices = [
		actions_to_indices(vas)
		for vas in valid_actions
	]
	argmax = pred_argmax(pred, valid_actions_indices)
	rand = np.random.random(len(games)) < epsilon # for each game, epsilon is the chance of the agent performing a random valid action
	actions = [
		random_action(game, valid_actions=vas)
		if use_epsilon else
		indices_to_action(game, indices)
		for indices, vas, use_epsilon, game in zip(argmax, valid_actions, rand, games)
	]
	return actions


def predict_str(model, game):
	'''
	Return a nicely formatted string showing what the model thinks about the current position in the game (from the perspective of the current player to move).

	Each of the possible valid actions is listed on a separate line.
	Preceding each action is the raw Q-value of that action (produced by the model) and the relative "mass" of that Q-value in comparison to the Q-values of the other valid actions.
	For example, the bar will be twice as long for a Q-value that is twice as "good" (by some definition that is difficult to describe).
	Q-values that are close to each other will have bars of nearly identical length because that implies the model thinks the actions are equally as good.
	'''
	actions = game_valid_actions(game)
	if len(actions) == 0:
		return '#'*32 + ' | N/A | ' + action_to_str(game.end_turn, ())
	pred = model_predict(model, [game])[0]
	action_indices = actions_to_indices(actions)
	action_strs = list(map(action_to_str,actions))
	Y,X,K = action_indices.T
	q_values = pred[Y,X,K]
	interpolated = np.interp(q_values, (q_values.min(), q_values.max()), (0, 1))
	discrete_interpolated = (interpolated*32+.5).astype(int)
	return '\n'.join(
		'{} | {:+3.3f} | {}'.format(
			'#'*max(1,interp) + ' '*(32-max(1,interp)),
			pred[y,x,k],
			action_str
		)
		for action_str, (y, x, k), interp in sorted(zip(action_strs, action_indices, discrete_interpolated), key=lambda t:pred[t[1][0],t[1][1],t[1][2]], reverse=True)
	)

def evaluate_models(models, game_generator = None, num_games_per_pair = 100, verbose = True):
	'''
	Return the total points scored by each model in a round-robin tournament.

	If game_generator is None, then random setups are used to set up new TableTactics instances for each game.
	'''
	if game_generator is None:
		game_generator = lambda i: TableTactics(setup=random_board_setup((6,6)), auto_end_turn=True, record_replay=False)
	scores = [0] * len(models)
	num_pairings = len(models) * (len(models) - 1) // 2
	if verbose:
		print(f'Beginning round-robin tournament between {len(models)} contestants...')
		print(f'Total pairings: {num_pairings}')
		print(f'Total number of games per contestant: {num_games_per_pair * (len(models) - 1)}')
		print(f'Total number of games to simulate: {num_games_per_pair * num_pairings}')
		print()
	K = 0
	for i, model1 in enumerate(models[:-1]):
		for j, model2 in enumerate(models[i+1:]):
			j += i+1
			if verbose:
				print(f'Pairing {K+1} / {num_pairings}')
			games = [game_generator(z) for z in range(num_games_per_pair)]
			model1_player = [n%2 for n in range(num_games_per_pair)]
			while len(games) > 0:
				model1_games = []
				model2_games = []
				for game, player in zip(games, model1_player):
					if game.turn == player:
						model1_games.append(game)
					else:
						model2_games.append(game)
				to_remove = []
				if len(model1_games) > 0:
					simulate(predict_actions(model1, model1_games, 0))
					for game in model1_games:
						if game.is_game_over():
							to_remove.append(game)
				if len(model2_games) > 0:
					simulate(predict_actions(model2, model2_games, 0))
					for game in model2_games:
						if game.is_game_over():
							to_remove.append(game)
				for game in to_remove:
					score = game.get_score()
					_i = model1_player[games.index(game)]
					scores[i] += score[_i]
					scores[j] += score[1-_i]
					del model1_player[games.index(game)]
					games.remove(game)
			K += 1
			if verbose:
				print(f'Current scores: {total_scores_to_str(scores)}\n')
	return scores
