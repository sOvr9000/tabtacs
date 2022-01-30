
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense, Conv2D, Flatten, Reshape

from ...game import action_to_str
from .data import game_valid_actions, games_to_input, games_valid_actions, indices_to_action, indices_to_actions, pred_argmax, actions_to_indices, random_action


def build_model():
	model = tf.keras.Sequential([
		InputLayer((6,6,7)),
		Conv2D(128, (2,2), (1,1), 'valid', activation='relu'),
		Conv2D(256, (3,3), (1,1), 'valid', activation='relu'),
		Conv2D(128, (2,2), (1,1), 'valid', activation='relu'),
		Flatten(),
		Dense(512, 'relu'),
		Dense(512, 'relu'),
		Dense(396),
		Reshape((6,6,11)), # maybe this just... works?
	])
	model.compile('adam', 'msle')
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
	scaled = q_values/max(abs(q_values.max()),abs(q_values.min()))
	interpolated = np.exp(scaled)
	interpolated /= interpolated.sum() # like softmax
	discrete_interpolated = (interpolated*32+.5).astype(int)
	return '\n'.join(
		'{} | {:+3.3f} | {}'.format(
			'#'*interp + ' '*(32-interp),
			pred[y,x,k],
			action_str
		)
		for action_str, (y, x, k), interp in sorted(zip(action_strs, action_indices, discrete_interpolated), key=lambda t:pred[t[1][0],t[1][1],t[1][2]], reverse=True)
	)

