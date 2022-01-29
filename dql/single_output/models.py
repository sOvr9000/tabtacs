
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense, Conv2D, Flatten, Reshape
from .data import games_to_input, games_valid_actions, indices_to_action, pred_argmax, actions_to_indices, random_action


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


