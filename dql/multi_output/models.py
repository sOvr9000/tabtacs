
from random import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, Concatenate, Flatten, Reshape, Conv2DTranspose
from .data import games_to_input, indices_to_actions, pred_argmax, actions_to_indices, random_actions


def build_model():
	input_board = Input((6,6,6))
	input_extra = Input((3,))

	x = Conv2D(256, (2,2), (1,1), 'valid', activation='relu')(input_board)
	x = Conv2D(256, (2,2), (1,1), 'valid', activation='relu')(x)
	c3x3 = Conv2D(128, (2,2), (1,1), 'valid', activation='relu')(x)
	sub_output_board1 = Flatten()(c3x3)
	x = Conv2D(128, (2,2), (1,1), 'valid', activation='relu')(c3x3)
	sub_output_board2 = Flatten()(x)
	x = Concatenate()((sub_output_board1, sub_output_board2, input_extra))

	x = Dense(512, 'relu')(x)
	x = Dense(256, 'relu')(x)
	sub_hidden_output = Dense(256, 'relu')(x)
	hidden_output = Dense(180, 'relu')(sub_hidden_output)

	x = Reshape((3,3,20))(hidden_output)
	x = Concatenate()((x, c3x3))
	x = Conv2DTranspose(256, (2,2), (1,1), 'valid', dilation_rate=(2,2), activation='relu')(x)
	output_board = Conv2D(11, (2,2), (1,1), 'same')(x)

	x = Dense(256, 'relu')(sub_hidden_output)
	x = Dense(256, 'relu')(x)
	output_end_turn = Dense(1)(x)

	model = tf.keras.Model(inputs=[input_board, input_extra], outputs=[output_board, output_end_turn])
	model.compile('adam', 'mse')
	return model


def model_predict(model, games):
	return model.predict(games_to_input(games))


def predict_actions(model, games, epsilon):
	pred = model_predict(model, games)
	valid_actions_indices = [actions_to_indices(game.valid_actions()) for game in games]
	action_indices = pred_argmax(pred, valid_actions_indices)
	actions = indices_to_actions(games, action_indices)
	r_epsilon = np.random.random(len(valid_actions_indices)) < epsilon
	random_action = random_actions(games)
	return [
		ra if r else a
		for ra,a,r in zip(random_action, actions, r_epsilon)
	]


