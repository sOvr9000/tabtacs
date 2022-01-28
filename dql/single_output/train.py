
'''
Using data.py and models.py to implement deep Q-learning.
'''

import numpy as np
from .data import action_to_indices, actions_to_indices, games_to_input, heuristic_score, heuristic_scores, pred_argmax, random_actions, simulate
from .models import predict_actions

def train_model(
	model,
	parallel_games,
	game_generator,

	memory_capacity = 500000,

	epsilon_initial = 1.0,
	epsilon_min = 0.05,
	epsilon_rate = 0.99,

	fit_batch_size = 256,
	fit_epochs = 2,
	fit_callbacks = None,
):
	'''Train a neural network to play Table Tactics.'''

	if fit_callbacks is None:
		fit_callbacks = []

	old_states = np.zeros((memory_capacity, *model.input_shape[1:]))
	new_states = old_states.copy()
	action_indices = np.zeros((memory_capacity, len(model.output_shape)-1), dtype=int)
	observed_rewards = np.zeros(memory_capacity)
	terminated = np.zeros(memory_capacity, dtype=bool)
	valid_actions_indices = [None] * memory_capacity

	transition_index = 0
	steps_since_experience_replay = 0
	populating_transitions = True

	epsilon = epsilon_initial
	games = [game_generator(i) for i in range(parallel_games)]

	while True:
		if populating_transitions:
			actions = random_actions(games)
		else:
			actions = predict_actions(model, games, epsilon)
		_old_states = games_to_input(games)
		rewards = heuristic_scores(games, limit=20)
		simulate(actions)

		auto_play_games = [game for game in games if game.turn == 1]
		while len(auto_play_games) > 0:
			simulate(random_actions(auto_play_games))
			for i in range(len(auto_play_games)-1,-1,-1):
				if auto_play_games[i].turn == 0 or auto_play_games[i].is_game_over():
					del auto_play_games[i]

		_new_states = games_to_input(games)
		rewards = heuristic_scores(games, limit=20) - rewards

		do_experience_replay = False

		for i, (game, old_state, new_state, action, reward) in enumerate(zip(games, _old_states, _new_states, actions, rewards)):
			# save transition
			old_states[transition_index] = old_state
			new_states[transition_index] = new_state
			action_indices[transition_index] = action_to_indices(action)
			observed_rewards[transition_index] = reward

			if game.is_game_over():
				terminated[transition_index] = True
				if game.record_replay:
					yield game.get_replay()
				games[i] = game_generator(i)
				epsilon = max(epsilon_min, epsilon * epsilon_rate)
				if i == 0 and not populating_transitions:
					do_experience_replay = True
			else:
				terminated[transition_index] = False
				valid_actions_indices[transition_index] = actions_to_indices(game.valid_actions(include_end_turn=False))

			transition_index = (transition_index + 1) % memory_capacity
			if transition_index == 0 and populating_transitions:
				# trigger the training of the agent
				populating_transitions = False
			steps_since_experience_replay += 1

		if not populating_transitions and do_experience_replay:
			samples = steps_since_experience_replay * 4
			sample_indices = np.random.randint(0, memory_capacity, samples)
			sample_old_states = old_states[sample_indices]
			sample_new_states = new_states[sample_indices]
			sample_action_indices = action_indices[sample_indices]
			sample_rewards = observed_rewards[sample_indices]
			sample_terminated = terminated[sample_indices]

			pred_old_states = model.predict(sample_old_states)
			pred_new_states = model.predict(sample_new_states)
			pred_new_states_argmax = pred_argmax(pred_new_states, map(valid_actions_indices.__getitem__, sample_indices))
			Y1,X1,K1 = sample_action_indices.T
			Y2,X2,K2 = pred_new_states_argmax.T
			pred_old_states[np.arange(samples),Y1,X1,K1] = sample_rewards + 0.9 * (1 - sample_terminated.astype(int)) * pred_new_states[np.arange(samples),Y2,X2,K2]

			model.fit(sample_old_states, pred_old_states, batch_size=fit_batch_size, epochs=fit_epochs, callbacks=fit_callbacks)

			steps_since_experience_replay = 0




