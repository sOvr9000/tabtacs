
'''
Using data.py and models.py to implement deep Q-learning.
'''

import numpy as np
from .data import action_to_indices, actions_to_indices, games_to_input, heuristic_score, heuristic_scores, pred_argmax, random_actions, simulate
from .models import predict_actions, copy_model

def train_model(
	model,
	parallel_games,
	game_generator,

	memory_capacity = 500000,
	steps_per_experience_replay = 10000,

	p2_action_selection = None,

	epsilon_min = 0.01,
	epsilon_max = 0.99,
	tau = 0.01,

	fit_batch_size = 256,
	fit_epochs = 2,
	fit_callbacks = None,

	verbose = True,
):
	'''Train a neural network to play Table Tactics.'''

	if fit_callbacks is None:
		fit_callbacks = []
	
	if p2_action_selection is None:
		p2_action_selection = random_actions

	def verbose_print(m='', **kwargs):
		if verbose:
			print(m, **kwargs)
	
	verbose_print('Initializing run...')

	target_model = copy_model(model)

	old_states = np.zeros((memory_capacity, *model.input_shape[1:]))
	new_states = old_states.copy()
	action_indices = np.zeros((memory_capacity, len(model.output_shape)-1), dtype=int)
	observed_rewards = np.zeros(memory_capacity)
	terminated = np.zeros(memory_capacity, dtype=bool)
	valid_actions_indices = [None] * memory_capacity

	transition_index = 0
	steps_since_experience_replay = 0
	populating_transitions = True

	epsilon = np.linspace(epsilon_min, epsilon_max, parallel_games, True)
	games = [game_generator(i) for i in range(parallel_games)]

	replays = []
	scores = []
	rewards_history = []

	verbose_print('Populating transition history...')

	while True:
		if populating_transitions:
			actions = random_actions(games)
		else:
			verbose_print('| Predicting actions...')
			actions = predict_actions(model, games, epsilon)
		_old_states = games_to_input(games)
		rewards = heuristic_scores(games, limit=20)
		verbose_print('| Simulating actions...')
		simulate(actions)

		verbose_print('| Simulating player 2 responses...')
		auto_play_games = [game for game in games if game.turn == 1]
		while len(auto_play_games) > 0:
			verbose_print('.', end='')
			simulate(p2_action_selection(auto_play_games))
			for i in range(len(auto_play_games)-1,-1,-1):
				if auto_play_games[i].turn == 0 or auto_play_games[i].is_game_over():
					del auto_play_games[i]

		_new_states = games_to_input(games, turn=0)
		rewards = heuristic_scores(games, limit=20) - rewards

		rewards_history.append(rewards)

		verbose_print('| Computing transition data...')
		num_reset = 0
		for i, (game, old_state, new_state, action, reward) in enumerate(zip(games, _old_states, _new_states, actions, rewards)):

			if not populating_transitions and steps_since_experience_replay >= steps_per_experience_replay:
				if num_reset > 0:
					verbose_print()
					num_reset = 0
				verbose_print(f'| | Experience replay...     Total games / steps simulated: {len(scores)} / {len(rewards)}')
				samples = steps_per_experience_replay * 2
				sample_indices = np.random.randint(0, memory_capacity, samples)
				sample_old_states = old_states[sample_indices]
				sample_new_states = new_states[sample_indices]
				sample_action_indices = action_indices[sample_indices]
				sample_rewards = observed_rewards[sample_indices]
				sample_terminated = terminated[sample_indices]

				pred_old_states = model.predict(sample_old_states)
				pred_new_states = model.predict(sample_new_states)
				pred_new_states_target = target_model.predict(sample_new_states)
				pred_new_states_argmax = pred_argmax(pred_new_states_target, map(valid_actions_indices.__getitem__, sample_indices))
				Y1,X1,K1 = sample_action_indices.T
				Y2,X2,K2 = pred_new_states_argmax.T
				pred_old_states[np.arange(samples),Y1,X1,K1] = sample_rewards + 0.9 * (1 - sample_terminated.astype(int)) * pred_new_states[np.arange(samples),Y2,X2,K2]

				model.fit(sample_old_states, pred_old_states, batch_size=fit_batch_size, epochs=fit_epochs, callbacks=fit_callbacks)

				# Polyak averaging
				target_model.set_weights([tau*w1+(1-tau)*w2 for w1, w2 in zip(model.get_weights(), target_model.get_weights())])

				steps_since_experience_replay = 0

				yield replays, scores, rewards_history

			# save transition
			old_states[transition_index] = old_state
			new_states[transition_index] = new_state
			action_indices[transition_index] = action_to_indices(action)
			observed_rewards[transition_index] = reward

			if game.is_game_over():

				if num_reset == 0:
					verbose_print(f'| | Reset games:', end='')
				if num_reset % 8 == 0:
					verbose_print(f',\n| | | {i:05d}', end='')
				else:
					verbose_print(f', {i:05d}', end='')
				num_reset += 1

				terminated[transition_index] = True
				if game.record_replay:
					replays.append(game.get_replay())
				scores.append(game.get_score())
				games[i] = game_generator(i)
			else:
				terminated[transition_index] = False
				valid_actions_indices[transition_index] = actions_to_indices(game.valid_actions(include_end_turn=False))

			transition_index = (transition_index + 1) % memory_capacity
			if transition_index == 0 and populating_transitions:
				# trigger the training of the agent
				if num_reset > 0:
					verbose_print()
					num_reset = 0
				verbose_print('Finished populating transition history')
				steps_since_experience_replay = 0
				populating_transitions = False

			steps_since_experience_replay += 1




