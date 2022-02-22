
'''
Using data.py and models.py to implement deep Q-learning.
'''

import numpy as np
from .data import action_to_indices, actions_to_indices, games_to_input, heuristic_scores, pred_argmax, random_actions, simulate, state_flip_symmetry, state_rot_symmetry, action_symmetry, actions_symmetry
from .models import predict_actions, copy_model

def train_model(
	model,
	parallel_games,
	game_generator,

	memory_capacity = 500000,
	steps_per_experience_replay = 10000,

	iteration_duration = 10000,
	iteration_callbacks = None, # mainly used to update opponent_action_selection

	opponent_action_selection = None,

	epsilon_min = 0.01,
	epsilon_max = 0.99,
	tau = 0.01,

	gamma = 0.97,

	fit_batch_size = 256,
	fit_epochs = 2,
	fit_callbacks = None,

	verbose = True,
):
	'''Train a neural network to play Table Tactics.'''

	if fit_callbacks is None:
		fit_callbacks = []

	if iteration_callbacks is None:
		iteration_callbacks = []
	
	if opponent_action_selection is None:
		opponent_action_selection = random_actions

	def verbose_print(m='', **kwargs):
		if verbose:
			print(m, **kwargs)

	target_model = copy_model(model)

	epsilon = np.linspace(epsilon_min, epsilon_max, parallel_games, True)
	games = [game_generator(i) for i in range(parallel_games)]
	agent_player = np.arange(parallel_games, dtype=int) % 2
	agent_player_score_mult = 1-2*agent_player

	def simulate_responses():
		# For each running game in the simulation, ensure that the current player to move is the agent that's training.
		# All moves played by the agent that isn't training are selected with opponent_action_selection().
		_games = []
		_agent_player = []
		for game, player in zip(games, agent_player):
			if game.turn != player and not game.is_game_over():
				_games.append(game)
				_agent_player.append(player)
		while len(_games) > 0:
			verbose_print('.', end='')
			simulate(opponent_action_selection(_games))
			for i in range(len(_games)-1,-1,-1):
				if _games[i].turn == _agent_player[i] or _games[i].is_game_over():
					del _games[i]
					del _agent_player[i]

	iteration = 0
	while True:
		verbose_print(f'=== Initializing run for iteration #{iteration}... ===')

		old_states = np.zeros((memory_capacity, *model.input_shape[1:]))
		new_states = old_states.copy()
		action_indices = np.zeros((memory_capacity, len(model.output_shape)-1), dtype=int)
		observed_rewards = np.zeros(memory_capacity)
		terminated = np.zeros(memory_capacity, dtype=bool)
		valid_actions_indices = [None] * memory_capacity

		transition_index = 0
		steps_since_experience_replay = 0
		populating_transitions = True

		replays = []
		scores = []
		rewards_history = []

		verbose_print('=== Populating transition history... ===')

		while True:
			if len(scores) >= iteration_duration:
				# Start the next iteration!
				verbose_print('=== Reached end of iteration ===')
				for cb in iteration_callbacks:
					cb(iteration)
				verbose_print('=== Starting next iteration... ===')
				break
			
			if populating_transitions:
				verbose_print(f'\n| Current transition index / max: {transition_index} / {memory_capacity}')

			verbose_print('\n| Initializing reset games...')
			simulate_responses() # they all start with player 1 to move, so where the agent is player 2, bring each game to the state where it's player 2 to move

			if populating_transitions:
				actions = random_actions(games)
			else:
				verbose_print('\n| Predicting actions...')
				actions = predict_actions(model, games, epsilon)
			_old_states = games_to_input(games)
			verbose_print('\n| Simulating actions...')
			simulate(actions)

			verbose_print('| Simulating responses...')
			simulate_responses()

			_new_states = games_to_input(games, turn=agent_player)
			rewards = heuristic_scores(games, limit=20)
			rewards *= agent_player_score_mult

			rewards_history.append(rewards)

			verbose_print('\n| Computing transition data...')
			num_reset = 0
			for i, (game, old_state, new_state, action, reward, player) in enumerate(zip(games, _old_states, _new_states, actions, rewards, agent_player)):

				if not populating_transitions and steps_since_experience_replay >= steps_per_experience_replay:
					if num_reset > 0:
						verbose_print()
						num_reset = 0
					verbose_print(f'| | Experience replay...     Total games simulated: {len(scores)}')

					samples = steps_per_experience_replay * 16
					verbose_print(f'| | | Sampling transition memory... (samples = {samples})')
					sample_indices = np.random.randint(0, memory_capacity, samples)
					sample_old_states = old_states[sample_indices]
					sample_new_states = new_states[sample_indices]
					sample_action_indices = action_indices[sample_indices]
					sample_rewards = observed_rewards[sample_indices]
					sample_terminated = terminated[sample_indices]

					verbose_print('| | | Correcting model predictions...')
					pred_old_states = model.predict(sample_old_states)
					pred_new_states = model.predict(sample_new_states)
					pred_new_states_target = target_model.predict(sample_new_states)
					pred_new_states_argmax = pred_argmax(pred_new_states_target, map(valid_actions_indices.__getitem__, sample_indices))
					Y1,X1,K1 = sample_action_indices.T
					Y2,X2,K2 = pred_new_states_argmax.T
					verbose_print('| | | | Indexing...') # This part can take a while for large arrays
					if samples >= 4096:
						# speed work-around (far fewer hash lookups, much faster)
						for entry_index in range(0, samples, 2048):
							M = min(entry_index+2048,samples)
							pred_old_states[
								np.arange(entry_index,M),
								Y1[entry_index:M],
								X1[entry_index:M],
								K1[entry_index:M]
							] = sample_rewards[entry_index:M] + gamma * (
								1 - sample_terminated[entry_index:M].astype(int)
							) * pred_new_states[
								np.arange(entry_index,M),
								Y2[entry_index:M],
								X2[entry_index:M],
								K2[entry_index:M]
							]
					else:
						pred_old_states[np.arange(samples),Y1,X1,K1] = sample_rewards + gamma * (1 - sample_terminated.astype(int)) * pred_new_states[np.arange(samples),Y2,X2,K2]

					verbose_print('| | | Fitting...')
					model.fit(sample_old_states, pred_old_states, batch_size=fit_batch_size, epochs=fit_epochs, callbacks=fit_callbacks)

					# Polyak averaging
					verbose_print('| | | Updating target model weights...')
					target_model.set_weights([tau*w1+(1-tau)*w2 for w1, w2 in zip(model.get_weights(), target_model.get_weights())])

					steps_since_experience_replay = 0

					yield iteration, replays, scores, rewards_history

				t_old_state = old_state
				t_new_state = new_state
				t_action_indices = action_to_indices(action)
				t_observed_reward = reward
				t_valid_actions_indices = None

				if game.is_game_over():

					if num_reset == 0:
						verbose_print(f'| | Reset games:', end='')
					if num_reset % 8 == 0:
						if num_reset > 0:
							verbose_print(',', end='')
						verbose_print(f'\n| | | {i:05d}', end='')
					else:
						verbose_print(f', {i:05d}', end='')
					num_reset += 1

					t_terminated = True
					if game.record_replay:
						replays.append(game.get_replay())
					scores.append(game.get_score()[player])
					games[i] = game_generator(i)
				else:
					t_terminated = False
					t_valid_actions_indices = actions_to_indices(game.valid_actions(include_end_turn=False))

				# save transition and its symmetries
				for s_old_state, \
					s_new_state, \
					s_action_indices, \
					s_observed_reward, \
					s_terminated, \
					s_valid_actions_indices \
				in symmetric_transitions(
					t_old_state,
					t_new_state,
					t_action_indices,
					t_observed_reward,
					t_terminated,
					t_valid_actions_indices
				):
					old_states[transition_index] = s_old_state
					new_states[transition_index] = s_new_state
					action_indices[transition_index] = s_action_indices
					observed_rewards[transition_index] = s_observed_reward
					terminated[transition_index] = s_terminated,
					valid_actions_indices[transition_index] = s_valid_actions_indices
					transition_index = (transition_index + 1) % memory_capacity

					if transition_index == 0 and populating_transitions:
						# trigger the training of the agent
						if num_reset > 0:
							verbose_print()
							num_reset = 0
						verbose_print('=== Finished populating transition history ===')
						steps_since_experience_replay = 0
						populating_transitions = False

				steps_since_experience_replay += 1

		iteration += 1




def symmetric_transitions(old_state, new_state, action_indices, reward, terminated, valid_actions_indices):
	old_state_flipped = state_flip_symmetry(old_state, 1)
	new_state_flipped = state_flip_symmetry(new_state, 1)
	for flip in range(2):
		for k in range(4):
			yield \
			state_rot_symmetry(old_state_flipped if flip else old_state, k), \
			state_rot_symmetry(new_state_flipped if flip else new_state, k), \
			action_symmetry(action_indices, k, flip), \
			reward, \
			terminated, \
			actions_symmetry(valid_actions_indices, k, flip)
