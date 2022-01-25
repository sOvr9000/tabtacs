
'''
Randomly simulate the game.
'''


import numpy as np
from .taclib import actions_to_str, action_to_str



class RandomSim:
	def __init__(self, game, display = True, input_pause = True, step_callbacks = []):
		self.game = game
		self.display = display
		self.input_pause = input_pause # use input() to pause each step in the game to see what's happening (only used if display = True)
		self.step_callbacks = step_callbacks
	def run(self):
		self.game.reset()
		while True:
			va = list(self.game.valid_actions())
			print(actions_to_str(va))
			if len(va) == 0:
				break
			if len(va) > 1:
				va = va[1:] # maybe don't end the turn if other actions are available...
			action_func, action_args = va[np.random.randint(len(va))]
			action_func(*action_args)
			if self.display:
				print(f'Selected action: {action_to_str(action_func, action_args)}\n')
				print(self.game)
			for cb in self.step_callbacks:
				cb()
			if self.input_pause:
				input()

