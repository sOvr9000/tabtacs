
import json

from numpy import place
from .taclib import DIRECTION_NAMES
from .enums import SoldierType



ACTION_LEGEND = {
	'a': 'attack_soldier',
	'p': 'add_soldier',
	'm': 'move_soldier',
	'_': 'end_turn',
}
ACTION_LEGEND_INV = {v:k for k,v in ACTION_LEGEND.items()}

SOLDIER_LEGEND = {
	'+': SoldierType.Noble,
	'f': SoldierType.Fighter,
	't': SoldierType.Thief,
}
SOLDIER_LEGEND_INV = {v:k for k,v in SOLDIER_LEGEND.items()}



class Replay:
	def __init__(self, game_setup):
		self.game_setup = game_setup
		self.action_history = []
	def append_action(self, func, args):
		self.action_history.append((func.__name__, args)) # do not save bound methods, just their names
	def __repr__(self):
		return 'Action history:\n{}'.format(
			'\n'.join(
					'End Turn'
				if fname == 'end_turn' else
					f'Place {args[2].name} at ({args[0]}, {args[1]})'
				if fname == 'add_soldier' else
					'{} {} from ({}, {})'.format(
						'Move Soldier' if fname == 'move_soldier' else 'Attack Soldier',
						DIRECTION_NAMES[args[2]],
						args[0], args[1]
					)
				for fname, args in self.action_history
			)
		)
	def show(self):
		from .tabletactics import TableTactics
		game = TableTactics(setup = self.game_setup, auto_end_turn = False, record_replay = False)
		print(game)
		for fname, args in self.action_history:
			getattr(game, fname)(*args)
			print(game)
	def save(self, fpath):
		'''
		Append to the file at fpath a line that encodes this replay.  Existing data is never overwritten.
		'''
		# There's potential to encode the entire history as binary data to compress by a factor of 100x - 10000x
		with open(fpath, 'a') as f:
			f.write('{};{}\n'.format(
				json.dumps(self.game_setup),
				','.join(
					(ACTION_LEGEND_INV[fname] + ' ' + ' '.join((str(args[0]), str(args[1]), SOLDIER_LEGEND_INV[args[2]])))
					if fname == 'add_soldier' else
					(ACTION_LEGEND_INV[fname] + ' ' + ' '.join(map(str,args)))
					for fname, args in self.action_history
				)
			))

def load_replays(fpath, start_index = 0, end_index = None):
	if end_index is None:
		end_index = 1e16 # if there are more than 1e16 lines in your file then what I'm concerned about is NOT this assumption of a hard limit being made
	with open(fpath, 'r') as f:
		for i, line in enumerate(f):
			if i < start_index:
				continue
			if i >= end_index:
				break
			js, action_history_str = line.split(';')
			game_setup = json.loads(js)
			for _i, placement_space in enumerate(game_setup['placement_space']):
				game_setup['placement_space'][_i] = list(map(tuple, placement_space))
			for _i, soldiers in enumerate(game_setup['soldiers']):
				fixed_soldiers = {}
				for k, v in soldiers.items():
					fixed_soldiers[SoldierType(int(k))] = v
				game_setup['soldiers'][_i] = fixed_soldiers
			action_history = []
			for s in action_history_str.split(','):
				params = s.split(' ')
				fname = ACTION_LEGEND[params[0]]
				if fname == 'end_turn':
					args = ()
				elif fname == 'add_soldier':
					args = (int(params[1]), int(params[2]), SOLDIER_LEGEND[params[3]])
				else:
					args = (int(params[1]), int(params[2]), int(params[3]))
				action_history.append((fname, args))
			replay = Replay(game_setup)
			replay.action_history = action_history
			yield replay
