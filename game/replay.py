
import json

from .board_setups import fix_board_setup
from .taclib import DIRECTION_NAMES, action_to_str, count_lines
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



def fix_args(args):
	if len(args) > 0:
		if not isinstance(args[2], SoldierType):
			return (int(args[0]), int(args[1]), int(args[2]))
		return (int(args[0]), int(args[1]), args[2])
	return args

class Replay:
	def __init__(self, game_setup):
		self.game_setup = game_setup
		self.action_history = []
	def append_action(self, func, args):
		self.action_history.append((func.__name__, fix_args(args))) # do not save bound methods, just their names
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
	def step(self):
		'''
		Run the replay of the game and yield an instance of TableTactics in the current state and the action which brought it to that state from the previous one.
		'''
		from .tabletactics import TableTactics
		game = TableTactics(setup = self.game_setup, auto_end_turn = False, record_replay = False)
		yield game, (None, ())
		for fname, args in self.action_history:
			func = getattr(game, fname)
			func(*args)
			yield game, (func, args)
	def show(self, input_pause=False, show_stats=True):
		if self.is_empty():
			print('Empty replay')
		for game, (func, args) in self.step():
			print(game)
			if func is not None:
				print('Played move: ' + action_to_str(func, args))
			if show_stats:
				print(' ' * 16 + 'White   Black')
				print(f'    Pieces        {game.get_soldiers_remaining(0)}       {game.get_soldiers_remaining(1)}')
				print(f'   Hitpoints      {sum(map(game.get_soldier_hitpoints_remaining,game.soldiers_of_army(0)))}       {sum(map(game.get_soldier_hitpoints_remaining,game.soldiers_of_army(1)))}')
				print(f'Noble Hitpoints   {sum(map(game.get_soldier_hitpoints_remaining,game.soldiers_of_army(0, SoldierType.Noble)))}       {sum(map(game.get_soldier_hitpoints_remaining,game.soldiers_of_army(1, SoldierType.Noble)))}')
				print()
			yield game, (func, args)
			if input_pause:
				input()
	def is_empty(self):
		return len(self.action_history) == 0
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
	def final_state(self):
		for game, _ in self.step(): pass
		return game

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
			fix_board_setup(game_setup)
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

def count_replays(fpath):
	'''
	Return the number of lines (possible replays) in a file.
	Useful for knowing how large a file of replays is and how many can or should be loaded.
	'''
	return count_lines(fpath)

def final_states(replays):
	for replay in replays:
		yield replay.final_state()
