
from .enums import SoldierType



DIRECTION_NAMES = ['right', 'up', 'left', 'down'] # "up" and "down" may be flipped if the array is printed directly, as this assumes positive y direction means pointing upward



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



def action_to_str(f, a=None):
	# Actions are pairs of bound methods and possible arguments.  To prevent print from spamming the screen, use this method instead of print(f,a).
	if isinstance(f, tuple):
		f,a = f
	fname = f.__name__
	if fname == 'end_turn':
		return 'End turn'
	if fname == 'add_soldier':
		return f'Place {a[2].name} at {a[0],a[1]}'
	if fname == 'attack_soldier':
		return f'Attack {DIRECTION_NAMES[a[2]]} from {a[0],a[1]}'
	if fname == 'move_soldier':
		return f'Move {DIRECTION_NAMES[a[2]]} from {a[0],a[1]}'
	return '({}, ({}))'.format(fname, ', '.join(map(str,a)))

def actions_to_str(va, sep='\n'):
	return sep.join(
		action_to_str(f, a)
		for f, a in va
	)

def parse_action(game, s):
	'''
	Given a string of space-separated values, return an action bound to game that the string represents.

	Example strings:
	- "p 1 2 +" -> (game.add_soldier, (1, 2, SoldierType.Noble))
	- "m 4 4 0" -> (game.move_soldier, (4, 4, 0))
	- "a 3 0 3" -> (game.attack_soldier, (3, 0, 3))
	- "_" -> (game.end_turn, ())
	'''
	split = s.split(' ')
	if len(split) == 4:
		if split[0] == 'p':
			return game.add_soldier, (int(split[1]), int(split[2]), SOLDIER_LEGEND[split[3]])
		elif split[0] == 'm' or split[0] == 'a':
			return (game.move_soldier if split[0] == 'm' else game.attack_soldier), (int(split[1]), int(split[2]), int(split[3]))
	elif len(split == 1):
		if split[0] == '_':
			return (game.end_turn, ())
	raise ValueError(f'Failed to parse the string "{s}" as an action.')

def get_adjacent_position(x, y, d):
	if d == 0:
		return x+1,y
	elif d == 1:
		return x,y+1
	elif d == 2:
		return x-1,y
	else:
		return x,y-1

def count_lines(fpath):
	with open(fpath, 'r') as f:
		for i,_ in enumerate(f):
			pass
		return i

def obstacles_to_str(obstacles):
	'''
	When obstacles is a list of lists (in native Python), use this to format it nicely as a string.
	'''
	return '\n'.join(map(' '.join,(map('-#'.__getitem__,map(int,a)) for a in obstacles)))

def total_scores_to_str(total_scores):
	score_sum = sum(total_scores)
	return ' - '.join(
		map(str,total_scores)
	) + ' [' + ' - '.join(
		f'{int(0.5+s*100./score_sum):2d}%'
		for s in total_scores
	) + ']'
