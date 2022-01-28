
DIRECTION_NAMES = ['right', 'up', 'left', 'down'] # "up" and "down" may be flipped if the array is printed directly, as this assumes positive y direction means pointing upward

def action_to_str(f, a):
	# Actions are pairs of bound methods and possible arguments.  To prevent print from spamming the screen, use this method instead of print(va).
	return '({}, ({}))'.format(
		f.__name__,
		', '.join(map(str,a))
	)

def actions_to_str(va, sep='\n'):
	return sep.join(
		action_to_str(f, a)
		for f, a in va
	)

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

