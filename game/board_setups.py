
from copy import deepcopy
from random import choice
from .enums import SoldierType



def get_board_setup(name):
	if name not in board_setups:
		raise ValueError(f'Unrecognized board setup name: {name}')
	return board_setups[name]

def copy_board_setup(name):
	return deepcopy(get_board_setup(name))

def set_board_setup(name, setup):
	if name in board_setups:
		raise ValueError(f'A board setup already exists under the name \'{name}\'')
	board_setups[name] = setup

def random_obstacles(board_size, num_obstacles = 6):
	w,h = board_size
	wh = w*h
	obstacles = [[False]*w for _ in range(h)]
	open_set = [(x,y) for y in range(h) for x in range(w)]
	total = 0

	def is_connected():
		for y in range(h):
			for x in range(w):
				if not obstacles[y][x]:
					arr = deepcopy(obstacles)
					flood_fill(x,y,arr)
					return sum(map(sum,arr)) == wh
	
	def flood_fill(x,y,arr):
		if not arr[y][x]:
			arr[y][x] = True
			if x+1 < w:
				flood_fill(x+1,y,arr)
			if x > 0:
				flood_fill(x-1,y,arr)
			if y+1 < h:
				flood_fill(x,y+1,arr)
			if y > 0:
				flood_fill(x,y-1,arr)

	while len(open_set) > 0 and total < num_obstacles:
		p = choice(open_set)
		open_set.remove(p)
		obstacles[p[1]][p[0]] = True
		if not is_connected():
			obstacles[p[1]][p[0]] = False
		total += 1

	return obstacles


board_setups = {
	'standard': {
		'board_size': (6,6),
		'placement_space': [
			[(0,0),(1,0),(0,1),(1,1)],
			[(5,5),(4,5),(5,4),(4,4)],
		],
		'soldiers': [
			{
				SoldierType.Fighter: 2,
				SoldierType.Thief: 1,
			},
			{
				SoldierType.Fighter: 2,
				SoldierType.Thief: 1,
			},
		]
	},
}

board_setups['tweaked'] = deepcopy(board_setups['standard'])
board_setups['tweaked']['soldiers'][1][SoldierType.Fighter] = 1
board_setups['tweaked']['soldiers'][1][SoldierType.Thief] = 2

