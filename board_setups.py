
from copy import deepcopy
from .enums import SoldierType


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

def get_board_setup(name):
	if name not in board_setups:
		raise ValueError(f'Unrecognized board setup name: {name}')
	return board_setups[name]

