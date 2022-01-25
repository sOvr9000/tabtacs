'''
Standard attributes for each SoldierType.
'''

from .enums import SoldierType

soldier_prototypes = {
	SoldierType.Noble: {
		'hitpoints': 4,
		'speed': 3,
	},
	SoldierType.Fighter: {
		'hitpoints': 3,
		'speed': 3,
	},
	SoldierType.Thief: {
		'hitpoints': 2,
		'speed': 4,
	},
}

def load_soldier_prototypes(prototypes):
	'''
	If the predefined settings aren't preferable, then use another one!
	'''
	global soldier_prototypes
	soldier_prototypes = prototypes

def get_soldier_hitpoints(soldier_type):
	return soldier_prototypes[soldier_type]['hitpoints']

def get_soldier_speed(soldier_type):
	return soldier_prototypes[soldier_type]['speed']

def get_soldier_actions(soldier_type):
	return 2 # yup
