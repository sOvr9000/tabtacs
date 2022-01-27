
import numpy as np
from copy import deepcopy
from .enums import SoldierType
from .taclib import get_adjacent_position
from .prototypes import get_soldier_hitpoints, get_soldier_speed, get_soldier_actions
from .board_setups import get_board_setup




class TableTactics:
	SOLDIER_DATA_SLICE = slice(1,5)
	def __init__(self, setup = None):
		if setup is None:
			setup = 'standard'
		if isinstance(setup, str):
			setup = get_board_setup(setup)
		self.setup = deepcopy(setup)
		self.board = -np.ones((*self.setup['board_size'], 5), dtype=int)
		# board array dimensions: (tile position Y, tile position X, tile data)
		# tile data index 0 corresponds to obstacle existence (-1 for nonexistent, 1 otherwise)
		# index 1 corresponds to army belongingness (-1 for None, and 0 to n-1 for each player)
		# index 2 corresponds to soldier type (-1 if no soldier, etc.)
		# index 3 corresponds to current soldier hitpoints (-1 if no soldier)
		# index 4 corresponds to soldier actions remaining for the current turn (-1 if no soldier)
		if not all(self.is_valid_position(x,y) for space in self.setup['placement_space'] for x,y in space):
			raise ValueError(f'Invalid position in the placement space in the provided setup.')
		self.num_armies = len(self.setup['placement_space'])
		self.placement_mask = -np.ones(self.setup['board_size'])
		for army, space in enumerate(self.setup['placement_space']):
			for x,y in space:
				self.placement_mask[y, x] = army - 1
		for sr in self.setup['soldiers']:
			sr[SoldierType.Noble] = 1
		self.reset()
	def reset(self):
		'''
		Clear the pieces on the board.
		'''
		self.board[:,:,self.SOLDIER_DATA_SLICE] = -1 # Set all soldier information to None, and keep the obstacle configuration (the "map")
		self.last_action_location_x = -1
		self.last_action_location_y = -1
		self.current_steps_remaining = 0
		self.turn = 0
		sr = {
			soldier_type: 0
			for soldier_type in SoldierType
		}
		self.soldiers_remaining = [deepcopy(sr) for _ in range(self.num_armies)]
		self.setup_phase = True
	def is_valid_position(self, x, y):
		return x >= 0 and y >= 0 and x < self.board.shape[1] and y < self.board.shape[0]
	def is_position_unoccupied(self, x, y):
		return np.all(self.board[y, x] == -1)
	def is_soldier(self, x, y):
		return self.get_army(x, y) >= 0
	def is_obstacle(self, x, y): # alias?
		return self.get_obstacle(x, y)
	def get_obstacle(self, x, y):
		return self.board[y, x, 0] == 1 # return bool type
	def get_army(self, x, y):
		return self.board[y, x, 1]
	def get_soldier_type(self, x, y):
		return self.board[y, x, 2]
	def get_soldier_hitpoints_remaining(self, x, y):
		return self.board[y, x, 3]
	def get_soldier_actions_remaining(self, x, y):
		return self.board[y, x, 4]
	def get_soldiers_remaining(self, army, soldier_type=None):
		if soldier_type is None:
			return sum(self.soldiers_remaining[army].values()) # all types
		return self.soldiers_remaining[army][soldier_type]
	def reset_actions_remaining(self, x, y):
		self.board[y, x, 4] = get_soldier_actions(self.get_soldier_type(x, y))
	def decrement_soldier_actions_remaining(self, x, y):
		self.board[y, x, 4] -= 1
	def decrement_soldier_hitpoints_remaining(self, x, y):
		self.board[y, x, 3] -= 1
		if self.get_soldier_hitpoints_remaining(x, y) <= 0:
			# PURGE HIM FROM EXISTENCE
			self.remove_soldier(x, y)
	def remove_soldier(self, x, y):
		soldier_type = self.get_soldier_type(x, y)
		army = self.get_army(x, y)
		self.soldiers_remaining[army][soldier_type] -= 1
		self.board[y, x, self.SOLDIER_DATA_SLICE] = -1
		if soldier_type == SoldierType.Noble:
			for _x, _y in self.soldiers_of_army(army):
				self.remove_soldier(_x, _y)
	def end_turn(self):
		for x,y in self.soldiers_of_army(self.turn):
			self.reset_actions_remaining(x, y)
		self.turn = (self.turn + 1) % self.num_armies
		if self.setup_phase and self.turn == 0:
			self.setup_phase = False
	def can_add_soldier(self, x, y, soldier_type):
		return \
			self.setup_phase and \
			((soldier_type == SoldierType.Noble) ^ (self.get_soldiers_remaining(self.turn, SoldierType.Noble) > 0)) and \
			(x,y) in self.get_placement_space(self.turn) and \
			self.is_position_unoccupied(x, y) and \
			self.get_soldiers_remaining(self.turn, soldier_type) < self.get_soldier_composition(self.turn)[soldier_type]
	def can_move_soldier(self, x, y, d):
		nx, ny = get_adjacent_position(x, y, d)
		return \
			not self.setup_phase and \
			self.is_valid_position(x, y) and self.is_valid_position(nx, ny) and \
			(
				self.get_soldier_actions_remaining(x, y) > 0
				or
				(self.current_steps_remaining > 0 and x == self.last_action_location_x and y == self.last_action_location_y)
			) and \
			not self.get_obstacle(nx, ny) and self.get_army(x, y) == self.turn and self.is_position_unoccupied(nx, ny)
	def can_attack_soldier(self, x, y, d):
		if self.setup_phase:
			return False
		nx, ny = get_adjacent_position(x, y, d)
		if not self.is_valid_position(x, y) or not self.is_valid_position(nx, ny):
			return False
		a = self.get_army(x, y)
		na = self.get_army(nx, ny)
		return self.get_soldier_actions_remaining(x, y) > 0 and a == self.turn and na != -1 and a != na
	def add_soldier(self, x, y, soldier_type):
		if not self.can_add_soldier(x, y, soldier_type):
			raise Exception(f'Cannot place {SoldierType(soldier_type)} at {x,y}.')
		self.board[y, x, self.SOLDIER_DATA_SLICE] = self.turn, soldier_type, get_soldier_hitpoints(soldier_type), get_soldier_actions(soldier_type)
		self.soldiers_remaining[self.turn][soldier_type] += 1
	def move_soldier(self, x, y, d):
		if not self.can_move_soldier(x, y, d):
			raise Exception(f'Cannot move soldier at {x,y} in direction {d}.')
		if x != self.last_action_location_x or y != self.last_action_location_y: # If a different soldier moves or attacks from the last action by the army...
			self.current_steps_remaining = 0
		if self.current_steps_remaining == 0: # This can also be caused by completing a full "move" action on a soldier, and then using another action to perform another "move" action.
			self.decrement_soldier_actions_remaining(x, y)
			self.current_steps_remaining = get_soldier_speed(self.get_soldier_type(x, y))
		nx, ny = get_adjacent_position(x, y, d)
		self.board[ny, nx, self.SOLDIER_DATA_SLICE] = self.board[y, x, self.SOLDIER_DATA_SLICE]
		self.board[y, x, self.SOLDIER_DATA_SLICE] = -1
		self.current_steps_remaining -= 1
		self.last_action_location_x = nx
		self.last_action_location_y = ny
	def attack_soldier(self, x, y, d):
		if not self.can_attack_soldier(x, y, d):
			raise Exception(f'Cannot attack a soldier in direction {d} from {x,y}.')
		nx, ny = get_adjacent_position(x, y, d)
		self.decrement_soldier_actions_remaining(x, y)
		self.decrement_soldier_hitpoints_remaining(nx, ny)
	def is_game_over(self):
		return \
			(not self.setup_phase or sum(len(list(self.unoccupied_tiles(self.get_placement_space(a)))) for a in range(self.num_armies)) == 0) and \
			sum(sum(sr.values()) > 0 for sr in self.soldiers_remaining) <= 1
	def get_placement_space(self, army):
		return self.setup['placement_space'][army]
	def get_soldier_composition(self, army):
		return self.setup['soldiers'][army]
	def valid_actions(self, include_end_turn = True):
		if self.is_game_over():
			return
		if include_end_turn:
			yield self.end_turn, ()
		for y in range(self.board.shape[0]):
			for x in range(self.board.shape[1]):
				if self.setup_phase:
					for soldier_type in SoldierType:
						if self.can_add_soldier(x, y, soldier_type):
							yield self.add_soldier, (x, y, soldier_type)
				else:
					for d in range(4):
						if self.can_move_soldier(x, y, d):
							yield self.move_soldier, (x, y, d)
						if self.can_attack_soldier(x, y, d):
							yield self.attack_soldier, (x, y, d)
	def unoccupied_tiles(self, tile_set = None):
		'''
		Iterate over the unoccupied positions on the board.

		If tile_set is a list, tuple, or any other generator of tile positions, then iterate over only those which are unoccupied.
		'''
		if tile_set is None:
			for y in range(self.board.shape[0]):
				for x in range(self.board.shape[1]):
					if self.is_position_unoccupied(x, y):
						yield x, y
		else:
			for x, y in tile_set:
				if self.is_position_unoccupied(x, y):
					yield x, y
	def soldiers_of_army(self, army):
		for y in range(self.board.shape[0]):
			for x in range(self.board.shape[1]):
				if self.get_army(x, y) == army:
					yield x, y
	def __repr__(self, spacing = 6):
		D = self.board.shape[1] * 2 - 1 + spacing
		s = ''
		for t in ('Obstacles','Armies','Soldier Types','Soldier HP','Actions Remaining'):
			s += t + ' ' * (D - len(t))
		for y in range(self.board.shape[0]):
			s += '\n'
			for n in range(self.board.shape[2]):
				if n > 0:
					s += ' ' * spacing
				for x in range(self.board.shape[1]):
					if x > 0:
						s += ' '
					s += str(self.board[y,x,n]) if self.board[y,x,n] >= 0 else '-'
		return s


