

from game import TableTactics, RandomSim
from dql.data import heuristic_score


def print_heuristic():
    print(f'Heuristic score: {heuristic_score(game)}')

game = TableTactics()
sim = RandomSim(game, step_callbacks=[print_heuristic])

sim.run()


