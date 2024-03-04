from game import Game
from model import NeuralNetwork
from GA import GeneticAlgorithm as GA
import warnings
warnings.filterwarnings("ignore") # to ignore RuntimeWarning by exp function in numpy

game = Game(jump_size=8)

ga = GA(population_size=100,
        mutation_rate=0.15,
        nn=NeuralNetwork,
        game=game,
        crossover_probability=0.85
        )

# Training
best_nn = ga.train(epochs=200, max_score=800, hidden_size=8, thresh=0.5)