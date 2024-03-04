import numpy as np
from game import Game
from model import NeuralNetwork as NN

"""
Steps:
1. Create a population of N elements
2. While the termination condition is not met, do:
    1. Evaluate the population
    2. Select parents
    3. Crossover parents
    4. Mutate offspring
    5. add offspring to the population
3. Return the best solution
"""

class GeneticAlgorithm:
    def __init__(self, population_size: int, nn: NN, game: Game, mutation_rate=0.1, crossover_probability=0.9, random_seed=None)-> None:
        self.population_size = population_size
        self.nn = nn
        self.game = game
        self.mu = mutation_rate
        self.cp = crossover_probability
        self.random_seed = random_seed

    def __init_population(self, hs=8) -> list[NN]:
        '''
        Initialize population of specified size.
        '''
        pop = [self.nn(hidden_size=hs) for _ in range(self.population_size)] # Initializes bird population with the default Neural network params.
        return pop

    def __crossover(self, parent1: NN, parent2: NN)-> tuple[NN, NN]:
        '''
        Uniform crossover
        '''
        c1 = parent1.copy()
        c2 = parent2.copy()
        for i in range(len(c1.weights)):
            mask = np.random.rand(*c1.weights[i].shape) <= self.cp
            c1.weights[i] = mask * c1.weights[i] + (1 - mask) * c2.weights[i]
            c2.weights[i] = mask * c2.weights[i] + (1 - mask) * c1.weights[i]
        return c1, c2

    def __mutate(self, nn: NN) -> NN:
        '''
        Mutate the neural network's weights by adding Gaussian noise to the weights.
        '''
        for i in range(len(nn.weights)):
            mask = np.random.rand(*nn.weights[i].shape) <= self.mu
            nn.weights[i] += mask * np.random.randn(*nn.weights[i].shape) # this either adds a random number from weights or leaves it as it is
        return nn
    
    def __select_parents(self, fitness_scores: np.ndarray, pop: list[NN]) -> np.ndarray:
        '''
        Select parents based on fitness scores by using roulette wheel selection algorithm.
        param fitness_scores: fitness scores of the population
        param num_parents: number of parents to select
        param pop: population
        return: selected parents
        '''
        if np.sum(fitness_scores) == 0:
            probabilities = np.ones_like(fitness_scores) / len(fitness_scores)
        else:
            probabilities = fitness_scores / np.sum(fitness_scores)
        
        return np.random.choice(pop, size=2, p=probabilities, replace=True)    

    def __get_best_network(self, fitness_scores, pop)-> NN:
        '''
        Returns the neural network with the highest fitness score.
        '''
        # Check if this generation has the best neural network ever found
        if np.max(fitness_scores) > self.best_nn[1]:
            self.best_nn = (pop[np.argmax(fitness_scores)], np.max(fitness_scores))

    def train(self, epochs=1000, max_score = None, hidden_size=8, thresh=0.5)-> NN:
        '''
        Runs the whole thing.
        param epochs: number of generations
        param max_score: maximum score to break the loop
        param hidden_size: number of nodes in the hidden layer of the neural network
        param thresh: threshold for the output of the neural network
        return: best neural network
        '''
        try:
            if self.random_seed is not None:
                np.random.seed(self.random_seed)

            self.game.init_screen()
            pop = self.__init_population(hs=hidden_size)

            self.best_nn = (None, -1) # (best neural network, best score)
            for epoch in range(epochs):
                # Evaluate
                fitness_scores = self.game.evaluate_population(pop, epoch, thresh)

                # select parents
                parents = self.__select_parents(fitness_scores, pop) # select 2 parents

                # crossover
                c1, c2 = self.__crossover(parents[0], parents[1])

                # mutate
                c1, c2 = self.__mutate(c1), self.__mutate(c2)

                # Add children to the population
                pop.append(c1)
                pop.append(c2)

                # get best network ever found
                self.__get_best_network(fitness_scores, pop)

                # Show iteration information
                print(f'Epoch {epoch}: Best Score = {fitness_scores.max()}')

                if max_score and fitness_scores.max() > max_score:
                    print('breaking')
                    break

        except Exception as e:
            print(e)

        finally:
            self.game.quit()

        return self.best_nn[0]

    def test(self, nn: NN, num_episodes=10):
        '''
        Test the neural network.
        param nn: neural network
        param num_episodes: number of episodes to run
        '''
        for i in range(num_episodes):
            self.game.run(nn, i)
        self.game.quit()