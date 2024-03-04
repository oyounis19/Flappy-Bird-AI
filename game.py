import pygame
import random
import numpy as np
from model import NeuralNetwork as NN

class Game:
    def __init__(self, speed=9 , fps=30, pipe_width=5, pipe_height_range=(50,400), num_pipes=3, gap_size=150, color=(0,255,0), width=800, height=600, jump_size=12):
        self.speed = speed
        self.fps = fps
        self.pipe_width = pipe_width
        self.pipe_height_range = pipe_height_range
        self.gap_size = gap_size
        self.color = color
        self.width, self.height = width, height
        self.jump_size = jump_size
        self.pipe_distance = width // num_pipes + 100
        self.num_pipes = num_pipes
        self.bird_width = 36
        self.bird_height = 27
        self.bird_x = 100 # bird's x position
        self.inited = False
        self.high_score = 0

    def init_screen(self) -> None:
        '''
        Initializes game variables.
        Must run before evaluation.
        '''
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        bg = pygame.image.load("assets/background.jpg")
        self.bg = pygame.transform.scale(bg, (self.width, self.height))
        bird_original = pygame.image.load("assets/bird.png")
        self.bird_original = pygame.transform.scale(bird_original, (self.bird_width, self.bird_height))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Flappy Bird")
        self.inited = True

    def __init_vars(self):
        pipes = [{'x': self.width, 'width': self.pipe_width, 'height': random.randint(*self.pipe_height_range)}]
        return pipes, 0, True

    def __rotate_bird(self, angle, velocity):
        bird = self.bird_original.copy()
        rotation_angle = angle if velocity < 0 else -angle
        bird = pygame.transform.rotate(bird, rotation_angle)
        return bird
    
    def __draw_pipes(self, pipes, score):
        for pipe in pipes:
            pygame.draw.rect(self.screen, self.color, [pipe['x'], 0, pipe['width'], pipe['height']])
            pygame.draw.rect(self.screen, self.color, [pipe['x'], pipe['height'] + self.gap_size, pipe['width'], pipe['height'] + self.gap_size + self.height])
            pipe['x'] -= self.speed
            if pipe['x'] <= -self.pipe_width:
                pipes.remove(pipe)
                score += 1
        return score
    
    def __append_pipe(self, pipes):
        if pipes[-1]['x'] <= self.width - self.pipe_distance:
                pipes.append({'x': self.width, 'width': self.pipe_width, 'height': random.randint(*self.pipe_height_range)})
        return pipes
    
    def __check_collided(self, pipe, bird_y):
        '''
        Checks for collision.
        '''
        if self.bird_x < pipe['x'] < self.bird_x + self.bird_width:
                if bird_y < pipe['height'] or bird_y + self.bird_height > pipe['height'] + self.gap_size:
                    return True
        return False
    
    def __off_screen(self, bird_y):
        '''
        Check if bird went out of the game frame.
        '''
        if bird_y < 0 or bird_y + self.bird_height > self.height:
                return True
        return False

    def __init_birds(self, pop: list):
        ''' initializes the birds.'''
        birds = []
        for nn in pop:
            birds.append({'x': self.bird_x, 'y': self.height // 2, 'velocity': 0, 'nn': nn, 'fitness': 0, 'collided': False})
        return birds
    
    def evaluate_population(self, pop: list, epoch: int, thresh: float = 0.5):
        ''' trains birds populations and returns the fitness scores.'''
        if self.inited == False:
            print("Game not initialized. Call init_screen() first."); return ValueError
        
        # Initialize variables
        pipes, score, running = self.__init_vars()
        birds = self.__init_birds(pop)
        alive_birds: int = len(birds)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Draw background
            self.screen.blit(self.bg, (0, 0))

            # move birds & check for collision
            for i, bird in enumerate(birds):
                if bird['collided']:
                    continue

                # Inputs for the neural network
                bird_y = bird['y']
                distance_to_pipe = pipes[0]['x'] - bird['x']
                height_of_gap = self.gap_size
                bird_velocity = bird['velocity']
                gap_center = pipes[0]['height'] + (self.gap_size // 2)

                # Get the jump decision from the neural network
                jump = bird['nn'].forward(np.array([bird_y, distance_to_pipe, height_of_gap, bird_velocity, gap_center]), thresh)

                # Move the bird
                bird['velocity'] = -self.jump_size if jump == 1 else bird['velocity']

                # Update bird's position
                bird['y'] += bird['velocity'] # move bird up or down
                bird['velocity'] += 1 # gravity

                # rotate the bird
                bird_img = self.__rotate_bird(45, bird['velocity'])

                # Draw Birds
                self.screen.blit(bird_img, (bird['x'], bird['y']))

                # Check for collision & off screen
                if self.__check_collided(pipes[0], bird['y']) or \
                    self.__off_screen(bird['y']):
                    bird['collided'] = True # flag the bird as collided
                    alive_birds -= 1
                    if alive_birds == 0:
                        running = False

                # Update fitness if bird passed the pipe
                elif bird['x'] > pipes[0]['x'] + pipes[0]['width'] and bird['x'] < pipes[0]['x'] + pipes[0]['width'] + self.speed:
                    bird['fitness'] += 1

            # Draw Pipes & Update High Score
            score = self.__draw_pipes(pipes, score)
            self.high_score = max(self.high_score, score)

            # Add new pipe
            pipes = self.__append_pipe(pipes)

            # Show score & generation number
            font = pygame.font.Font(None, 36)
            text_score = font.render(f"Score: {score}", True, (255, 255, 255))
            text_high_score = font.render(f"High Score: {self.high_score}", True, (255, 255, 255))
            text_epoch = font.render(f"Generation: {epoch}", True, (255, 255, 255))
            text_alive = font.render(f"Alive: {alive_birds}", True, (255, 255, 255))
            self.screen.blit(text_score, (10, 10))
            self.screen.blit(text_epoch, (10, 50))
            self.screen.blit(text_high_score, (10, 90))
            self.screen.blit(text_alive, (10, 130))
            self.clock.tick(self.fps)
            pygame.display.update()

        return np.array([bird['fitness'] for bird in birds])

    def run(self, nn: NN, episode: int):
        '''
        Runs the game with a neural network for testing.
        '''
        self.init_screen()
        # Initialize variables
        pipes, score, running = self.__init_vars()
        bird_y = self.height // 2
        bird_velocity = 0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Draw background
            self.screen.blit(self.bg, (0, 0))

            # Inputs for the neural network
            distance_to_pipe = pipes[0]['x'] - self.bird_x
            height_of_gap = self.gap_size
            gap_center = (pipes[0]['height'] + (pipes[0]['height'] + self.gap_size)) // 2

            # Get the jump decision from the neural network
            jump = nn.forward(np.array([bird_y, distance_to_pipe, height_of_gap, bird_velocity, gap_center]), 0.5)

            # Move the bird
            bird_velocity = -self.jump_size if jump == 1 else bird_velocity # jump

            # Update bird's position
            bird_y += bird_velocity
            bird_velocity += 1 # gravity

            # Rotate the bird, draw pipes & update score
            bird = self.__rotate_bird(45, bird_velocity)
            self.screen.blit(bird, (self.bird_x, bird_y))
            score = self.__draw_pipes(pipes, score)
            self.high_score = max(self.high_score, score)
            pipes = self.__append_pipe(pipes)

            # Check for collision & off screen
            if self.__check_collided(pipes[0], bird_y) or self.__off_screen(bird_y):
                print(f"Game Over! Score: {score}")
                running = False

            # Show score & high score
            font = pygame.font.Font(None, 36)
            text_score = font.render(f"Score: {score}", True, (255, 255, 255))
            text_episode = font.render(f"Episode: {episode+1}", True, (255, 255, 255))
            self.screen.blit(text_score, (10, 10))
            self.screen.blit(text_episode, (10, 50))
            self.clock.tick(self.fps)
            pygame.display.update()

        pygame.quit()

    def quit(self):
        ''' quits the game. '''
        pygame.quit()

    def run_human(self):
        ''' 
        NOT USED
        Runs the game by human.
        '''
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        bg = pygame.image.load("flappy bird/assets/background.jpg")
        bg = pygame.transform.scale(bg, (self.width, self.height))
        bird_original = pygame.image.load("flappy bird/assets/bird.png")
        bird_original = pygame.transform.scale(bird_original, (self.bird_width, self.bird_height))
        bird = bird_original
        clock = pygame.time.Clock()
        pygame.display.set_caption("Flappy Bird")
        pipes = [{'x': self.width, 'width': self.pipe_width, 'height': random.randint(*self.pipe_height_range)}]
        bird_y = self.height // 2
        bird_velocity = 0
        score = 0
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        bird_velocity = -self.jump_size # jump

            bird_y += bird_velocity
            bird_velocity += 1

            # Rotate the bird
            bird = bird_original.copy()  # Create a copy of the original bird surface
            rotation_angle = 45 if bird_velocity < 0 else -45
            bird = pygame.transform.rotate(bird, rotation_angle)

            screen.blit(bg, (0, 0))
            screen.blit(bird, (self.bird_x, bird_y))

            for pipe in pipes:
                pygame.draw.rect(screen, self.color, [pipe['x'], 0, pipe['width'], pipe['height']])
                pygame.draw.rect(screen, self.color, [pipe['x'], pipe['height'] + self.gap_size, pipe['width'], pipe['height'] + self.gap_size + self.height])
                pipe['x'] -= self.speed

                if pipe['x'] <= -self.pipe_width:
                    pipes.remove(pipe)
                    score += 1

            if pipes[-1]['x'] <= self.width - self.pipe_distance:
                pipes.append({'x': self.width, 'width': self.pipe_width, 'height': random.randint(*self.pipe_height_range)})

            if self.bird_x < pipes[0]['x'] < self.bird_x + self.bird_width: # collision
                if bird_y < pipes[0]['height'] or bird_y + self.bird_height > pipes[0]['height'] + self.gap_size:
                    # reset the game
                    bird_y = self.height // 2
                    bird_velocity = 0
                    pipes = [{'x': self.width, 'width': self.pipe_width, 'height': random.randint(*self.pipe_height_range)}]
                    score = 0

            if bird_y < 0 or bird_y + self.bird_height > self.height: # off screen
                # reset the game
                bird_y = self.height // 2
                bird_velocity = 0
                pipes = [{'x': self.width, 'width': self.pipe_width, 'height': random.randint(*self.pipe_height_range)}]
                score = 0
        
            # Show score
            font = pygame.font.Font(None, 36)
            text = font.render(f"Score: {score}", True, (255, 255, 255))
            screen.blit(text, (10, 10))
            pygame.display.update()
            clock.tick(self.fps)
        pygame.quit()
