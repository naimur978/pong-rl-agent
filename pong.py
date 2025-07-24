import pygame
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Constants
TRAINING = False
MAX_SCORE_TRAINING = 5
MAX_SCORE_PLAY = 11
FPS_TRAINING = 480
FPS_PLAY = 60
COLOR = (139, 23, 137)
WHITE = (255, 255, 255)
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 100
SCREEN_WIDTH = 840
SCREEN_HEIGHT = 600
PADDLE_SPEED = 10
BALL_RADIUS = 10
SIMPLE_AI_SPEED = 7


class Ball(pygame.sprite.Sprite):
    def __init__(self, color, radius):
        super().__init__()
        self.image = pygame.Surface((radius * 2, radius * 2))
        self.image.fill(WHITE)
        self.image.set_colorkey(WHITE)
        self.radius = radius
        self.velocity = [np.random.randint(2, 5), np.random.randint(-4, 5)]
        pygame.draw.circle(self.image, color, (radius, radius), radius)
        self.rect = self.image.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        
    def update(self):
        self.rect.centerx += self.velocity[0]
        self.rect.centery += self.velocity[1]
    
    def bounce(self):
        self.velocity[0] = -self.velocity[0]
        self.velocity[1] = np.random.uniform(-4, 5)


class Paddle(pygame.sprite.Sprite):
    def __init__(self, color, name, alpha=0.4, gamma=0.7, epsilon_decay=0.00001, epsilon_min=0.01, epsilon=1):
        super().__init__()
        self.image = pygame.Surface((PADDLE_WIDTH, PADDLE_HEIGHT))
        self.image.fill(WHITE)
        self.image.set_colorkey(WHITE)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}
        self.rewards, self.episodes, self.average = [], [], []
        self.name = name
        pygame.draw.rect(self.image, color, [0, 0, PADDLE_WIDTH, PADDLE_HEIGHT])
        self.rect = self.image.get_rect()
    
    def move_up(self, pixels):
        self.rect.y -= pixels
        self.rect.y = max(self.rect.y, 0)
          
    def move_down(self, pixels):
        self.rect.y += pixels
        self.rect.y = min(self.rect.y, SCREEN_HEIGHT - PADDLE_HEIGHT)

    def simple_ai(self, ball_pos_y, pixels):
        if ball_pos_y + BALL_RADIUS > self.rect.y + PADDLE_HEIGHT / 2:
            self.rect.y += pixels
        if ball_pos_y + BALL_RADIUS < self.rect.y + PADDLE_HEIGHT / 2:
            self.rect.y -= pixels
        self.rect.y = max(0, min(self.rect.y, SCREEN_HEIGHT - PADDLE_HEIGHT))
    
    def epsilon_greedy(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * (1 - self.epsilon_decay))
    
    def get_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(3)
        if TRAINING:
            self.epsilon_greedy()
            if np.random.uniform() < self.epsilon:
                action = np.random.choice(3)
            else:
                action = np.argmax(self.q_table[state])
        else:
            action = np.argmax(self.q_table[state])
        return action
    
    def update_q_table(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(3)
        td_target = reward + self.gamma * np.max(self.q_table[next_state])
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
    
    def save(self, episode):
        with open(f'player_{self.name}_{episode}_qtable.pkl', 'wb') as file:
            pickle.dump(self.q_table, file)

    def load(self, name):
        with open(name, 'rb') as file:
            self.q_table = pickle.load(file)

    def plot_model(self, reward, episode):
        self.rewards.append(reward)
        self.episodes.append(episode)
        self.average.append(sum(self.rewards) / len(self.rewards))
        plt.plot(self.episodes, self.average, 'r')
        plt.plot(self.episodes, self.rewards, 'b')
        plt.ylabel('Reward', fontsize=18)
        plt.xlabel('Games', fontsize=18)
        try:
            plt.savefig(f'player_{self.name}_evolution.png')
        except OSError as e:
            print(f"Error saving file: {e}")


class Game:
    def __init__(self, player_a, player_b):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("QL Pong")
        self.paddle_a = player_a
        self.paddle_a.rect.topleft = (0, (SCREEN_HEIGHT - PADDLE_HEIGHT) // 2)
        self.paddle_b = player_b
        self.paddle_b.rect.topleft = (SCREEN_WIDTH - PADDLE_WIDTH, (SCREEN_HEIGHT - PADDLE_HEIGHT) // 2)
        self.ball = Ball(COLOR, BALL_RADIUS)
        self.all_sprites = pygame.sprite.Group(self.paddle_a, self.paddle_b, self.ball)
        self.clock = pygame.time.Clock()
        self.finish = False
        self.score_a, self.score_b = 0, 0
        self.reward = 0
        
    def get_reward(self):
        max_reward = PADDLE_HEIGHT // 2
        min_reward = -max_reward
        y_distance = abs(self.paddle_a.rect.centery - self.ball.rect.centery)
        reward = - (y_distance / SCREEN_HEIGHT) * max_reward
        if y_distance < PADDLE_HEIGHT // 2:
            reward += max_reward  
        return max(min_reward, reward)
    
    def distille_state(self):
        if (self.paddle_a.rect.centery - BALL_RADIUS <= self.ball.rect.centery <= self.paddle_a.rect.y + BALL_RADIUS):
            distilled_state = 0 
        elif self.ball.rect.centery < self.paddle_a.rect.centery:
            distilled_state = 1 
        else:
            distilled_state = 2 
        return distilled_state
    
    def play(self):
        global TRAINING
        action_a = 0
        while not self.finish:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.finish = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.finish = True
            distilled_state = self.distille_state()
            self.state = (
                distilled_state,
                action_a
            )
            reward_a = 0
            action_a = self.paddle_a.get_action(self.state)
            self.paddle_b.simple_ai(self.ball.rect.y, SIMPLE_AI_SPEED)
            if action_a == 1:
                self.paddle_a.move_up(PADDLE_SPEED)
            elif action_a == 2:
                self.paddle_a.move_down(PADDLE_SPEED)
            self.ball.update()
            if TRAINING:
                reward_a = self.get_reward()
            if pygame.sprite.spritecollide(self.ball, [self.paddle_a], False):
                self.ball.bounce()
            if pygame.sprite.spritecollide(self.ball, [self.paddle_b], False):
                self.ball.bounce()
            if self.ball.rect.x > SCREEN_WIDTH:
                self.ball.rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
                self.ball.velocity[0] *= -1
                self.score_a += 1
            elif self.ball.rect.x < 0:
                self.ball.rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
                self.ball.velocity[0] *= -1
                self.score_b += 1
            if self.ball.rect.y > SCREEN_HEIGHT - 2 * BALL_RADIUS or self.ball.rect.y < 0:
                self.ball.velocity[1] *= -1
            if TRAINING:
                next_distilled_state = self.distille_state()
                next_state = (
                    next_distilled_state,
                    action_a
                )
                self.paddle_a.update_q_table(self.state, action_a, reward_a, next_state)
            self.screen.fill(WHITE)
            pygame.draw.line(self.screen, COLOR, [SCREEN_WIDTH // 2, 0], [SCREEN_WIDTH // 2, SCREEN_HEIGHT], 5)
            self.all_sprites.draw(self.screen)
            font = pygame.font.Font(None, 74)
            text_a = font.render(str(self.score_a), 1, COLOR)
            self.screen.blit(text_a, (SCREEN_WIDTH // 4, 10))
            text_b = font.render(str(self.score_b), 1, COLOR)
            self.screen.blit(text_b, (3 * SCREEN_WIDTH // 4, 10))
            self.all_sprites.update()
            pygame.display.flip()
            self.clock.tick(FPS_TRAINING if TRAINING else FPS_PLAY)
            if TRAINING:
                self.reward += reward_a
            if self.score_a == MAX_SCORE_TRAINING or self.score_b == MAX_SCORE_TRAINING:
                self.finish = True
                pygame.quit()


if __name__ == "__main__":
    player_b = Paddle(COLOR, "B")
    if TRAINING:
        player_a = Paddle(COLOR, "A")
        for i in range(501):
            game = Game(player_a, player_b)
            game.play()
            if i % 10 == 0:
                player_a.save(i)
            player_a.plot_model(game.reward, i)
            print(f"Game: {i}, Epsilon A: {player_a.epsilon}, Score A: {game.score_a}, Rewards A: {game.reward}")
    else:
        player_a = Paddle(COLOR, "A")
        player_a.load('player_A_250_qtable.pkl')
        game = Game(player_a, player_b)
        game.play()
