import numpy as np
import pygame
import random
import time
from pygame.locals import *

# Configuración del entorno
GRID_SIZE = 10
CELL_SIZE = 60
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE + 150  # Más espacio para información

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)

# Configuración Q-Learning
ACTIONS = [0, 1, 2, 3]  # 0: arriba, 1: derecha, 2: abajo, 3: izquierda
ALPHA = 0.1  # Tasa de aprendizaje
GAMMA = 0.9  # Factor de descuento
EPSILON = 0.9  # Mayor exploración para ambientes cambiantes

# Parámetros del ambiente aleatorio
NUM_BOXES = 5
NUM_OBSTACLES = 15

class WarehouseEnv:
    def __init__(self):
        # No inicializamos posiciones fijas, se generarán aleatoriamente
        self.reset()
    
    def generate_random_positions(self, count, exclude_positions=[]):
        """Genera posiciones aleatorias que no se solapen"""
        positions = []
        attempts = 0
        max_attempts = count * 10
        
        while len(positions) < count and attempts < max_attempts:
            pos = [random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)]
            
            # No permitir posición en (0,0) ni solapamientos
            if pos != [0, 0] and pos not in positions and pos not in exclude_positions:
                positions.append(pos)
            
            attempts += 1
        
        return positions
    
    def reset(self):
        # Reiniciar el grid
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
        
        # Reiniciar posición del robot (siempre en esquina superior izquierda)
        self.robot_pos = [0, 0]
        self.has_box = False
        
        # Generar posición de entrega (siempre en esquina inferior derecha)
        self.delivery_pos = [GRID_SIZE-1, GRID_SIZE-1]
        
        # Posiciones excluidas (robot y entrega)
        excluded = [self.robot_pos, self.delivery_pos]
        
        # Generar obstáculos aleatorios
        self.obstacles = self.generate_random_positions(NUM_OBSTACLES, excluded)
        excluded.extend(self.obstacles)
        
        # Generar cajas aleatorias
        self.boxes = self.generate_random_positions(NUM_BOXES, excluded)
        self.available_boxes = self.boxes.copy()  # Cajas disponibles para recoger
        
        # Configurar obstáculos en el grid
        for obs in self.obstacles:
            self.grid[obs[0], obs[1]] = -1  # Obstáculo
        
        # Configurar cajas disponibles en el grid
        for box in self.available_boxes:
            self.grid[box[0], box[1]] = 2  # Caja
        
        # Configurar punto de entrega
        self.grid[self.delivery_pos[0], self.delivery_pos[1]] = 3  # Punto de entrega
        
        # Guardar configuración actual para referencia
        self.current_config = {
            'boxes': self.boxes.copy(),
            'obstacles': self.obstacles.copy(),
            'delivery': self.delivery_pos.copy()
        }
        
        return self.get_state()
    
    def get_state(self):
        # Estado: (x, y, has_box)
        return (self.robot_pos[0], self.robot_pos[1], int(self.has_box))
    
    def step(self, action):
        reward = -1  # Costo por movimiento
        done = False
        
        new_pos = self.robot_pos.copy()
        
        # Mover según la acción
        if action == 0:  # Arriba
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 1:  # Derecha
            new_pos[1] = min(GRID_SIZE - 1, new_pos[1] + 1)
        elif action == 2:  # Abajo
            new_pos[0] = min(GRID_SIZE - 1, new_pos[0] + 1)
        elif action == 3:  # Izquierda
            new_pos[1] = max(0, new_pos[1] - 1)
        
        # Verificar si es un movimiento válido
        if new_pos in self.obstacles:
            reward = -50  # Penalización por chocar
        else:
            self.robot_pos = new_pos
            
            # Verificar si está en una caja disponible y no lleva ninguna
            if not self.has_box and self.robot_pos in self.available_boxes:
                self.has_box = True
                reward = 10  # Recompensa por recoger caja
                # Remover la caja de las disponibles (desaparece visualmente)
                self.available_boxes.remove(self.robot_pos)
                self.grid[self.robot_pos[0], self.robot_pos[1]] = 0  # Limpiar celda
            
            # Verificar si está en el punto de entrega con caja
            if self.has_box and self.robot_pos == self.delivery_pos:
                reward = 100  # Gran recompensa por entregar
                self.has_box = False
                # La caja se entrega y desaparece completamente
        
        return self.get_state(), reward, done
    
    def get_configuration_info(self):
        """Retorna información sobre la configuración actual"""
        return self.current_config

class QLearningAgent:
    def __init__(self, num_states, num_actions):
        # Tabla Q con dimensiones: x, y, has_box, acción
        self.q_table = np.zeros((num_states[0], num_states[1], num_states[2], num_actions))
    
    def choose_action(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.choice(ACTIONS)  # Exploración
        else:
            return np.argmax(self.q_table[state[0], state[1], state[2]])  # Explotación
    
    def update_q_table(self, state, action, reward, next_state):
        old_value = self.q_table[state[0], state[1], state[2], action]
        next_max = np.max(self.q_table[next_state[0], next_state[1], next_state[2]])
        
        new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
        self.q_table[state[0], state[1], state[2], action] = new_value
    
    def print_q_table_summary(self):
        print("\n=== RESUMEN DE TABLA Q ===")
        print(f"Dimensión: {self.q_table.shape}")
        print(f"Valor máximo: {np.max(self.q_table):.2f}")
        print(f"Valor mínimo: {np.min(self.q_table):.2f}")
        print(f"Valor promedio: {np.mean(self.q_table):.2f}")

def draw_grid(screen, env, episode, total_reward, steps, deliveries, config_info):
    screen.fill(WHITE)
    
    # Dibujar grid
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            
            if [i, j] in env.obstacles:
                pygame.draw.rect(screen, BLACK, rect)  # Obstáculo
            elif [i, j] in env.available_boxes:
                pygame.draw.rect(screen, YELLOW, rect)  # Caja disponible
            elif [i, j] == env.delivery_pos:
                pygame.draw.rect(screen, GREEN, rect)  # Punto de entrega
            else:
                pygame.draw.rect(screen, GRAY, rect)  # Pasillo
            
            pygame.draw.rect(screen, BLACK, rect, 1)  # Bordes
    
    # Dibujar robot
    robot_rect = pygame.Rect(env.robot_pos[1] * CELL_SIZE, env.robot_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    if env.has_box:
        pygame.draw.rect(screen, ORANGE, robot_rect)  # Robot con caja
    else:
        pygame.draw.rect(screen, RED, robot_rect)  # Robot sin caja
    
    # Dibujar información
    font = pygame.font.SysFont(None, 20)
    
    # Información del episodio
    info_text = f"Episodio: {episode} | Recompensa: {total_reward} | Pasos: {steps} | Entregas: {deliveries}"
    text_surface = font.render(info_text, True, BLACK)
    screen.blit(text_surface, (10, HEIGHT - 140))
    
    # Parámetros Q-Learning
    q_text = f"Q-Learning: α={ALPHA}, γ={GAMMA}, ε={EPSILON:.3f}"
    text_surface = font.render(q_text, True, BLACK)
    screen.blit(text_surface, (10, HEIGHT - 120))
    
    # Estado actual
    state_text = f"Estado: ({env.robot_pos[0]}, {env.robot_pos[1]}, {int(env.has_box)})"
    text_surface = font.render(state_text, True, BLACK)
    screen.blit(text_surface, (10, HEIGHT - 100))
    
    # Cajas disponibles
    boxes_text = f"Cajas: {len(env.available_boxes)}/{NUM_BOXES} disponibles"
    text_surface = font.render(boxes_text, True, PURPLE)
    screen.blit(text_surface, (10, HEIGHT - 80))
    
    # Obstáculos
    obs_text = f"Obstáculos: {NUM_OBSTACLES}"
    text_surface = font.render(obs_text, True, BLACK)
    screen.blit(text_surface, (10, HEIGHT - 60))
    
    pygame.display.flip()

def main():
    global EPSILON
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Robot en Bodega - Ambientes Aleatorios con Q-Learning")
    
    env = WarehouseEnv()
    num_states = (GRID_SIZE, GRID_SIZE, 2)  # x, y, has_box
    agent = QLearningAgent(num_states, len(ACTIONS))
    
    clock = pygame.time.Clock()
    running = True
    episode = 0
    total_deliveries = 0
    
    print("=== SIMULACIÓN CON AMBIENTES ALEATORIOS ===")
    print(f"Cajas por episodio: {NUM_BOXES}")
    print(f"Obstáculos por episodio: {NUM_OBSTACLES}")
    print("Cada episodio tiene una configuración diferente!\n")
    
    while running:
        episode += 1
        state = env.reset()
        total_reward = 0
        steps = 0
        episode_deliveries = 0
        done = False
        
        # Mostrar información del nuevo ambiente
        config_info = env.get_configuration_info()
        print(f"Episodio {episode} - Nueva configuración:")
        print(f"  Cajas: {config_info['boxes']}")
        print(f"  Obstáculos: {len(config_info['obstacles'])}")
        print(f"  Entrega: {config_info['delivery']}")
        
        while not done and steps < 300:  # Límite de pasos por episodio
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                    done = True
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                        done = True
                    elif event.key == K_q:
                        # Mostrar tabla Q al presionar 'q'
                        agent.print_q_table_summary()
                    elif event.key == K_r:
                        # Reiniciar episodio al presionar 'r'
                        done = True
            
            action = agent.choose_action(state, EPSILON)
            next_state, reward, done = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Contar entregas (cuando recompensa es 100)
            if reward == 100:
                episode_deliveries += 1
                total_deliveries += 1
            
            draw_grid(screen, env, episode, total_reward, steps, total_deliveries, config_info)
            clock.tick(2000)  # Velocidad moderada para ambientes cambiantes
            
            # Terminar episodio si no quedan cajas disponibles
            if len(env.available_boxes) == 0 and not env.has_box:
                done = True
                print(f"  ¡Todas las cajas recogidas! Recompensa final: {total_reward}")
        
        # Reducir epsilon más lentamente para ambientes cambiantes
        EPSILON = max(0.1, EPSILON * 0.995)
        
        print(f"  Resultado: Recompensa={total_reward}, Pasos={steps}, Entregas={episode_deliveries}")
        
        # Mostrar tabla Q cada 5 episodios
        if episode % 5 == 0:
            agent.print_q_table_summary()
        
        # Pausa breve entre episodios para apreciar los cambios
        time.sleep(0.5)
        
        if episode >= 300:  # Detener después de 100 episodios
            running = False
    
    # Mostrar estadísticas finales
    print(f"\n=== ENTRENAMIENTO COMPLETADO ===")
    print(f"Total de episodios: {episode}")
    print(f"Total de entregas: {total_deliveries}")
    print(f"Epsilon final: {EPSILON:.3f}")
    
    # Mostrar tabla Q final
    print("\n=== TABLA Q FINAL ===")
    agent.print_q_table_summary()
    
    pygame.quit()

if __name__ == "__main__":
    main()