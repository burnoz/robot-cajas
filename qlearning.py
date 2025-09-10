import numpy as np
import matplotlib.pyplot as plt
import pygame
import sys
import time
import json

# Ambiente
rows = 11
cols = 11

# Tabla para valores Q
q_table = np.zeros((rows, cols, 4))

# Acciones
actions = ['up', 'down', 'left', 'right', 'pick_box', 'drop_box']

# Recompensas
# -1 por cada paso, +10 por recoger caja, +100 por dejar caja en destino
# -100 por choque con estante, -200 por choque con maquinaria
# -50 por dejar caja en lugar incorrecto
rewards = np.full((rows, cols), -1)

# Definir estantes (obstaculos)
shelves = [
    # Horizontales
    (0,7), (0,8), (0,9), (0,10),
    (10, 0), (10, 1),
    (10, 3), (10, 4), (10, 5), (10, 6), (10, 7),

    # Vericales
    # 3 - 7
    (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
    (3, 3), (4, 3), (5, 3), (6, 3), (7, 3),
    (3, 7), (4, 7), (5, 7), (6, 7), (7, 7),
    (3, 10), (4, 10), (5, 10), (6, 10), (7, 10)
]

for r, c in shelves:
    rewards[r, c] = -100

# Definir maquinaria (obstaculos)
machinery = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)
]

for r, c in machinery:
    rewards[r, c] = -200


# Punto de entrega
delivery_point = (10, 10)

rewards[delivery_point] = 100

# Imprime el tablero de recompensas
# for r in range(rows):
#     for c in range(cols):
#         print(f"{rewards[r, c]:4}", end=" ")
#     print()


def is_terminal_state(state):
    r, c = state
    
    if rewards[r, c] == 100 or rewards[r, c] == -100 or rewards[r, c] == -200:
        return True
    
    return False



def get_starting_location():
    while True:
        r = np.random.randint(rows)
        c = np.random.randint(cols)

        if rewards[r, c] == -1:  # Solo empezar en lugares libres
            return (r, c)


def next_action(state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(4)  # Acción aleatoria (exploración)
    else:
        r, c = state
        return np.argmax(q_table[r, c])  # Mejor acción (explotación)
    

def next_location(state, action):
    r, c = state
    if actions[action] == 'up' and r > 0:
        r -= 1

    elif actions[action] == 'down' and r < rows - 1:
        r += 1

    elif actions[action] == 'left' and c > 0:
        c -= 1

    elif actions[action] == 'right' and c < cols - 1:
        c += 1
        
    return (r, c)


def update_q_table(state, action, reward, next_state, alpha, gamma):
    r, c = state
    next_r, next_c = next_state
    q_table[r, c, action] = (1 - alpha) * q_table[r, c, action] + \
        alpha * (reward + gamma * np.max(q_table[next_r, next_c]))
    

def get_optimal_path(start):
    if is_terminal_state(start):
        return []
    
    else:
        path = []
        current_state = start
        path.append(current_state)

        while not is_terminal_state(current_state):
            r, c = current_state
            action = np.argmax(q_table[r, c])
            current_state = next_location(current_state, action)
            path.append(current_state)

        return path


# Parametros de Q-learning
epsilon = 0.9  # Probabilidad de exploracion
discount_factor = 0.9  # Factor de descuento
learning_rate = 0.9  # Tasa de aprendizaje

num_episodes = 10000


# --- Para graficar la evolución del valor Q en zonas clave ---
key_positions = [
    (delivery_point[0], delivery_point[1]-1), # izquierda de entrega
    (delivery_point[0]-1, delivery_point[1]), # arriba de entrega
    (3, 1), # junto a estante vertical
    (1, 7), # junto a estante horizontal
]
q_evolution = {pos: [] for pos in key_positions}

for episode in range(num_episodes):
    state = get_starting_location()
    total_reward = 0
    steps = 0

    while True:
        action = next_action(state, epsilon)
        next_state = next_location(state, action)
        r, c = next_state
        reward = rewards[r, c]
        total_reward += reward

        update_q_table(state, action, reward, next_state, learning_rate, discount_factor)

        state = next_state
        steps += 1

        if reward == 100 or reward == -100 or reward == -200:
            break

    # Guardar el valor máximo de Q en cada posición clave
    for pos in key_positions:
        q_evolution[pos].append(np.max(q_table[pos[0], pos[1]]))

    if (episode + 1) % 1000 == 0:
        print(f"Episode {episode + 1}: Total Reward: {total_reward}, Steps: {steps}")


# Imprime la tabla Q
# print("\nQ-Table:")
# for r in range(rows):
#     for c in range(cols):
#         print(f"{q_table[r, c]} ", end=" ")
#     print()

# Prueba el agente
start = get_starting_location()
optimal_path = get_optimal_path(start)
print(f"\nMejor camino desde {start} hasta {delivery_point}:")
print(optimal_path)


output = {
    "shelves": shelves,
    "machinery": machinery,
    "delivery_point": delivery_point,
    "robot_path": optimal_path
}

with open("output.json", "w") as f:
    json.dump(output, f, indent=4)
    print("Archivo output.json generado con la información del ambiente y el recorrido.")

# Obtener el valor máximo de Q para cada celda
q_max = np.max(q_table, axis=2)

plt.figure(figsize=(6, 6))
plt.imshow(q_max, cmap='hot', interpolation='nearest')
plt.colorbar(label='Valor máximo Q')
plt.title('Mapa de calor de la tabla Q')
plt.xlabel('Columna')
plt.ylabel('Fila')

plt.show()

# Gráfica de la evolución del valor Q en zonas clave
plt.figure(figsize=(10, 6))
for pos in key_positions:
    plt.plot(q_evolution[pos], label=f"{pos}")
plt.xlabel('Episodio')
plt.ylabel('Valor máximo Q')
plt.title('Evolución del valor Q en zonas clave')
plt.legend(title='Posición (fila, columna)')
plt.tight_layout()
plt.show()


# Parámetros visuales
CELL_SIZE = 40
MARGIN = 2
WINDOW_SIZE = (cols * CELL_SIZE, rows * CELL_SIZE)

pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption('Recorrido óptimo Q-Learning')

def draw_grid(path=None):
    for r in range(rows):
        for c in range(cols):
            color = (200, 200, 200)  # libre
            if (r, c) in shelves:
                color = (0, 0, 0)  # estante
            elif (r, c) in machinery:
                color = (128, 0, 128)  # maquinaria
            elif (r, c) == delivery_point:
                color = (0, 255, 0)  # destino
            if path and (r, c) in path:
                color = (30, 144, 255)  # camino
            if selected_cell is not None and (r, c) == selected_cell:
                color = (255, 255, 0)  # celda seleccionada
            if auto_box_cell is not None and (r, c) == auto_box_cell:
                color = (255, 140, 0)  # caja automática
            pygame.draw.rect(
                screen,
                color,
                [c * CELL_SIZE + MARGIN, r * CELL_SIZE + MARGIN, CELL_SIZE - MARGIN, CELL_SIZE - MARGIN]
            )

def animate_path(path):
    # Actualiza el JSON de salida antes de animar
    output = {
        "shelves": shelves,
        "machinery": machinery,
        "delivery_point": delivery_point,
        "robot_path": path
    }
    
    with open("output.json", "w") as f:
        json.dump(output, f, indent=4)

    for idx, pos in enumerate(path):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        screen.fill((0, 0, 0))
        draw_grid(path[:idx+1])
        # Dibuja el agente
        r, c = pos
        pygame.draw.circle(
            screen,
            (255, 0, 0),
            (c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2),
            CELL_SIZE // 3
        )
        pygame.display.flip()
        time.sleep(0.2)

selected_cell = None
path_to_animate = None
auto_box_cell = None
auto_box_timer = time.time()
auto_box_interval = 5  # segundos
auto_box_animating = False

# Bucle principal de PyGame
while True:
    current_time = time.time()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            col = mouse_x // CELL_SIZE
            row = mouse_y // CELL_SIZE
            if 0 <= row < rows and 0 <= col < cols:
                if rewards[row, col] == -1:
                    selected_cell = (row, col)
                    path_to_animate = None
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and selected_cell is not None:
                path_to_animate = get_optimal_path(selected_cell)
                animate_path(path_to_animate)

    # Animación automática cada cierto tiempo
    if not auto_box_animating and current_time - auto_box_timer > auto_box_interval:
        # Selecciona una celda libre aleatoria
        while True:
            r = np.random.randint(rows)
            c = np.random.randint(cols)
            if rewards[r, c] == -1:
                auto_box_cell = (r, c)
                break
        auto_box_animating = True
        auto_box_timer = current_time
        auto_path = get_optimal_path(auto_box_cell)
        animate_path(auto_path)
        auto_box_animating = False
        auto_box_cell = None

    screen.fill((0, 0, 0))
    draw_grid(path_to_animate)
    pygame.display.flip()
