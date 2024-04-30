import pygame
from sklearn.metrics.pairwise import euclidean_distances
import random

pygame.init()

# Определение цветов
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)

COLORS = [GREEN, YELLOW, RED]

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# определение размеров точек
POINT_RADIUS = 5

# функция для рисования точек на экране
def draw_points(screen, points, colors):
    for point, color in zip(points, colors):
        pygame.draw.circle(screen, color, point, POINT_RADIUS)

# функция для определения соседей точки
def find_neighbors(points, point_index, eps):
    distances = euclidean_distances([points[point_index]], points)[0]
    neighbors = [i for i, dist in enumerate(distances) if dist <= eps and i != point_index]
    return neighbors

def dbscan(points, eps, min_pts):
    cluster_labels = [0] * len(points)
    cluster_id = 0
    for i, point in enumerate(points):
        if cluster_labels[i] != 0:
            continue
        neighbors = find_neighbors(points, i, eps)
        if len(neighbors) < min_pts:
            cluster_labels[i] = -1  # помечаем как выброс
            continue
        cluster_id += 1
        cluster_labels[i] = cluster_id
        while neighbors:
            current_neighbor = neighbors[0]
            if cluster_labels[current_neighbor] == -1:
                cluster_labels[current_neighbor] = cluster_id
            elif cluster_labels[current_neighbor] == 0:
                cluster_labels[current_neighbor] = cluster_id
                new_neighbors = find_neighbors(points, current_neighbor, eps)
                if len(new_neighbors) >= min_pts:
                    neighbors.extend(new_neighbors)
            del neighbors[0]
    return cluster_labels

def main():
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("DBSCAN Algorithm")

    points = []
    colors = []

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    points.append(event.pos)
                    colors.append(random.choice(COLORS))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    cluster_labels = dbscan(points, eps=50, min_pts=5)

                    for i, label in enumerate(cluster_labels):
                        if label == -1:
                            colors[i] = BLACK
                        elif label == 1:
                            colors[i] = GREEN
                        elif label == 2:
                            colors[i] = YELLOW
                        elif label == 3:
                            colors[i] = RED

                    screen.fill(BLACK)
                    draw_points(screen, points, colors)
                    pygame.display.flip()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_RETURN]:
            if colors:
                colors[-1] = random.choice(COLORS)

                screen.fill(BLACK)
                draw_points(screen, points, colors)
                pygame.display.flip()

        screen.fill(BLACK)
        draw_points(screen, points, colors)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
