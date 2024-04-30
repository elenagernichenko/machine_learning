from random import random

import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn.datasets import load_iris

# Загрузка данных
iris = load_iris()
X = iris.data

# Количество кластеров
k = 3

# Инициализация центроидов
np.random.seed(0)
centroids = X[np.random.choice(range(len(X)), k, replace=False)]


# Функция для вычисления расстояния между точками
# сумма квадратов расстояний
def distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# Функция для присвоения точек к кластерам
def assign_clusters(X, centroids):
    clusters = []
    for point in X:
        distances = [distance(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)


# Функция для обновления центроидов
def update_centroids(X, clusters, k):
    centroids = []
    for i in range(k):
        cluster_points = X[clusters == i]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    return np.array(centroids)


# Основной цикл алгоритма K-means
max_iter = 100

# Создание списка для хранения изображений
images = []

for iteration in range(max_iter):
    # Присвоение точек к кластерам
    clusters = assign_clusters(X, centroids)

    # Визуализация текущего состояния
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b']
    for i in range(k):
        cluster_points = X[clusters == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Кластер {i + 1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', label='Центроиды')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title(f'K-means Clustering - Итерация {iteration + 1}')
    plt.legend()

    # Сохранение текущего изображения в списке
    plt.savefig(f'iteration_{iteration}.png')
    plt.close()

    # Добавление текущего изображения в список изображений для создания GIF
    images.append(imageio.imread(f'iteration_{iteration}.png'))

    # Пересчет центроидов
    new_centroids = update_centroids(X, clusters, k)

    # Проверка сходимости
    if np.all(centroids == new_centroids):
        print("Алгоритм сходится на шаге", iteration + 1)
        break

    centroids = new_centroids

# Создание GIF изображения
imageio.mimsave('kmeans_animation.gif', images, fps=2)
