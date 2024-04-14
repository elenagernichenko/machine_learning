from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Загрузка данных Iris
iris = datasets.load_iris()
X = iris.data

# Список для хранения значений инерции
inertia = []

# Пробуем разное количество кластеров и записываем значения инерции
for n_clusters in range(1, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Построение графика "локтя"
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
