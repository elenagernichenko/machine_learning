import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.decomposition import PCA

data = pd.read_csv('bikes_rent.csv')

# п.2  Построение прямой линии регрессии для зависимости
# спроса от благоприятности погоды
def simple_linear_regression(data):
    X = data[['weathersit']]  # Признак - благоприятность погоды
    y = data['cnt']           # Целевая переменная - количество арендованных велосипедов

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели линейной регрессии
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Построение прямой линии регрессии
    plt.scatter(X_test, y_test, color='blue')
    plt.plot(X_test, model.predict(X_test), color='red', linewidth=3)
    plt.xlabel('Благоприятность погоды')
    plt.ylabel('Количество арендованных велосипедов')
    plt.title('Простая линейная регрессия')
    plt.show()

simple_linear_regression(data)

# п. 3 и 4: Предсказание значения cnt и построение 2D графика
def predict_cnt(data, weathersit_value):
    X = data[['weathersit']]  # Признак - благоприятность погоды
    y = data['cnt']           # Целевая переменная - количество арендованных велосипедов

    # Обучение модели линейной регрессии
    model = LinearRegression()
    model.fit(X, y)

    # Предсказание значения cnt для заданной благоприятности погоды
    cnt_prediction = model.predict([[weathersit_value]])
    return cnt_prediction[0]

def plot_2d_prediction(data):
    X = data.drop(columns=['cnt'])  # Все признаки, кроме целевой переменной
    y = data['cnt']  # Целевая переменная - количество арендованных велосипедов

    # Уменьшение размерности пространства до 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Обучение модели линейной регрессии
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Построение графика предсказания cnt в 2D пространстве
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis')
    plt.colorbar(label='Количество арендованных велосипедов')
    plt.xlabel('Главная компонента 1')
    plt.ylabel('Главная компонента 2')
    plt.title('Предсказание cnt в 2D пространстве')
    plt.show()

# Пример вызова функций
weathersit_value = 2  # Пример значения благоприятности погоды
predicted_cnt = predict_cnt(data, weathersit_value)
print(f"Предсказанное количество арендованных велосипедов при благоприятности погоды {weathersit_value}: {predicted_cnt}")
plot_2d_prediction(data)

# п.5  Регуляризация Lasso и
# определение признака с наибольшим влиянием на результат cnt
def lasso_regularization(data):
    X = data.drop(columns=['cnt'])  # Признаки, кроме целевой переменной
    y = data['cnt']                 # Целевая переменная - количество арендованных велосипедов

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели с регуляризацией Lasso
    model = Lasso(alpha=0.1)
    model.fit(X_train, y_train)

    # Определение признака с наибольшим влиянием на результат cnt
    max_coefficient_index = np.argmax(np.abs(model.coef_))
    max_influence_feature = X.columns[max_coefficient_index]
    return max_influence_feature

# Пример вызова функции
max_influence_feature = lasso_regularization(data)
print(f"Признак, оказывающий наибольшее влияние на результат cnt: {max_influence_feature}")
