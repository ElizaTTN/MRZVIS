import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from time import time
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Устанавливаем параметры для отображения всех строк и столбцов
pd.set_option('display.max_rows', None)  # Максимальное количество строк, None - все строки
pd.set_option('display.max_columns', None)  # Максимальное количество столбцов, None - все столбцы
pd.set_option('display.width', None)  # Автоматическая настройка ширины
pd.set_option('display.max_colwidth', None)  # Без ограничения по длине столбцов

# Загрузка датасета Abalone
def load_abalone(file_path):
    data = pd.read_csv(file_path)
    # Преобразование Sex в one-hot encoding
    data = pd.get_dummies(data, columns=['Sex'], drop_first=True)
    # Преобразование Rings в бинарный класс (молодые: <10, взрослые: >=10)
    data['Rings'] = np.where(data['Rings'] < 10, 0, 1)
    X = data.drop(columns=['Rings']).values
    y = data['Rings'].values
    return X, y


# Загрузка и предобработка данных
file_path = "abalone.csv"
X, y = load_abalone(file_path)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Параметры сети
input_size = X.shape[1]
hidden_size = 10
output_size = 1
epochs = 100
batch_size = 32
initial_learning_rate = 0.05
EPSILON = 1e-8

np.random.seed(42)
weights_input_hidden = np.random.randn(input_size, hidden_size)
weights_hidden_output = np.random.randn(hidden_size, output_size)

# Инициализация порогов
bias_hidden = np.zeros((1, hidden_size))  # Пороги для скрытого слоя
bias_output = np.zeros((1, output_size))  # Пороги для выходного слоя


# Функция активации и её производная для сигмоиды
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Активация и её производная
def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


# Функция потерь (кросс-энтропия)
def cross_entropy_loss(y, output):
    # Для числовых вычислений будем использовать np.clip, чтобы избежать log(0)
    epsilon = 1e-12
    output = np.clip(output, epsilon, 1. - epsilon)
    return -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))


def cross_entropy_gradient(y, output):
    epsilon = 1e-12
    output = np.clip(output, epsilon, 1. - epsilon)
    return (output - y) * (output * (1 - output))


# Прямое распространение
def forward(X):
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = tanh(hidden_input)  # Для скрытого слоя оставляем tanh
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)  # Используем сигмоиду для выходного слоя
    return hidden_output, final_output


# Обратное распространение с использованием кросс-энтропийной ошибки
def backward(X, y, hidden_output, final_output, learning_rate):
    global weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

    # Используем производную от кросс-энтропии
    d_output = cross_entropy_gradient(y, final_output) * sigmoid_derivative(final_output)  # Для сигмоиды
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden_layer * tanh_derivative(hidden_output)

    # Обновление весов и порогов
    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate


# Функция для вычисления адаптивного шага обучения
def calculate_adaptive_learning_rate(targets, output_layer, activation_derivative):
    numerator = 0.0
    denominator_1 = 0.0
    denominator_2 = 0.0
    denominator = 0.0

    for j in range(len(targets)):
        error = targets[j] - output_layer[j]
        f_prime = activation_derivative(output_layer[j])
        numerator += (error ** 2) * f_prime
        denominator_1 += f_prime * (1 + (output_layer[j] ** 2))
        denominator_2 += (error ** 2 * f_prime) ** 2
        denominator += denominator_1 * denominator_2

    if denominator < EPSILON:
        return EPSILON

    adaptive_learning_rate = numerator / denominator

    return min(adaptive_learning_rate, 0.01)



# Функция для вычисления точности, полноты и F1-меры
def calculate_additional_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    return precision, recall, f1, auc


def train(X, y, learning_rate, learning_type='constant', update_type='batch'):
    y = y.reshape(-1, 1)
    metrics = {'epochs': 0, 'training_time': 0, 'test_error': 0, 'accuracy': 0,
               'train_error': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'roc_auc': 0, 'epoch_time': 0}

    start_time = time()
    for epoch in range(epochs):
        epoch_start_time = time()

        # Обучение модели
        if learning_type == 'adaptive':
            hidden_output, final_output = forward(X)
            learning_rate = calculate_adaptive_learning_rate(y, final_output, tanh_derivative)

        if update_type == 'online':
            for i in range(len(X)):
                hidden_output, final_output = forward(X[i:i + 1])
                backward(X[i:i + 1], y[i:i + 1], hidden_output, final_output, learning_rate)

        elif update_type == 'batch':
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                hidden_output, final_output = forward(X_batch)
                backward(X_batch, y_batch, hidden_output, final_output, learning_rate)

        # Вычисление точности, полноты и других метрик
        _, predictions = forward(X_test)
        predictions = (predictions > 0.5).astype(int)
        precision, recall, f1, auc = calculate_additional_metrics(y_test, predictions)

        accuracy = accuracy_score(y_test, predictions)
        test_error = cross_entropy_loss(y_test, predictions)  # Кросс-энтропийная ошибка

        # Ошибка на обучении
        _, train_predictions = forward(X_train)
        train_predictions = (train_predictions > 0.5).astype(int)
        train_error = cross_entropy_loss(y_train, train_predictions)

        # Время на одну эпоху
        epoch_time = time() - epoch_start_time

        metrics['epochs'] = epoch + 1
        metrics['test_error'] = test_error
        metrics['accuracy'] = accuracy
        metrics['train_error'] = train_error
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        metrics['roc_auc'] = auc
        metrics['epoch_time'] = epoch_time

    metrics['training_time'] = time() - start_time
    return metrics


# Оценка сети
def evaluate():
    results = []
    metrics = train(X_train, y_train, learning_rate=initial_learning_rate, learning_type='constant',
                    update_type='batch')
    results.append({'method': 'Const, Batch', **metrics})

    metrics = train(X_train, y_train, learning_rate=initial_learning_rate, learning_type='constant',
                    update_type='online')
    results.append({'method': 'Const, Online', **metrics})

    metrics = train(X_train, y_train, learning_rate=initial_learning_rate, learning_type='adaptive',
                    update_type='batch')
    results.append({'method': 'Adapt , Batch', **metrics})

    metrics = train(X_train, y_train, learning_rate=initial_learning_rate, learning_type='adaptive',
                    update_type='online')
    results.append({'method': 'Adapt, Online', **metrics})

    results_df = pd.DataFrame(results)
    print(results_df)



evaluate()
