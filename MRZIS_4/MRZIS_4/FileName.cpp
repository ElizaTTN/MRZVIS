#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <random>
#include "create_dataset.h"
using namespace std;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

const int epochs = 5000; 
const double learning_rate = 0.1;
const int batch_size = 5;

const int input_size = 2;
const int hidden_size = 15;
const int output_size = 1;

vector<vector<double>> weights_input_hidden(input_size, vector<double>(hidden_size));
vector<double> bias_hidden(hidden_size);
vector<vector<double>> weights_hidden_output(hidden_size, vector<double>(output_size));
double bias_output;
vector<double> hidden_layer;
double output;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<double> dist(-1.0, 1.0);

double Random(double min, double max) {
    return dist(gen);
}


void initialize_weights() {
    // Инициализация весов случайными значениями
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            weights_input_hidden[i][j] = Random(-1, 1); // Значения от -1 до 1
        }
    }

    for (int j = 0; j < hidden_size; ++j) {
        bias_hidden[j] = 0; // Начальное значение смещения
    }

    for (int j = 0; j < hidden_size; ++j) {
        for(int k = 0; k < output_size; k++)
        weights_hidden_output[j][k] = Random(-1, 1); // Значения от -1 до 1
    }

    bias_output = 0; // Начальное значение смещения
}

void forward(const vector<double>& input) {
    hidden_layer.resize(hidden_size);
    for (int j = 0; j < hidden_size; ++j) {
        hidden_layer[j] = bias_hidden[j];
        for (int i = 0; i < input_size; ++i)
            hidden_layer[j] += input[i] * weights_input_hidden[i][j];
        hidden_layer[j] = sigmoid(hidden_layer[j]);
    }

    output = bias_output;
    for (int j = 0; j < hidden_size; ++j)
        for (int k = 0; k < output_size; k++)
        output += hidden_layer[j]* weights_hidden_output[j][k];
    output = sigmoid(output);
}

double calculate_loss(const vector<double>& e, const vector<double>& y) {
    double loss = 0.0;
    for (size_t i = 0; i < e.size(); ++i) {
        double error = e[i] - y[i];
        loss += error * error;
    }
    return loss / e.size();
}

// Функция для расчета метрик
void calculate_metrics(const vector<double>& y_true, const vector<double>& y_pred) {
    int tp = 0, fp = 0, fn = 0, tn = 0;

    for (size_t i = 0; i < y_true.size(); ++i) {
        int predicted = y_pred[i] > 0.5 ? 1 : 0;
        int actual = y_true[i] > 0.5 ? 1 : 0;

        if (predicted == 1 && actual == 1) tp++;
        else if (predicted == 1 && actual == 0) fp++;
        else if (predicted == 0 && actual == 1) fn++;
        else if (predicted == 0 && actual == 0) tn++;
    }

    double precision = tp + fp > 0 ? static_cast<double>(tp) / (tp + fp) : 0.0;
    double recall = tp + fn > 0 ? static_cast<double>(tp) / (tp + fn) : 0.0;
    double f1_score = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0.0;

    cout << "Precision: " << precision << endl;
    cout << "Recall: " << recall << endl;
    cout << "F1 Score: " << f1_score << endl;
}

// Модифицированный train, чтобы сохранить предсказания
void train(const vector<vector<double>>& X, const vector<double>& y) {
    vector<double> predictions(X.size(), 0.0); // Теперь доступна везде
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;

        // Градиенты
        vector<double> weights_hidden_output_gradient(hidden_size, 0.0);
        vector<double> bias_hidden_gradient(hidden_size, 0.0);
        vector<vector<double>> weights_input_hidden_gradient(input_size, vector<double>(hidden_size, 0.0));
        double bias_output_gradient = 0.0;

        for (size_t i = 0; i < X.size(); i += batch_size) {
            size_t batch_end = min(i + batch_size, X.size());

            for (size_t b = i; b < batch_end; ++b) {
                forward(X[b]);
                double output_delta = (y[b] - output) * sigmoid_derivative(output);

                for (int j = 0; j < hidden_size; ++j) {
                    for (int k = 0; k < output_size; k++) {
                        double hidden_error = (y[b] - output) * sigmoid_derivative(output) * weights_hidden_output[j][k];
                        double hidden_delta = hidden_error * sigmoid_derivative(hidden_layer[j]);

                        weights_hidden_output_gradient[j] += hidden_layer[j] * output_delta;
                        bias_hidden_gradient[j] += hidden_delta;

                        for (int k = 0; k < input_size; ++k) {
                            weights_input_hidden_gradient[k][j] += X[b][k] * hidden_delta;
                        }
                    }
                }

                bias_output_gradient += output_delta;
                predictions[b] = output;
            }

            // Обновление весов
            for (int j = 0; j < hidden_size; ++j) {
                for (int k = 0; k < output_size; k++) {
                    weights_hidden_output[j][k] += weights_hidden_output_gradient[j] * learning_rate / batch_size;
                    bias_hidden[j] += bias_hidden_gradient[j] * learning_rate / batch_size;
                }
            }

            for (int j = 0; j < input_size; ++j) {
                for (int k = 0; k < hidden_size; ++k) {
                    weights_input_hidden[j][k] += weights_input_hidden_gradient[j][k] * learning_rate / batch_size;
                }
            }

            bias_output += bias_output_gradient * learning_rate / batch_size;

            fill(weights_hidden_output_gradient.begin(), weights_hidden_output_gradient.end(), 0.0);
            fill(bias_hidden_gradient.begin(), bias_hidden_gradient.end(), 0.0);
            for (auto& row : weights_input_hidden_gradient) {
                fill(row.begin(), row.end(), 0.0);
            }
            bias_output_gradient = 0.0;

            total_loss += calculate_loss(predictions, y);
        }

        if (epoch % 100 == 0) {
            cout << "Epoch " << epoch << ", Loss: " << total_loss / (X.size() / batch_size) << endl;
        }
    }

    // Вывод метрик после тренировки
    calculate_metrics(y, predictions);
}


double predict(const vector<double>& x) {
    forward(x);
    return output;
}


int main() {

    initialize_weights();

    const int dataset_size = 100;

    auto dataset = generate_dataset(dataset_size);

    vector<vector<double>> inputs;
    vector<double> outputs;

    for (const auto& data : dataset) {
        inputs.push_back(data.first);
        outputs.push_back(data.second);
    }

    // Разделение на обучающую и тестовую выборки
    const double test_ratio = 0.2; // 20% тестовых данных
    size_t test_size = static_cast<size_t>(dataset_size * test_ratio);
    vector<vector<double>> test_inputs(inputs.end() - test_size, inputs.end());
    vector<double> test_outputs(outputs.end() - test_size, outputs.end());
    inputs.resize(inputs.size() - test_size);
    outputs.resize(outputs.size() - test_size);

    train(inputs, outputs);

    int iteration = 1; 
    for (const auto& input : inputs) {
        double prediction = predict(input);
        cout << "Iteration " << iteration << ": Input: " << input[0] << ", " << input[1]
            << " => Prediction: " << (prediction > 0.5 ? 1 : 0)
            << " (" << prediction << ")" << endl;
        ++iteration; 
    }
    vector<vector<double>> p = {
       {0.3, 0.3},
       {0.6, 0.3},
       {0.3, 0.6},
       {0.6, 0.6}
    };

    for (auto& input : p)
    {
        double prediction = predict(input);
        cout << "Inputs: " << input[0] << " " << input[1] << "  Prediction: " << (prediction > 0.5 ? 1 : 0) << " (" << prediction << ")" << endl;
    }
    // Тестирование
    vector<double> test_predictions;
    for (const auto& input : test_inputs) {
        double prediction = predict(input);
        test_predictions.push_back(prediction > 0.5 ? 1.0 : 0.0);
    }

    // Метрики
    calculate_metrics(test_outputs, test_predictions);




    return 0;
}
