#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>

using namespace std;

const int INPUT_SIZE = 6;
const int HIDDEN_SIZE = 2;
const int BATCH_SIZE = 20;
const int EPOCHS = 2000;
const double EPSILON = 1e-8;

class NeuralNetwork {
public:
    NeuralNetwork() {
        srand(static_cast<unsigned>(time(0)));

        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < HIDDEN_SIZE; ++j) {
                weights_input_hidden[i][j] = static_cast<double>(rand()) / RAND_MAX * sqrt(2.0 / (INPUT_SIZE + HIDDEN_SIZE));
            }
        }

        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            weights_hidden_output[j] = static_cast<double>(rand()) / RAND_MAX * sqrt(2.0 / (HIDDEN_SIZE + 1));
        }
    }

    double activationFunction(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    double activationFunctionDerivative(double x) {
        double sigmoid = activationFunction(x);
        return sigmoid * (1.0 - sigmoid);
    }

    double outputActivationFunction(double x) {
        return x; // Линейная активация для выхода
    }

    double outputActivationFunctionDerivative(double x) {
        return 1.0;
    }

    void forward(const vector<double>& inputs) {
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            hidden_layer[j] = 0.0;
            for (int i = 0; i < INPUT_SIZE; ++i) {
                hidden_layer[j] += inputs[i] * weights_input_hidden[i][j];
            }
            hidden_layer[j] = activationFunction(hidden_layer[j]);
        }

        output_layer = 0.0;
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            output_layer += hidden_layer[j] * weights_hidden_output[j];
        }
        output_layer = outputActivationFunction(output_layer);
    }

    void backward(const vector<double>& inputs, double target, double learning_rate) {
        double output_error = target - output_layer;
        double output_delta = output_error * outputActivationFunctionDerivative(output_layer);

        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            double hidden_error = output_delta * weights_hidden_output[j];
            double hidden_delta = hidden_error * activationFunctionDerivative(hidden_layer[j]);

            weights_hidden_output[j] += learning_rate * output_delta * hidden_layer[j];

            for (int i = 0; i < INPUT_SIZE; ++i) {
                weights_input_hidden[i][j] += learning_rate * hidden_delta * inputs[i];
            }
        }
    }

    double calculateAdaptiveLearningRate(const vector<double>& targets) {
        double numerator = 0.0;
        double denominator_1 = 0.0;
        double denominator_2 = 0.0;
        double denominator = 0.0;

        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            double error = targets[j] - output_layer;
            double f_prime = activationFunctionDerivative(output_layer);
            numerator += pow(error, 2) * f_prime;
            denominator_1 += f_prime * (1 + pow(output_layer, 2));
            denominator_2 += pow(pow(error, 2) * f_prime, 2);
            denominator += denominator_1 * denominator_2;
        }

        if (denominator < EPSILON) {
            return EPSILON;
        }

        double adaptive_learning_rate = numerator / denominator;

        return min(adaptive_learning_rate, 0.01);
    }

    void train(const vector<vector<double>>& training_data, const vector<double>& targets) {
        double previous_error = 0.0;

        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            double epoch_error = 0.0;
            double learning_rate = calculateAdaptiveLearningRate(targets);


            for (size_t i = 0; i < training_data.size(); i += BATCH_SIZE) {
                for (int b = 0; b < BATCH_SIZE && (i + b) < training_data.size(); ++b) {
                    forward(training_data[i + b]);
                    double target = targets[i + b];
                    double error = target - output_layer;
                    epoch_error += error * error;
                    backward(training_data[i + b], target, learning_rate);
                }
            }

            cout << "Epoch " << epoch + 1 << ", Error: " << epoch_error / training_data.size() << endl;
        }
    }

    double predict(const vector<double>& inputs) {
        forward(inputs);
        return output_layer;
    }

private:
    double weights_input_hidden[INPUT_SIZE][HIDDEN_SIZE] = { 0 };
    double weights_hidden_output[HIDDEN_SIZE] = { 0 };
    double hidden_layer[HIDDEN_SIZE] = { 0 };
    double output_layer = 0.0;
};

double targetFunction(double x) {
    const double a = 0.4, b = 0.4, c = 0.08, d = 0.4;
    return a * cos(b * x) + c * sin(d * x);
}

int main() {
    NeuralNetwork nn;

    vector<vector<double>> training_data;
    vector<double> targets;

    for (double x = 0.0; x <= 200.0; x += 1.0) {
        vector<double> input(INPUT_SIZE);
        input[0] = targetFunction(x);
        input[1] = targetFunction(x + 1);
        input[2] = targetFunction(x + 2);
        input[3] = targetFunction(x + 3);
        input[4] = targetFunction(x + 4);
        input[5] = targetFunction(x + 5);

        training_data.push_back(input);
        targets.push_back(targetFunction(x + 6));
    }

    nn.train(training_data, targets);

    // Тестирование
    for (double x = 0.0; x <= 200.0; x += 1.0) {
        vector<double> input(INPUT_SIZE);
        input[0] = targetFunction(x);
        input[1] = targetFunction(x + 1);
        input[2] = targetFunction(x + 2);
        input[3] = targetFunction(x + 3);
        input[4] = targetFunction(x + 4);
        input[5] = targetFunction(x + 5);

        double prediction = nn.predict(input);
        cout << "Input: " << x << ", Prediction: " << prediction << ", Target: " << targetFunction(x + 6) << endl;
    }

    return 0;
}
