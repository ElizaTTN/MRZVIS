#include "create_dataset.h"
#include <random>
#include <vector>
#include <utility>

using namespace std;

double Random4ik(double min, double max) {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min, max);
    return dist(gen);
}

vector<pair<vector<double>, double>> generate_dataset(int dataset_size) {
    vector<pair<vector<double>, double>> dataset;
    dataset.reserve(dataset_size); // Оптимизация для предотвращения лишних аллокаций

    for (int i = 0; i < dataset_size; ++i) {
        // Генерация случайных входов
        double x1 = Random4ik(0, 1);
        double x2 = Random4ik(0, 1);

        // XOR логика
        double y = ((x1 >= 0.5) != (x2 >= 0.5)) ? 1 : 0;

        // Добавляем пару вход-выход в датасет
        dataset.push_back({ {x1, x2}, y });
    }

    return dataset;
}

