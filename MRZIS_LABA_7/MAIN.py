import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)
        self.dropout_rate = dropout_rate

    def relu(self, z):
        return np.maximum(0.01 * z, z)

    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0.01)

    def tanh(self, z):
        return np.tanh(z)

    def tanh_derivative(self, z):
        return 1 - np.tanh(z) ** 2

    @staticmethod
    def compute_loss(predicted, actual, loss_type):
        if loss_type == "MSE":
            return (predicted - actual) ** 2
        elif loss_type == "MAE":
            return np.abs(predicted - actual)
        return 0.0

    def apply_dropout(self, activations, is_training):
        if is_training:
            mask = np.random.rand(*activations.shape) > self.dropout_rate
            return activations * mask
        else:
            return activations * (1 - self.dropout_rate)

    def forward(self, x, is_training=True):
        a1 = self.relu(np.dot(x, self.W1) + self.b1)
        a1 = self.apply_dropout(a1, is_training)
        z2 = self.tanh(np.dot(a1, self.W2) + self.b2)
        return z2

    def backward(self, x, y, output, learning_rate, lambda_):
        dz2 = (output - y) * self.tanh_derivative(output)

        a1 = self.relu(np.dot(x, self.W1) + self.b1)
        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(a1)

        self.W2 -= learning_rate * (np.outer(a1, dz2) + lambda_ * self.W2)
        self.b2 -= learning_rate * dz2

        self.W1 -= learning_rate * (np.outer(x, dz1) + lambda_ * self.W1)
        self.b1 -= learning_rate * dz1

    def train_batch(self, data, learning_rate, epochs, batch_size, lambda_, loss_type):
        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                for sequence in batch:
                    x, y = sequence[:-1], sequence[-1]
                    output = self.forward(x, is_training=True)
                    loss = self.compute_loss(output, y, loss_type)
                    total_loss += loss
                    self.backward(x, y, output, learning_rate, lambda_)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss ({loss_type}): {total_loss / len(data)}")

    def test(self, data, loss_type):
        total_loss = 0.0
        for i, sequence in enumerate(data):
            x, y = sequence[:-1], sequence[-1]
            output = self.forward(x, is_training=False)
            loss = self.compute_loss(output, y, loss_type)
            total_loss += loss
            print(f"{i + 1}: y = {y:.4f}, pred = {output[0]:.4f}, error = {output[0] - y:.4f}")

        return total_loss / len(data)


def generate_y_values(a, b, c, d, x_values):
    return a * np.cos(b * x_values) + c * np.sin(d * x_values)

def prepare_data(y_values, sequence_length, start, end):
    data = []
    for i in range(start, end - sequence_length):
        sequence = y_values[i:i + sequence_length + 1]
        data.append(sequence)
    return np.array(data)

if __name__ == "__main__":
    a, b, c, d = 0.2, 0.2, 0.06, 0.2
    x_values = np.arange(0, 1200) * 0.1
    y_values = generate_y_values(a, b, c, d, x_values)

    train_data = prepare_data(y_values, 8, 0, 600)
    test_data = prepare_data(y_values, 8, 600, 1200)

    input_size = 8
    hidden_size = 3
    output_size = 1
    mlp = MLP(input_size, hidden_size, output_size)

    mlp.train_batch(train_data, learning_rate=0.01, epochs=1000, batch_size=6, lambda_=0.001, loss_type="MSE")
    test_loss_mse = mlp.test(test_data, loss_type="MSE")
    print(f"Test Loss (MSE): {test_loss_mse}")

    mlp.train_batch(train_data, learning_rate=0.01, epochs=1000, batch_size=6, lambda_=0.001, loss_type="MAE")
    test_loss_mae = mlp.test(test_data, loss_type="MAE")
    print(f"Test Loss (MAE): {test_loss_mae}")
