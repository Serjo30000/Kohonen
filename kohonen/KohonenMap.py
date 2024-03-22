import numpy as np
import matplotlib.pyplot as plt

class KohonenMap:
    def __init__(self, input_size, map_size, seed, epochs, learning_rate):
        self.input_size = input_size
        self.map_size = map_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        if seed is not None:
            np.random.seed(seed)
        self.weights = np.random.rand(map_size[0], map_size[1], input_size)

    def train(self, data):
        for epoch in range(self.epochs):
            for input_vector in data:
                bmu_idx = self._find_best_matching_unit(input_vector)
                self._update_weights(input_vector, bmu_idx, epoch)
            if (epoch % 400 == 0 and epoch > 0):
                for i in range(len(data[0])):
                    self.plot_map_index(i)

    def _find_best_matching_unit(self, input_vector):
        distances = np.linalg.norm(self.weights - input_vector, axis=-1)
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx

    def _update_weights(self, input_vector, bmu_idx, epoch):
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                distance = np.linalg.norm(np.array(bmu_idx) - np.array([i, j]))
                adjusted_distance = distance + 0.1
                influence = self._calculate_influence(adjusted_distance, epoch)
                current_learning_rate = self.learning_rate * np.exp(-epoch / self.epochs)
                self.weights[i][j] += current_learning_rate * influence * (input_vector - self.weights[i][j])

    def _calculate_influence(self, distance, epoch):
        sigma = self.map_size[0] / (2 * np.exp(epoch / 100))
        return np.exp(-distance**2 / (2 * sigma**2))

    def plot_map_index(self, index):
        selected_values = np.take_along_axis(self.weights, np.array([index]).reshape(1, 1, -1), axis=-1)
        mean_weights = np.mean(selected_values, axis=-1)
        plt.imshow(mean_weights, cmap='viridis')
        plt.title('Kohonen Map ' + str(index + 1))
        plt.colorbar()
        plt.show()

    def plot_map(self):
        mean_weights = np.mean(self.weights, axis=-1)
        plt.imshow(mean_weights, cmap='viridis')
        plt.title('Kohonen Map ')
        plt.colorbar()
        plt.show()

    @staticmethod
    def conversionToFloat(inputs):
        category_mapping = {"Big": 0, "Middle": 1, "Little": 2, "Micro": 3}

        for row in inputs:
            category_column_index = len(row) - 1
            category = row[category_column_index]
            if category in category_mapping:
                row[category_column_index] = category_mapping[category]
            else:
                row[category_column_index] = len(category_mapping)

        inputs = np.array(inputs, dtype=float)
        return inputs

    @staticmethod
    def normalize(inputs):
        min_vals1 = np.min(inputs, axis=0)
        max_vals1 = np.max(inputs, axis=0)
        normalized_data = (inputs - min_vals1) / (max_vals1 - min_vals1)
        return normalized_data