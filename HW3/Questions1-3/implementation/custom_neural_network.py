import numpy as np
from dataclasses import dataclass


@dataclass
class NeuralLayer:
    node_raw_values: np.array
    node_activated_values: np.array
    weights: np.array
    activation = None
    activation_derivation = None
    errors: np.array = None

    def __init__(self, node_raw_values, weights, activation, activation_derivation):
        self.node_activated_values = None
        self.node_raw_values = node_raw_values
        self.weights = weights
        self.activation = activation
        self.activation_derivation = activation_derivation
        self.errors = np.zeros(weights.shape)


class NeuralNetwork:
    hidden_layers: list
    output_layer: NeuralLayer

    def __init__(self, input_features_count, hidden_layers_arch, output_layer):
        self.hidden_layers = []
        for i in range(len(hidden_layers_arch)):
            hidden_layer = hidden_layers_arch[i]
            if i == 0:
                weights = np.random.rand(hidden_layer['count'], input_features_count + 1)
            else:
                weights = np.random.rand(hidden_layer['count'], len(self.hidden_layers[-1].node_raw_values))
            self.hidden_layers.append(NeuralLayer(np.zeros(hidden_layer['count'] + 1), weights,
                                                  hidden_layer['activation'], hidden_layer['activation_derivation']))
        if len(self.hidden_layers) > 0:
            weights = np.random.rand(output_layer['count'], len(self.hidden_layers[-1].node_raw_values))
        else:
            weights = np.random.rand(output_layer['count'], input_features_count + 1)
        self.output_layer = NeuralLayer(np.zeros(output_layer['count']), weights,
                                        output_layer['activation'], output_layer['activation_derivation'])

    def _forward_propagate(self, datapoint):
        input = self._append_bias(datapoint)
        for layer in self.hidden_layers:
            layer.node_raw_values = np.matmul(layer.weights, input)
            layer.node_activated_values = self._append_bias(layer.activation(layer.node_raw_values))
            layer.node_raw_values = self._append_bias(layer.node_raw_values)
            input = layer.node_activated_values
        self.output_layer.node_raw_values = np.matmul(self.output_layer.weights, input)
        self.output_layer.node_activated_values = self.output_layer.activation(self.output_layer.node_raw_values)
        return self.output_layer.node_activated_values

    def _backward_propagate(self, datapoint, expected_outputs):
        sigmas = [self._append_bias(self.output_layer.node_activated_values - expected_outputs)]
        next_weights = self.output_layer.weights
        for i in reversed(range(len(self.hidden_layers))):
            layer = self.hidden_layers[i]
            a = np.matmul(next_weights.T, sigmas[-1][1:])
            b = np.multiply(a, layer.activation_derivation(layer.node_raw_values))
            sigmas.append(b)
            next_weights = layer.weights
        sigmas = list(reversed(sigmas))
        previous_outputs = self._append_bias(datapoint)
        for i in range(len(self.hidden_layers)):
            layer = self.hidden_layers[i]
            layer.errors += np.matmul(sigmas[i][1:][np.newaxis].T, previous_outputs[np.newaxis])
            previous_outputs = layer.node_activated_values
        self.output_layer.errors += np.matmul(sigmas[-1][1:][np.newaxis].T, previous_outputs[np.newaxis])

    def train_network(self, X, y, learning_rate, iter=1000, seed=np.random.rand(100, 100)):
        self.reset_network(seed)
        for epoch in range(iter):
            # Resetting previously learnt data
            self.reset_errors()
            # Computing gradient
            for i in range(len(X)):
                outputs = self._forward_propagate(X[i])
                self._backward_propagate(X[i], y[i])
            # Updating weights
            for layer in self.hidden_layers:
                layer.errors = layer.errors / len(X)
                layer.weights -= learning_rate * layer.errors
            self.output_layer.errors = self.output_layer.errors / len(X)
            self.output_layer.weights -= learning_rate * self.output_layer.errors

    def reset_errors(self):
        for layer in (self.hidden_layers + [self.output_layer]):
            layer.errors = np.zeros(layer.weights.shape)

    def reset_network(self, seed):
        for layer in (self.hidden_layers + [self.output_layer]):
            layer.errors = np.zeros(layer.weights.shape)
            layer.weights = seed[:layer.weights.shape[0], :layer.weights.shape[1]]
            layer.node_raw_values = np.zeros(layer.node_raw_values.shape)
            if layer.node_activated_values is not None:
                layer.node_activated_values = np.zeros(layer.node_activated_values.shape)

    def predict(self, datapoints):
        predictions = []
        for datapoint in datapoints:
            result = self._forward_propagate(datapoint)
            predictions.append(list(map(int, result == max(result))))
        return np.array(predictions)

    @staticmethod
    def _append_bias(row):
        return np.concatenate(([1], row))
