import pandas as pd
from sklearn import metrics
from implementation.activation_functions import *
from implementation.custom_neural_network import NeuralNetwork

hidden_layers_arch = [
    {'count': 8, 'activation': tanh, 'activation_derivation': tanh_derivation},
    {'count': 5, 'activation': tanh, 'activation_derivation': tanh_derivation}
]
output_layer = {'count': 3, 'activation': softmax, 'activation_derivation': softmax_derivation}
network = NeuralNetwork(13, hidden_layers_arch, output_layer)

drinks_df = pd.read_csv('../Data/drinks.csv')
drinks_df = drinks_df.sample(frac=1).reset_index(drop=True)
drinks_features = drinks_df.values[:, :13]
drinks_targets = drinks_df.values[:, 13:]

seed = np.random.rand(100, 100)
network.train_network(drinks_features[:120, :], drinks_targets[:120, :], learning_rate=.7, iter=200)
predictions = network.predict(drinks_features[120:, :])
print('\n'.join(map(lambda x: str(x[0]) + ' vs ' + str(x[1]), zip(predictions, drinks_targets[120:, :].astype('int')))))
print('Accuracy:', metrics.accuracy_score(predictions, drinks_targets[120:, :]))
