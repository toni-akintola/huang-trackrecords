import networkx as nx
import numpy as np


# Define class
class HuangTrackRecord:
    def __init__(self):
        # Define Parameters
        self.num_nodes = 30
        self.graph_type = 'complete'  # 'complete', 'wheel', or 'cycle'
        self.true_bias = 0.6
        self.num_steps = 100

        # Graph initialization
        self.graph: nx.Graph = None

    def initialize_graph(self):
        # Initialize graph shape
        if self.graph_type == 'complete':
            self.graph = nx.complete_graph(self.num_nodes)
        elif self.graph_type == 'cycle':
            self.graph = nx.cycle_graph(self.num_nodes)
        else:
            self.graph = nx.wheel_graph(self.num_nodes)

        # Initialize agents
        for node in self.graph.nodes():
            initial_data = {
                'credences': np.ones(6) / 6,
                'track_record': [],
                'c': np.random.uniform(0, 1),
                'b': np.random.uniform(0, 0.2),
                'm': np.random.uniform(0.05, 0.5)
            }
            self.graph.nodes[node].update(initial_data)
        return self.graph

    def timestep(self):
        coin_flip = 1 if np.random.rand() < self.true_bias else 0
        for node, node_data in self.graph.nodes(data=True):
            self._update_credences(node_data, coin_flip)
            self._solicit_testimony(node)
            self._record_track(node_data)
        return self.graph

    def _update_credences(self, node_data, coin_flip):
        ideal_update = self._bayesian_update(node_data, coin_flip)
        noisy_update = np.random.normal(ideal_update, node_data['b'])
        node_data['credences'] = node_data['c'] * noisy_update + \
            (1 - node_data['c']) * node_data['credences']
        node_data['credences'] = node_data['credences'] / \
            np.sum(node_data['credences'])

    def _bayesian_update(self, node_data, coin_flip):
        likelihoods = self._calculate_likelihoods(coin_flip)
        posteriors = node_data['credences'] * likelihoods
        return posteriors / np.sum(posteriors)

    def _calculate_likelihoods(self, coin_flip):
        biases = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        return biases if coin_flip == 1 else (1 - biases)

    def _solicit_testimony(self, node):
        neighbors = list(self.graph.neighbors(node))
        node_data = self.graph.nodes[node]
        if neighbors:
            informants = np.random.choice(neighbors, int(
                len(neighbors) * node_data['m']), replace=False)
            social_update = np.mean(
                [self.graph.nodes[informant]['credences'] for informant in informants], axis=0)
            node_data['credences'] = node_data['m'] * social_update + \
                (1 - node_data['m']) * node_data['credences']
            node_data['credences'] = node_data['credences'] / \
                np.sum(node_data['credences'])

    def _record_track(self, node_data):
        true_bias_vector = np.array(
            [1 if self.true_bias == bias else 0 for bias in [0, 0.2, 0.4, 0.6, 0.8, 1]])
        accuracy = 1 - \
            np.mean((node_data['credences'] - true_bias_vector) ** 2)
        node_data['track_record'].append(accuracy)
