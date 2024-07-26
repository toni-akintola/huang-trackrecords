import networkx as nx
import numpy as np
from modelpy_abm.main import AgentModel


def constructModel() -> AgentModel:
    def generateInitialData(model: AgentModel):
        initial_data = {
            "credences": list(np.ones(6) / 6),
            "track_record": [],
            "curr_brier": 1,
            "c": np.random.uniform(0, 1),
            "b": np.random.uniform(0, 0.2),
            "m": np.random.uniform(0.05, 0.5),
        }
        return initial_data

    def generateTimestepData(model: AgentModel):
        def _update_credences(node_data, coin_flip):
            ideal_update = _bayesian_update(node_data, coin_flip)
            noisy_update = np.random.normal(ideal_update, node_data["b"])
            node_data["credences"] = list(
                node_data["c"] * noisy_update
                + (1 - node_data["c"]) * np.array(node_data["credences"])
            )
            node_data["credences"] = list(
                np.array(node_data["credences"]) / np.sum(node_data["credences"])
            )

        def _calculate_likelihoods(coin_flip):
            biases = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            return biases if coin_flip == 1 else (1 - biases)

        def _bayesian_update(node_data, coin_flip):
            likelihoods = _calculate_likelihoods(coin_flip)
            posteriors = node_data["credences"] * likelihoods
            return posteriors / np.sum(posteriors)

        def _solicit_testimony(model: AgentModel, node):
            graph = model.get_graph()
            neighbors = list(graph.neighbors(node))
            node_data = graph.nodes[node]
            if neighbors:
                informants = np.random.choice(
                    neighbors, int(len(neighbors) * node_data["m"]), replace=False
                )
                social_update = np.mean(
                    [graph.nodes[informant]["credences"] for informant in informants],
                    axis=0,
                )
                node_data["credences"] = list(
                    node_data["m"] * social_update
                    + (1 - node_data["m"]) * np.array(node_data["credences"])
                )
                node_data["credences"] = list(
                    np.array(node_data["credences"]) / np.sum(node_data["credences"])
                )

        def _record_track(model: AgentModel, node_data):
            true_bias_vector = np.array(
                [
                    1 if model["true_bias"] == bias else 0
                    for bias in [0, 0.2, 0.4, 0.6, 0.8, 1]
                ]
            )
            accuracy = 1 - np.mean((node_data["credences"] - true_bias_vector) ** 2)
            node_data["track_record"].append(accuracy)
            node_data["curr_brier"] = accuracy

        coin_flip = 1 if np.random.rand() < model["true_bias"] else 0
        graph = model.get_graph()
        for node, node_data in graph.nodes(data=True):
            _update_credences(node_data, coin_flip)
            _solicit_testimony(model, node)
            _record_track(model, node_data)
        model.set_graph(graph)

    model = AgentModel()

    model.update_parameters(
        {"num_nodes": 30, "graph_type": "complete", "true_bias": 0.6}
    )

    model.set_initial_data_function(generateInitialData)
    model.set_timestep_function(generateTimestepData)

    return model
