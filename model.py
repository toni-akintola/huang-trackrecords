import random
import networkx as nx
import numpy as np
import scipy.stats
from typing import Dict, List
from emergent.main import AgentModel


def generateInitialData(model: AgentModel) -> Dict:
    """Initialize a scientist with appropriate attributes"""
    # Create uniform distribution that sums to exactly 1
    initial_cred = np.ones(6) / 6

    return {
        "record": [],  # Track record
        "m": round(random.uniform(0.05, 0.5), 2),  # Open-mindedness
        "hyp": np.round(np.arange(0, 1.001, 1 / 5), 2).tolist(),
        # Credence for each hyp - now exactly sums to 1
        "cred": initial_cred.tolist(),
        "noise": random.uniform(0.001, 0.2),
        "c": (round(random.random(), 2) if model["scientist_type"] != "patient" else 1),
        "social": None,
        "evidential": None,
        "brier_score": 1,
        "brier_history": [],
    }


def generateTimestepData(model: AgentModel):
    """Main timestep function"""

    def calculate_brier_score(cred: List[float], toss: int) -> float:
        """Calculate Brier score for a prediction"""
        return round(
            (toss - sum(np.array(cred) * np.array(model.get_graph().nodes[0]["hyp"])))
            ** 2,
            4,
        )
        # Take credence for each hypothesis, and weight each hypothesis to get the agent's overall belief.

    def update_evidence(node_data: Dict) -> Dict:
        """Update evidential component based on new evidence"""
        toss = np.random.binomial(1, model["truth"])
        node_data["brier_history"].append(
            calculate_brier_score(node_data["cred"], toss)
        )

        hyp = np.array(node_data["hyp"])
        cred = np.array(node_data["cred"])
        Pr_E_H = np.absolute((1 - toss) - hyp)
        posterior = Pr_E_H * cred / np.sum(cred * Pr_E_H)

        # Add noise
        loc = posterior
        scale = node_data["noise"]
        noisy = scipy.stats.truncnorm.rvs(
            (0.0001 - loc) / scale, (9.9999 - loc) / scale, loc=loc, scale=scale
        )
        node_data["evidential"] = (noisy / sum(noisy)).tolist()
        return node_data

    def update_social(node: int, node_data: Dict) -> Dict:
        """Update social component based on neighbors"""
        graph = model.get_graph()
        if model["scientist_type"] == "random":
            # Random scientist picks random neighbors
            neighbors = random.sample(
                # Exclude self from neighbors
                list(set(graph.nodes()) - {node}),
                max(1, round(len(graph.nodes()) * node_data["m"])),
            )
        else:
            # TR scientist picks best performing neighbors
            n = max(1, round(len(graph.nodes()) * node_data["m"]))
            neighbors = sorted(
                [n for n in graph.nodes() if n != node],  # Exclude self
                key=lambda x: (
                    sum(graph.nodes[x]["brier_history"])
                    / len(graph.nodes[x]["brier_history"])
                    if graph.nodes[x]["brier_history"]
                    else 1
                ),
            )[:n]

        # Calculate average credence of neighbors
        if neighbors:
            neighbor_creds = [np.array(graph.nodes[n]["cred"]) for n in neighbors]
            node_data["social"] = np.mean(neighbor_creds, axis=0).tolist()
        else:
            # If no neighbors, use current credence
            node_data["social"] = node_data["cred"]
        return node_data

    graph = model.get_graph()

    # First pass: update evidence and social components
    for node in graph.nodes():
        node_data = graph.nodes[node]
        node_data = update_evidence(node_data)
        node_data = update_social(node, node_data)

        # Update credence based on evidence and social information
        cred = np.array(node_data["cred"])
        social = np.array(node_data["social"])
        evidential = np.array(node_data["evidential"])
        c = node_data["c"]

        # Ensure new credence sums to 1
        new_cred = (1 - c) * social + c * evidential
        new_cred = new_cred / np.sum(new_cred)  # Normalize to sum to 1
        node_data["cred"] = new_cred.tolist()

        # Update Brier score
        t = np.zeros(len(node_data["hyp"]))
        t[int(model["truth"] * 5)] = 1
        node_data["brier_score"] = round(sum((new_cred - t) ** 2), 4)

        # Update track record if feedback is given
        if np.random.binomial(1, model["feedback_rate"]):
            node_data["record"].append(node_data["brier_history"][-1])

        graph.nodes[node].update(node_data)
    model.set_graph(graph)


def constructModel() -> AgentModel:
    model = AgentModel()
    model.set_initial_data_function(generateInitialData)
    model.set_timestep_function(generateTimestepData)
    model.update_parameters(
        {
            "num_nodes": 30,
            "graph_type": "complete",
            "convergence_data_key": "brier_score",
            "convergence_std_dev": 0.10,
            "truth": random.choice(np.round(np.arange(0, 1.001, 1 / 5), 2)),
            "feedback_rate": 1,
            "scientist_type": "tr",  # Can be 'tr', 'random', or 'patient'
        }
    )
    return model
