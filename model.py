import random
import networkx as nx
import numpy as np
import scipy.stats
from typing import Dict, List
from emergent.main import AgentModel


class ScientistModel(AgentModel):
    def __init__(self):
        super().__init__()
        self.update_parameters(
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
        self.set_initial_data_function(self.initialize_scientist)
        self.set_timestep_function(self.update_scientists)

    def initialize_scientist(self, model: AgentModel) -> Dict:
        """Initialize a scientist with appropriate attributes"""
        return {
            "record": [],  # Track record
            "m": round(random.uniform(0.05, 0.5), 2),  # Open-mindedness
            "hyp": np.round(np.arange(0, 1.001, 1 / 5), 2).tolist(),
            # Credence for each hyp
            "cred": np.round(np.full(6, 1 / 6), 2).tolist(),
            "noise": random.uniform(0.001, 0.2),
            "c": (
                round(random.random(), 2) if model["scientist_type"] != "patient" else 1
            ),
            "social": None,
            "evidential": None,
            "brier_score": 1,
            "brier_history": [],
        }

    def calculate_brier_score(self, cred: List[float], toss: int) -> float:
        """Calculate Brier score for a prediction"""
        return round(
            (toss - sum(np.array(cred) * np.array(self.get_graph().nodes[0]["hyp"])))
            ** 2,
            4,
        )

    def update_evidence(self, node_data: Dict) -> Dict:
        """Update evidential component based on new evidence"""
        toss = np.random.binomial(1, self["truth"])
        node_data["brier_history"].append(
            self.calculate_brier_score(node_data["cred"], toss)
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

    def update_social(self, node: int, node_data: Dict) -> Dict:
        """Update social component based on neighbors"""
        graph = self.get_graph()
        if self["scientist_type"] == "random":
            # Random scientist picks random neighbors
            neighbors = random.sample(
                list(graph.nodes()), max(1, round(len(graph.nodes()) * node_data["m"]))
            )
        else:
            # TR scientist picks best performing neighbors
            n = max(1, round(len(graph.nodes()) * node_data["m"]))
            neighbors = sorted(
                graph.nodes(),
                key=lambda x: (
                    sum(graph.nodes[x]["brier_history"])
                    / len(graph.nodes[x]["brier_history"])
                    if graph.nodes[x]["brier_history"]
                    else 1
                ),
            )[:n]

        # Calculate average credence of neighbors
        neighbor_creds = [np.array(graph.nodes[n]["cred"]) for n in neighbors]
        node_data["social"] = np.mean(neighbor_creds, axis=0).tolist()
        return node_data

    def update_scientists(self, model: AgentModel):
        """Main timestep function"""
        graph = model.get_graph()

        # First pass: update evidence and social components
        for node in graph.nodes():
            node_data = graph.nodes[node]
            node_data = self.update_evidence(node_data)
            node_data = self.update_social(node, node_data)

            # Update credence based on evidence and social information
            cred = np.array(node_data["cred"])
            social = np.array(node_data["social"])
            evidential = np.array(node_data["evidential"])
            c = node_data["c"]

            new_cred = np.round((1 - c) * social + c * evidential, 2)
            node_data["cred"] = new_cred.tolist()

            # Update Brier score
            t = np.zeros(len(node_data["hyp"]))
            t[int(model["truth"] * 5)] = 1
            node_data["brier_score"] = round(sum((new_cred - t) ** 2), 4)

            # Update track record if feedback is given
            if np.random.binomial(1, model["feedback_rate"]):
                node_data["record"].append(node_data["brier_history"][-1])

            graph.nodes[node].update(node_data)


def constructModel() -> AgentModel:
    """Run a simulation with specified scientist type and steps"""
    model = ScientistModel()

    return model


# Run a simulation with TR scientists
# tr_model = run_simulation(scientist_type="tr", steps=100)

# Run a simulation with random scientists
# random_model = run_simulation(scientist_type="random", steps=100)

# Run a simulation with patient scientists
# patient_model = run_simulation(scientist_type="patient", steps=100)
