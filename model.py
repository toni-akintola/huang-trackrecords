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
        "c": (
            round(random.random(), 2)
            if model["model_variation"] != "patient-scientist"
            else 1
        ),
        "social": None,
        "evidential": None,
        "brier_score": 1,
        "brier_history": [],
        "r_value": 0.0,  # Correlation between centrality and accuracy (community-wide metric)
    }


def generateTimestepData(model: AgentModel):
    """Main timestep function"""

    def calculate_brier_score(node_data: Dict, toss: int) -> float:
        """Calculate Brier score for a prediction"""
        return round(
            (toss - sum(np.array(node_data["cred"]) * np.array(node_data["hyp"]))) ** 2,
            4,
        )
        # Take credence for each hypothesis, and weight each hypothesis to get the agent's overall belief.

    def update_evidence(node_data: Dict) -> Dict:
        """Update evidential component based on new evidence"""
        toss = np.random.binomial(1, model["truth"])
        node_data["brier_history"].append(calculate_brier_score(node_data, toss))

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

    def update_social(node: int, node_data: Dict, trust_network: Dict) -> Dict:
        """Update social component based on neighbors"""
        graph = model.get_graph()
        if model["model_variation"] == "random-scientist":
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

        # Track trust relations for network analysis
        trust_network[node] = neighbors

        # Calculate average credence of neighbors
        if neighbors:
            neighbor_creds = [np.array(graph.nodes[n]["cred"]) for n in neighbors]
            node_data["social"] = np.mean(neighbor_creds, axis=0).tolist()
        else:
            # If no neighbors, use current credence
            node_data["social"] = node_data["cred"]
        return node_data

    graph = model.get_graph()
    
    # Track trust relations for computing centrality
    trust_network = {}

    # First pass: update evidence and social components
    for node in graph.nodes():
        node_data = graph.nodes[node]
        node_data = update_evidence(node_data)
        node_data = update_social(node, node_data, trust_network)

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
    
    # Compute and track metrics: R-value and average Brier score
    # Build directed graph from trust relations for HITS algorithm
    trust_graph = nx.DiGraph()
    trust_graph.add_nodes_from(graph.nodes())
    for node, informants in trust_network.items():
        for informant in informants:
            trust_graph.add_edge(node, informant)  # node trusts informant
    
    # Compute authority scores using HITS algorithm
    if len(trust_graph.edges()) > 0:
        try:
            hubs, authorities = nx.hits(trust_graph, max_iter=100, normalized=True)
            authority_scores = [authorities.get(node, 0.0) for node in graph.nodes()]
        except:
            # Fallback if HITS fails (e.g., disconnected graph)
            authority_scores = [0.0] * len(graph.nodes())
    else:
        # No trust relations yet
        authority_scores = [0.0] * len(graph.nodes())
    
    # Get Brier scores (error measure: lower is better)
    # Note: In the code, brier_score is the sum of squared errors, not the full Brier accuracy formula
    brier_errors = [graph.nodes[node]["brier_score"] for node in graph.nodes()]
    
    # Convert to Brier accuracy (higher is better) for correlation with authority
    # The paper defines Brier accuracy as: 1 - (1/n) * sum((P(Hj) - I(Hj))^2)
    # Since brier_score in code is the sum term, we normalize it
    # For correlation purposes, we want to see if higher authority = higher accuracy
    # So we'll use accuracy = 1 - normalized_error, or just correlate with -error
    num_hypotheses = len(graph.nodes[list(graph.nodes())[0]]["hyp"])
    brier_accuracies = [1 - (bs / num_hypotheses) for bs in brier_errors]
    
    # Compute correlation (R-value) between authority scores and Brier accuracy
    # Higher R-value indicates better meta-expertise (recognition tracks accuracy)
    # This is a community-wide metric, but stored on each node for tracking
    if (len(authority_scores) > 1 and 
        np.std(authority_scores) > 1e-10 and 
        np.std(brier_accuracies) > 1e-10):
        r_value = np.corrcoef(authority_scores, brier_accuracies)[0, 1]
        if np.isnan(r_value):
            r_value = 0.0
    else:
        r_value = 0.0
    
    # Store R-value and Brier score as node attributes (data items)
    for node in graph.nodes():
        graph.nodes[node]["r_value"] = float(r_value)
        # brier_score is already stored on each node
    
    model.set_graph(graph)


def constructModel() -> AgentModel:
    model = AgentModel()
    model.set_initial_data_function(generateInitialData)
    model.set_timestep_function(generateTimestepData)
    model.update_parameters(
        {
            "num_nodes": 30,
            "graph_type": "complete",
            "truth": random.choice(np.round(np.arange(0, 1.001, 1 / 5), 2)),
            "feedback_rate": 1,
            "model_variation": "tr",  # Can be 'tr', 'random', or 'patient'
        }
    )
    model["variations"] = ["tr-scientist", "random-scientist", "patient-scientist"]
    
    return model
