import networkx as nx
import numpy as np
from emergent.main import AgentModel
import random
import uuid
import scipy


def generateInitialData(model: AgentModel):
    hyp = np.round(np.arange(0, 1.001, 1 / 5), 2)
    initial_data = {
        "record": [],
        "m": round(random.uniform(0.05, 0.5), 2),  # Open-mindedness
        "model": model,
        "unique_id": uuid.uuid4(),
        "hyp": hyp,
        "cred": np.round(np.full(len(hyp), 1 / len(hyp)), 2),  # Credences
        "noise": random.uniform(0.001, 0.2),  # equivalent to sigma in paper
        "c": round(random.random(), 2),  # weight for evidence vs testimony
        "neighbors": [],
        "social": None,
        "evidential": None,
        "pr": 0,
        "ev": 0,
        "id": 0,
        "hub": 0,
        "authority": 0,
        "Brier": [],
        "BrierT": None,
        "crps": None,
    }
    return initial_data


def generateTimestepData(model: AgentModel):
    nodes = [node[1] for node in model.get_graph().nodes(data=True)]

    def _update_social(node_data: dict[str, any]):
        node_data["social"] = np.round(sum([]))

    def _update_evidence(node_data: dict[str, any]):
        toss = np.random.binomial(1, model["truth"])
        # Credence at previous time step against new toss
        node_data["Brier"].append(
            round((toss - sum(node_data["cred"] * node_data["hyp"])) ** 2, 4)
        )
        Pr_E_H = np.absolute((1 - toss) - node_data["hyp"])
        posterior = Pr_E_H * node_data["cred"] / np.sum(node_data["cred"] * Pr_E_H)
        loc = posterior
        scale = node_data["noise"]
        # No negative credences
        noisy = scipy.stats.truncnorm.rvs(
            (0.0001 - loc) / scale, (9.9999 - loc) / scale, loc=loc, scale=scale
        )
        node_data["evidential"] = noisy / sum(noisy)

    def _r_avg(node_data: dict[str, any]):
        if len(node_data["record"]) > 0:
            return round(sum(node_data["record"]) / len(node_data["record"]), 4)
        else:
            return 1

    def _update_neighbors(node_data: dict[str, any]):
        n = round(model["num_nodes"] * node_data["m"])
        if n < 1:
            # Agent trusts no one
            node_data["neighbors"] = [node_data]
        elif len(node_data["record"]) == 0:
            # No track records yet
            node_data["neighbors"] = random.sample(nodes, n)
        else:
            # Choose the best performing agents so far
            temp = []
            ls = nodes
            random.shuffle(ls)
            temp = sorted(ls, key=lambda x: _r_avg(node_data))[:n]
            if len(temp) < 1:
                temp.append(node_data)
            node_data["neighbors"] = temp

    for node in nodes:
        _update_evidence(node)
        _update_neighbors(node)
        _update_social(node)


def constructModel() -> AgentModel:
    model = AgentModel()
    t = random.choice(np.round(np.arange(0, 1.001, 1 / 5), 2))
    f = 1
    model.update_parameters(
        {
            "num_nodes": 30,
            "graph_type": "complete",
            "true_bias": 0.6,
            "truth": t,
            "feedback_rate": f,
        }
    )

    model.set_initial_data_function(generateInitialData)
    model.set_timestep_function(generateTimestepData)

    return model
