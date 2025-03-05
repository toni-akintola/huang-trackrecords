import unittest
import numpy as np
from model import (
    ScientistModel,
)  # Assuming the implementation is in 'scientist_model.py'
import networkx as nx


class TestScientistModel(unittest.TestCase):

    def setUp(self):
        """Initialize a ScientistModel instance for testing."""
        self.model = ScientistModel()
        self.model.initialize_graph()
        self.graph = self.model.get_graph()
        self.agent = self.graph.nodes[0]  # Pick a sample agent for testing

    def test_initialization(self):
        """Test that agents are initialized with correct parameters."""
        self.assertIn("cred", self.agent)
        self.assertEqual(len(self.agent["cred"]), 6)  # Hypothesis space of size 6
        self.assertAlmostEqual(sum(self.agent["cred"]), 1, places=2)

    def test_brier_score_calculation(self):
        """Verify that the Brier score is computed correctly."""
        cred = [0.1, 0.1, 0.2, 0.3, 0.2, 0.1]  # Example belief distribution
        toss = 1  # Assume the true outcome is heads
        score = self.model.calculate_brier_score(cred, toss)
        expected_score = round(
            (toss - sum(np.array(cred) * np.array(self.agent["hyp"]))) ** 2, 4
        )
        self.assertAlmostEqual(score, expected_score, places=4)

    def test_update_evidence(self):
        """Check that evidence updates the agent's beliefs properly."""
        before_update = self.agent["cred"][:]
        self.model.update_evidence(self.agent)
        after_update = self.agent["evidential"]
        self.assertNotEqual(before_update, after_update)
        self.assertAlmostEqual(sum(after_update), 1, places=2)

    def test_update_social(self):
        """Ensure social updates consider the correct set of informants."""
        node_id = list(self.graph.nodes())[0]
        self.model.update_social(node_id, self.agent)
        after_update = self.model.get_graph().nodes[node_id]["social"]
        self.assertAlmostEqual(sum(after_update), 1, places=1)


if __name__ == "__main__":
    unittest.main()
