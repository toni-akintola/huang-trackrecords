from model import HuangTrackRecord
import statistics
import networkx as nx

# Get all keys that have numerical values in the graph
model = HuangTrackRecord()
model.initialize_graph()

print("GRAPH_NODES:", model.graph.nodes(data=True, default={}))
print("GRAPH_EDGES:", model.graph.edges(data=True))

model.timestep()
print("GRAPH_NODES:", model.graph.nodes(data=True, default={}))
