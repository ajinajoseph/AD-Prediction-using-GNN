import os
import torch
import networkx as nx
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
graph_path = os.path.join(BASE_DIR, "graphs", "CN", "002_S_1280.pt")

graph = torch.load(graph_path)
G = nx.Graph()

edges = graph.edge_index.numpy().T
G.add_edges_from(edges)
plt.figure(figsize=(10, 10))

nx.draw_networkx(
    G,
    node_size=20,
    with_labels=False,
    node_color="skyblue",
    edge_color="gray"
)
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

components = list(nx.connected_components(G))
print("Connected components:", len(components))

for i, comp in enumerate(components):
    print(f"Component {i+1}: {len(comp)} nodes")
plt.title("Brain Connectivity Graph")
plt.axis("off")
plt.show()