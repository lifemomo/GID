import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.cluster import AgglomerativeClustering
import json
import re

os.environ["OMP_NUM_THREADS"] = "1"

# ----------------------- 数据加载 ----------------------- #
item_features = np.load('item_features.npy')
graph_data = np.load('movie_graph_edges.npz', allow_pickle=True)
valid_item_ids = np.load('valid_item_ids.npy')

# ----------------------- ID映射处理 ----------------------- #
original_item_ids = np.unique(graph_data['edge_index'])
id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(original_item_ids))}
edge_index = torch.tensor([[id_mapping[i] for i in edge_pair] for edge_pair in graph_data['edge_index'].T], dtype=torch.long).T
edge_weight = torch.tensor(graph_data['edge_weight'], dtype=torch.float)
x = torch.tensor(item_features, dtype=torch.float)
data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

# ----------------------- GCN模型定义 ----------------------- #
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.conv2(x, edge_index, edge_weight)
        return x

def train_gcn(data, hidden_dim=32, out_dim=16, epochs=100, lr=0.01):
    model = GCN(data.x.size(1), hidden_dim, out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        embeddings = model(data.x, data.edge_index, data.edge_weight)
        src = embeddings[data.edge_index[0]]
        dst = embeddings[data.edge_index[1]]
        loss = -F.cosine_similarity(src, dst, dim=-1).mean()
        loss.backward()
        optimizer.step()
    return embeddings.detach().numpy()

final_embeddings = train_gcn(data)

# ----------------------- 层次聚类 ----------------------- #
k = 20
clusterer = AgglomerativeClustering(n_clusters=k)
labels = clusterer.fit_predict(final_embeddings)

# ----------------------- 构建 item ↔ token 映射 ----------------------- #
item_map = {}
cluster_members = {i: [] for i in range(k)}
for item_id, label in zip(valid_item_ids, labels):
    cluster_members[label].append(item_id)

for label, items in cluster_members.items():
    for i, item_id in enumerate(items):
        item_map[str(item_id)] = f"<CI{label}><CI{i:02}>"

# ----------------------- 保存原始编码 ----------------------- #
original_output_file = f"item_collaborative_indexing_original_hierarchical_{k}.txt"
with open(original_output_file, 'w') as f:
    for item_id, path in item_map.items():
        f.write(f"{item_id} {path}\n")
print(f"原始格式已保存至 {original_output_file}")

# ----------------------- 简化编码格式 ----------------------- #
def simplify_token_format(line):
    line = re.sub(r'<CI0*([1-9]\d*)>', r'<CI\1>', line)
    line = re.sub(r'<CI0+0>', r'<CI0>', line)
    return line

final_output_file = f"item_collaborative_indexing_hierarchical_{k}_sequential.txt"
with open(original_output_file, 'r') as infile, open(final_output_file, 'w') as outfile:
    for line in infile:
        outfile.write(simplify_token_format(line))
print(f"简化版本已保存至 {final_output_file}")
