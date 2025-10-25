import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans
import json
import re

# ----------------------- 数据加载 ----------------------- #
item_features = np.load('item_features.npy')  # 物品特征
graph_data = np.load('movie_graph_edges.npz', allow_pickle=True)  # 图数据

original_item_ids = np.unique(graph_data['edge_index'])
id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(original_item_ids))}
reverse_id_mapping = {new_id: old_id for old_id, new_id in id_mapping.items()}

edge_index = torch.tensor([[id_mapping[i] for i in edge_pair] for edge_pair in graph_data['edge_index'].T],
                          dtype=torch.long).T
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

# ----------------------- 训练GCN ----------------------- #
def train_gcn(data, hidden_dim=32, out_dim=16, epochs=10, lr=0.01):
    model = GCN(data.x.size(1), hidden_dim, out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        embeddings = model(data.x, data.edge_index, data.edge_weight)
        similarity = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        loss = -similarity.mean()
        loss.backward()
        optimizer.step()
    return embeddings.detach().numpy()

final_embeddings = train_gcn(data)

# ----------------------- 聚类路径参数 ----------------------- #
k = 10
n = 100
cluster_tree = {}
global_counter = [k]  # 跳过 root-0 ~ root-(k-1)

# ----------------------- path 编号器 ----------------------- #
def get_next_index():
    idx = global_counter[0]
    global_counter[0] = (global_counter[0] + 1) % n
    return f"<CI{idx}>"

# ----------------------- 主聚类函数 ----------------------- #
def recursive_clustering_strict(embeddings, indices, node_name, path_prefix, depth=0, max_depth=10):
    if len(indices) <= n or depth >= max_depth:
        # 叶子节点：暂存 path，稍后统一分配 sample_paths
        cluster_tree[node_name] = {
            "indices": indices.tolist(),
            "path": path_prefix,
            "leaf": True
        }
        return

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings[indices])
    cluster_tree[node_name] = {"indices": indices.tolist(), "path": path_prefix}
    children = []

    for i in range(k):
        sub_indices = indices[labels == i]
        sub_node_name = f"{node_name}-{i}"
        if depth == 0:
            sub_path = [f"<CI{i}>"]
        else:
            sub_path = path_prefix + [None]  # 占位
        cluster_tree[sub_node_name] = {
            "indices": sub_indices.tolist(),
            "path": sub_path
        }
        children.append((sub_node_name, sub_indices, sub_path))

    # 左到右为子节点统一分配 path[-1]
    for _, _, path in children:
        path[-1] = get_next_index()

    # 递归处理每个子节点
    for sub_node_name, sub_indices, sub_path in children:
        recursive_clustering_strict(embeddings, np.array(sub_indices), sub_node_name, sub_path, depth + 1, max_depth)

# ----------------------- 执行聚类 ----------------------- #
recursive_clustering_strict(final_embeddings, np.arange(len(final_embeddings)), "root", [], 0)

# ----------------------- 为叶子节点分配 sample_paths ----------------------- #
for node, value in cluster_tree.items():
    if value.get("leaf"):
        value["sample_paths"] = []
        for _ in value["indices"]:
            value["sample_paths"].append(value["path"] + [get_next_index()])

# ----------------------- 保存聚类结构 ----------------------- #
tree_file = f"cluster_tree_{n}_{k}.json"
with open(tree_file, 'w', encoding='utf-8') as f:
    json.dump(cluster_tree, f, ensure_ascii=False, indent=4)

print(f"递归聚类结果已保存到 {tree_file}")

# ----------------------- 生成 item path 映射 ----------------------- #
item_map = {}
for node, value in cluster_tree.items():
    if "sample_paths" in value:
        for i, idx in enumerate(value["indices"]):
            token = "".join(value["sample_paths"][i])
            original_item_id = reverse_id_mapping[idx]
            item_map[f"{original_item_id}"] = token

# ----------------------- 保存原始映射 ----------------------- #
original_output_file = f"item_collaborative_indexing_original_{n}_{k}.txt"
with open(original_output_file, 'w') as f:
    for item_id, path in item_map.items():
        f.write(f"{item_id} {path}\n")

print(f"原始格式已保存至 {original_output_file}")

# ----------------------- 标准化 path 编码格式 ----------------------- #
def process_item_index_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = re.sub(r'<CI0*([1-9]\d*)>', r'<CI\1>', line)
            line = re.sub(r'<CI0+>', r'<CI0>', line)
            outfile.write(line)

output_file = f"item_collaborative_indexing_{n}_{k}_sequential.txt"
process_item_index_file(original_output_file, output_file)

print(f"处理后的物品ID文件已保存为 {output_file}")
