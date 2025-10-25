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
from collections import deque
from torch_geometric.utils import negative_sampling

# ----------------------- 数据加载 ----------------------- #
item_features = np.load('item_features.npy')
graph_data = np.load('movie_graph_edges.npz', allow_pickle=True)

original_item_ids = np.unique(graph_data['edge_index'])
id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(original_item_ids))}
reverse_id_mapping = {new_id: old_id for old_id, new_id in id_mapping.items()}

edge_index = torch.tensor([[id_mapping[i] for i in edge_pair] for edge_pair in graph_data['edge_index'].T],
                          dtype=torch.long).T
edge_weight = torch.tensor(graph_data['edge_weight'], dtype=torch.float)
x = torch.tensor(item_features, dtype=torch.float)
data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)


# 定义对比损失函数
def contrastive_loss(embeddings, edge_index, tau=0.5):
    row, col = edge_index
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)

    # 正样本相似度
    pos_sim = torch.sum(embeddings_norm[row] * embeddings_norm[col], dim=-1) / tau
    pos_sim_exp = torch.exp(pos_sim)

    # 计算所有节点之间的相似度矩阵
    sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t()) / tau
    sim_matrix_exp_sum = torch.exp(sim_matrix).sum(dim=1)

    # InfoNCE 损失
    loss = -torch.log(pos_sim_exp / (sim_matrix_exp_sum[row] - pos_sim_exp + 1e-8)).mean()

    return loss

# GAE重构损失函数
def reconstruction_loss(embeddings, edge_index, num_nodes):
    # 正样本的预测（真实存在的边）
    pos_pred = (embeddings[edge_index[0]] * embeddings[edge_index[1]]).sum(dim=1)
    pos_loss = -torch.log(torch.sigmoid(pos_pred) + 1e-8).mean()

    # 负样本的预测（随机采样不存在的边）
    neg_edge_index = negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=edge_index.size(1))
    neg_pred = (embeddings[neg_edge_index[0]] * embeddings[neg_edge_index[1]]).sum(dim=1)
    neg_loss = -torch.log(1 - torch.sigmoid(neg_pred) + 1e-8).mean()

    loss = pos_loss + neg_loss
    return loss

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
def train_gcn(data, hidden_dim=32, out_dim=16, epochs=100, lr=0.01):
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

def train_gcn_contrastive_loss(data, hidden_dim=32, out_dim=16, epochs=100, lr=0.01, patience=10, tau=0.5):
    model = GCN(data.x.size(1), hidden_dim, out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        embeddings = model(data.x, data.edge_index, data.edge_weight)

        loss = contrastive_loss(embeddings, data.edge_index, tau=tau)

        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        print(f'Epoch {epoch}, Loss: {current_loss:.4f}')

        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            best_embeddings = embeddings.detach().numpy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

    return best_embeddings

 # 更新训练函数 (GAE重构损失版)
def train_gcn_gae(data, hidden_dim=32, out_dim=16, epochs=200, lr=0.01, patience=10):
    model = GCN(data.x.size(1), hidden_dim, out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        embeddings = model(data.x, data.edge_index, data.edge_weight)

        loss = reconstruction_loss(embeddings, data.edge_index, data.num_nodes)

        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        print(f'Epoch {epoch}, Loss: {current_loss:.4f}')

        # Early Stopping
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            best_embeddings = embeddings.detach().numpy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

    return best_embeddings


# final_embeddings = train_gcn(data) #使用余弦相似度的负均值作为损失函数
final_embeddings = train_gcn_contrastive_loss(data)  #使用对比损失函数
# final_embeddings = train_gcn_gae(data)  # 使用GAE重构损失

# ----------------------- 聚类参数 ----------------------- #
k = 30
n = 500

# ----------------------- 构建聚类结构（不含路径） ----------------------- #
def build_cluster_structure(embeddings, indices, k, n, max_depth=10, depth=0):
    if len(indices) <= n or depth >= max_depth:
        return {"indices": indices.tolist(), "leaf": True, "children": []}
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings[indices])
    children = []
    for i in range(k):
        sub_indices = indices[labels == i]
        sub_node = build_cluster_structure(embeddings, sub_indices, k, n, max_depth, depth + 1)
        children.append(sub_node)
    return {"indices": indices.tolist(), "leaf": False, "children": children}

cluster_structure = build_cluster_structure(final_embeddings, np.arange(len(final_embeddings)), k, n)

# 保存纯聚类结构
structure_file = f"cluster_structure_{n}_{k}.json"
with open(structure_file, 'w', encoding='utf-8') as f:
    json.dump(cluster_structure, f, ensure_ascii=False, indent=4)
print(f"聚类结构已保存为 {structure_file}")

# ----------------------- 编码聚类结构 ----------------------- #
def assign_paths_bfs(tree_root, k, n):
    global_counter = [0]
    def get_next_index():
        idx = global_counter[0]
        global_counter[0] = (global_counter[0] + 1) % n
        return f"<CI{idx}>"

    queue = deque()
    cluster_tree = {}
    queue.append(("root", tree_root, []))

    while queue:
        level = list(queue)
        queue.clear()

        child_nodes_to_assign = []
        sample_nodes_to_assign = []

        for node_name, node_dict, parent_path in level:
            node_dict["path"] = parent_path
            cluster_tree[node_name] = node_dict

            if node_dict["leaf"]:
                node_dict["sample_paths"] = [["__PENDING__"] for _ in node_dict["indices"]]
                sample_nodes_to_assign.append((node_name, node_dict))
            else:
                for i, child in enumerate(node_dict["children"]):
                    sub_node_name = f"{node_name}-{i}"
                    sub_path = parent_path + [None]
                    cluster_tree[sub_node_name] = child
                    queue.append((sub_node_name, child, sub_path))
                    child_nodes_to_assign.append(sub_path)

        for path in child_nodes_to_assign:
            path[-1] = get_next_index()

        for node_name, node_dict in sample_nodes_to_assign:
            node_dict["sample_paths"] = []
            for _ in node_dict["indices"]:
                node_dict["sample_paths"].append(node_dict["path"] + [get_next_index()])

    return cluster_tree

# 编码并保存
cluster_tree = assign_paths_bfs(cluster_structure, k, n)
tree_file = f"cluster_tree_{n}_{k}.json"
with open(tree_file, 'w', encoding='utf-8') as f:
    json.dump(cluster_tree, f, ensure_ascii=False, indent=4)
print(f"路径编码后的聚类树已保存为 {tree_file}")

# ----------------------- 保存物品ID映射 ----------------------- #
item_map = {}
for node, value in cluster_tree.items():
    if "sample_paths" in value:
        for i, idx in enumerate(value["indices"]):
            token = "".join(value["sample_paths"][i])
            original_item_id = reverse_id_mapping[idx]
            item_map[f"{original_item_id}"] = token

original_output_file = f"item_collaborative_indexing_original_{n}_{k}.txt"
with open(original_output_file, 'w') as f:
    for item_id, path in item_map.items():
        f.write(f"{item_id} {path}\n")
print(f"原始格式已保存至 {original_output_file}")

# ----------------------- 格式标准化 ----------------------- #
def process_item_index_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = re.sub(r'<CI0*([1-9]\d*)>', r'<CI\1>', line)
            line = re.sub(r'<CI0+>', r'<CI0>', line)
            outfile.write(line)

output_file = f"item_collaborative_indexing_{n}_{k}_sequential.txt"
process_item_index_file(original_output_file, output_file)
print(f"处理后的物品ID文件已保存为 {output_file}")
