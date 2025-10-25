import os
from torch_geometric.utils import negative_sampling
os.environ["OMP_NUM_THREADS"] = "3"
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.cluster import AgglomerativeClustering
import json
import re
from collections import deque
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ----------------------- 读取ID映射 ----------------------- #
with open('index_to_asin.json', 'r', encoding='utf-8') as f:
    index_to_asin = json.load(f)

# ----------------------- 数据加载（保留这一段，删掉下面的重映射） ----------------------- #
item_features = np.load('item_features.npy')
graph_data = np.load('movie_graph_edges.npz')

edge_index = torch.from_numpy(graph_data['edge_index']).long()   # (2, E)
edge_weight = torch.from_numpy(graph_data['edge_weight']).float()
x = torch.from_numpy(item_features).float()

# 一致性自检
assert edge_index.max().item() < x.size(0), "edge_index 与特征维度不匹配"

# 反查表 key 是字符串
for _ in range(5):
    e = np.random.randint(edge_index.size(1))
    u = edge_index[0, e].item()
    v = edge_index[1, e].item()
    assert str(u) in index_to_asin and str(v) in index_to_asin, "index_to_asin 不覆盖图索引"


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
def contrastive_loss(embeddings, edge_index, tau=0.5):
    row, col = edge_index
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    pos_sim = torch.sum(embeddings_norm[row] * embeddings_norm[col], dim=-1) / tau
    pos_sim_exp = torch.exp(pos_sim)
    sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t()) / tau
    sim_matrix_exp_sum = torch.exp(sim_matrix).sum(dim=1)
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

def train_gcn(data, hidden_dim=32, out_dim=16, epochs=100, lr=0.01):
    model = GCN(data.x.size(1), hidden_dim, out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        embeddings = model(data.x, data.edge_index, data.edge_weight)
        src = embeddings[data.edge_index[0]]
        dst = embeddings[data.edge_index[1]]
        similarity = F.cosine_similarity(src, dst, dim=-1)
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
assert final_embeddings.shape[0] == x.size(0), "嵌入行数与节点数不一致"

# ----------------------- 使用层次聚类构建聚类结构 ----------------------- #
from scipy.cluster.hierarchy import linkage, fcluster

def build_cluster_structure_hierarchical(embeddings, n, max_depth=2):
    cluster_id = 0
    def recursive_cluster(indices, depth):
        nonlocal cluster_id
        if len(indices) <= n or depth >= max_depth:
            return {"indices": indices.tolist(), "leaf": True, "children": []}

        linkage_matrix = linkage(embeddings[indices], method='ward')
        labels = fcluster(linkage_matrix, t=2, criterion='maxclust')
        children = []
        for label in np.unique(labels):
            child_indices = indices[labels == label]
            child = recursive_cluster(child_indices, depth + 1)
            children.append(child)

        return {"indices": indices.tolist(), "leaf": False, "children": children}

    return recursive_cluster(np.arange(len(embeddings)), 0)

n = 500
"这里有一个非常重要的参数，即最大深度，它决定了聚类数的深度"
cluster_structure = build_cluster_structure_hierarchical(final_embeddings, n, max_depth=2)
structure_file = f"cluster_structure_{n}_hierarchical.json"
with open(structure_file, 'w', encoding='utf-8') as f:
    json.dump(cluster_structure, f, ensure_ascii=False, indent=4)
print(f"聚类结构已保存为 {structure_file}")

# ----------------------- 以下部分保持原样 ----------------------- #
from math import ceil, log

def assign_paths_bfs(tree_root, k, n):
    """
    为每个内部节点分配1个token（可回绕），为叶子里的每个样本分配“多位base-n”编码，保证全局唯一路径。
    n = 词表大小（<CI0>...<CI{n-1}>）
    """

    def int_to_base_n_digits(x, base, L):
        # 返回长度为 L 的“低位在后”的digits列表（每个digit是 0..base-1）
        ds = []
        for _ in range(L):
            ds.append(x % base)
            x //= base
        ds.reverse()
        return ds

    # 1) 先BFS把树铺开，并给“内部节点的每条出边”分配一个 token（单个位，回绕无妨）
    global_counter = 0
    def next_token():
        nonlocal global_counter
        t = global_counter % n
        global_counter += 1
        return f"<CI{t}>"

    queue = deque()
    cluster_tree = {}
    # (node_name, node_dict, parent_path)
    queue.append(("root", tree_root, []))

    while queue:
        node_name, node_dict, parent_path = queue.popleft()
        node_dict["path"] = parent_path
        cluster_tree[node_name] = node_dict

        if node_dict.get("leaf", False):
            continue

        # 给每个子节点的“入边”分配1个 token，并推入队列
        children = node_dict["children"]
        for i, child in enumerate(children):
            edge_tok = next_token()  # 单位token，回绕没关系，因为父路径不同
            sub_name = f"{node_name}-{i}"
            sub_path = parent_path + [edge_tok]
            queue.append((sub_name, child, sub_path))

    # 2) 对每个叶子：为其中每个样本分配 base-n 的多位编码，位数足够覆盖叶子大小，保证唯一
    total_items = 0
    uniq_tokens = set()
    for node_name, node_dict in cluster_tree.items():
        if not node_dict.get("leaf", False):
            continue
        indices = node_dict["indices"]
        m = len(indices)
        total_items += m
        node_dict["sample_paths"] = []

        if m == 0:
            continue

        # 需要的位数：最少1位；当 m > n 时自动增位
        L = max(1, ceil(log(m, n))) if m > 1 else 1

        for i in range(m):
            digits = int_to_base_n_digits(i, n, L)  # 长度L，每位是 0..n-1
            tail = [f"<CI{d}>" for d in digits]
            full_path = node_dict["path"] + tail
            node_dict["sample_paths"].append(full_path)
            uniq_tokens.add("".join(full_path))

    print(f"[PATH] 叶子样本总数={total_items}，唯一路径数={len(uniq_tokens)}")
    assert len(uniq_tokens) == total_items, "存在重复token路径，请检查！"

    return cluster_tree

cluster_tree = assign_paths_bfs(cluster_structure, k=10, n=n)
tree_file = f"cluster_tree_{n}_hierarchical.json"
with open(tree_file, 'w', encoding='utf-8') as f:
    json.dump(cluster_tree, f, ensure_ascii=False, indent=4)
print(f"路径编码后的聚类树已保存为 {tree_file}")

item_map = {}
for node, value in cluster_tree.items():
    if "sample_paths" in value:
        for i, idx in enumerate(value["indices"]):
            token = "".join(value["sample_paths"][i])
            original_item_id = index_to_asin[str(idx)]
            item_map[f"{original_item_id}"] = token

original_output_file = f"item_collaborative_indexing_original_{n}_hierarchical.txt"
with open(original_output_file, 'w') as f:
    for item_id, path in item_map.items():
        f.write(f"{item_id} {path}\n")
print(f"原始格式已保存至 {original_output_file}")

def process_item_index_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = re.sub(r'<CI0*([1-9]\d*)>', r'<CI\1>', line)
            line = re.sub(r'<CI0+>', r'<CI0>', line)
            outfile.write(line)

output_file = f"item_collaborative_indexing_{n}_2_sequential.txt"
process_item_index_file(original_output_file, output_file)
print(f"处理后的物品ID文件已保存为 {output_file}")