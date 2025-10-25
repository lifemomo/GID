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
import torch_geometric.utils as utils
from torch_geometric.utils import negative_sampling

# ----------------------- 数据加载 ----------------------- #
item_features = np.load('item_features.npy')  # 物品特征
graph_data = np.load('movie_graph_edges.npz', allow_pickle=True)  # 图数据

print("Item features shape:", item_features.shape)
print("Number of items in graph_data:", len(graph_data['edge_index'][0]))

# 数据预处理：确保物品ID连续
original_item_ids = np.unique(graph_data['edge_index'])
id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(original_item_ids))}
reverse_id_mapping = {new_id: old_id for old_id, new_id in id_mapping.items()}

# 更新edge_index，使物品ID连续
edge_index = torch.tensor([[id_mapping[i] for i in edge_pair] for edge_pair in graph_data['edge_index'].T],
                          dtype=torch.long).T

edge_weight = torch.tensor(graph_data['edge_weight'], dtype=torch.float)
x = torch.tensor(item_features, dtype=torch.float)

# 创建图数据对象
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
    """一般来说，GCN 的学习率可以设置在 0.001 到 0.01 之间进行调节。如果发现模型收敛较慢，可以逐渐调小学习率。如果模型训练出现震荡或发散，可以尝试降低学习率。
    隐藏层维度决定了模型的表示能力。较大的隐藏层维度可以让模型学习更复杂的特征表示，但也可能增加计算开销，并导致 过拟合。较小的隐藏层维度可能导致 欠拟合，无法充分学习数据中的复杂模式。
    """
    model = GCN(data.x.size(1), hidden_dim, out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 不再使用MSELoss，改为更合适的损失函数，如余弦相似度损失或结构损失
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        embeddings = model(data.x, data.edge_index, data.edge_weight)

        # 计算节点嵌入的相似性（可以用余弦相似度或其他）
        similarity = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        loss = -similarity.mean()  # 尝试最大化相似性
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    return embeddings.detach().numpy()

final_embeddings = train_gcn(data) #使用余弦相似度的负均值作为损失函数


# ----------------------- 递归聚类函数 ----------------------- #
def recursive_clustering(embeddings, indices, k, n, cluster_tree, node_name, path, depth=0, max_depth=10):
    if depth >= max_depth or len(indices) <= n:
        cluster_tree[node_name] = {"indices": indices.tolist(), "path": path}
        return

    current_embeddings = embeddings[indices]
    unique_embeddings = np.unique(current_embeddings, axis=0)

    if len(unique_embeddings) < k:
        cluster_tree[node_name] = {"indices": indices.tolist(), "path": path}
        return

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(current_embeddings)
    actual_k = len(set(labels))

    if actual_k < k:
        cluster_tree[node_name] = {"indices": indices.tolist(), "path": path}
        return

    # For each cluster, generate a unique path suffix
    for i in range(actual_k):
        sub_indices = indices[labels == i]
        sub_node_name = f"{node_name}-{i}"
        sub_path = path + [f"<CI{depth}{i:02}>"]  # Add the current node to the path with leading zeroes
        recursive_clustering(
            embeddings, sub_indices, k, n, cluster_tree, sub_node_name, sub_path, depth + 1, max_depth
        )

# ----------------------- 参数设定 ----------------------- #
k = 10
n = 100

# ----------------------- 执行递归聚类 ----------------------- #
cluster_tree = {}
recursive_clustering(final_embeddings, np.arange(len(final_embeddings)), k=k, n=n, cluster_tree=cluster_tree,
                     node_name="root", path=[])

# ----------------------- 保存聚类结构 ----------------------- #
tree_file = f"cluster_tree_{n}_{k}.json"
with open(tree_file, 'w', encoding='utf-8') as f:
    json.dump(cluster_tree, f, ensure_ascii=False, indent=4)

print(f"递归聚类结果已保存到 {tree_file}")

# ----------------------- 生成物品ID并保存 ----------------------- #
item_map = {}
index_counter = {"root": 0}  # To track the last index for each cluster

# Add the final recursive indexing with path
for node, value in cluster_tree.items():
    path_tokens = value["path"]  # 不要自己join
    for idx in value["indices"]:
        index_counter[node] = (index_counter.get(node, -1) + 1) % n  # ✅ 使用模运算
        final_token = f"<CI{index_counter[node]}>"

        item_token = "".join(path_tokens + [final_token])
        original_item_id = reverse_id_mapping[idx]
        item_map[f"{original_item_id}"] = item_token


# ----------------------- 保存原始映射 ----------------------- #
original_output_file = f"item_collaborative_indexing_original_{n}_{k}.txt"
with open(original_output_file, 'w') as f:
    for item_id, path in item_map.items():
        f.write(f"{item_id} {path}\n")

print(f"原始格式已保存至 {original_output_file}")


import re
# 逐个读取文件，修改物品ID格式
def process_item_index_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # 使用正则表达式替换 CI00x 为 CIx，CI0xy 为 CIxy，CI000 为 CI0
            line = re.sub(r'<CI0*([1-9]\d*)>', r'<CI\1>', line)
            line = re.sub(r'<CI0+>', r'<CI0>', line)

            # 将处理后的行写入输出文件
            outfile.write(line)


# 示例：将原始的 item_collaborative_indexing.txt 文件中的编号格式修正后保存到新文件中
output_file = f"item_collaborative_indexing_{n}_{k}_sequential.txt"
process_item_index_file(original_output_file, output_file)

print(f"处理后的物品ID文件已保存为 {output_file}")
