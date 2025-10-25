"任务1.预处理数据，方便后续的GCN处理及聚类"
import json
import numpy as np
import torch
from itertools import combinations
from collections import defaultdict
#
# # 路径设置
# user_sequence_path = 'user_sequence.txt'
# item_attr_json_path = 'LastFM_item2attributes.json'
#
# # 读取用户序列文件（格式：user_id item1 item2 ...）
# user_sequences = {}
# with open(user_sequence_path, 'r') as f:
#     for line in f:
#         tokens = line.strip().split()
#         if len(tokens) < 2:
#             continue
#         user_id = int(tokens[0])
#         item_ids = list(map(int, tokens[1:]))
#         user_sequences[user_id] = item_ids
#
# # 读取艺术家属性（json）
# with open(item_attr_json_path, 'r') as f:
#     item_attr_dict = json.load(f)
#
# # 获取所有出现在交互记录中的 item_id
# all_items = set()
# for items in user_sequences.values():
#     all_items.update(items)
#
# # 打印交互记录中的物品数量和ID
# # print(f"总共出现的物品数量：{len(all_items)}")
# # print(f"交互记录中的物品ID示例：{list(all_items)[:10]}")  # 打印前10个物品ID示例
#
# # 筛选出在属性文件中也存在的物品，并排除ID为0的项
# valid_items = sorted([int(i) for i in all_items if str(i) in item_attr_dict and int(i) != 0])
#
# # print(f"有效物品数量：{len(valid_items)}")
# # print(f"有效物品ID示例：{valid_items[:10]}")  # 打印前10个有效物品ID示例
#
# # 创建一个物品ID到索引的映射，保持原始物品ID
# item_id_to_index = {item_id: idx for idx, item_id in enumerate(valid_items)}
#
# # 打印物品ID到索引的映射
# # print(f"物品ID到索引的映射：{list(item_id_to_index.items())[:10]}")  # 打印前10个物品ID映射
#
# # 获取所有出现过的属性
# all_attr_ids = set()
# for item_id in valid_items:
#     all_attr_ids.update(item_attr_dict[str(item_id)])
# all_attr_ids = sorted(all_attr_ids)
# attr_id_to_index = {aid: i for i, aid in enumerate(all_attr_ids)}
#
# # 打印属性ID到索引的映射
# # print(f"属性ID到索引的映射：{list(attr_id_to_index.items())[:10]}")  # 打印前10个属性ID映射
#
# # 构建稀疏多热编码矩阵
# item_features = np.zeros((len(valid_items), len(all_attr_ids)), dtype=np.float32)
#
# # 使用原始物品ID作为索引，保持物品ID一致
# for i, item_id in enumerate(valid_items):
#     for aid in item_attr_dict[str(item_id)]:
#         item_features[i][attr_id_to_index[aid]] = 1.0
#
# # 保存 item_features.npy
# np.save('item_features.npy', item_features)
#
# # 打印保存后的矩阵大小
# print(f"item_features.npy 矩阵的形状：{item_features.shape}")
#
# # 构建 item-item 共现图
# co_occur_dict = defaultdict(int)
# for item_list in user_sequences.values():
#     filtered = [item for item in item_list if item in valid_items]
#     for i, j in combinations(filtered, 2):
#         if i != j:
#             co_occur_dict[tuple(sorted([i, j]))] += 1
#
# # 打印共现边数量
# print(f"共现边的数量：{len(co_occur_dict)}")
#
# # 构建边结构和权重
# edge_index = []
# edge_weight = []
# for (item1, item2), weight in co_occur_dict.items():
#     edge_index.append([item1, item2])  # 使用原始物品ID
#     edge_weight.append(weight)
#
# # 转为 tensor 格式
# edge_index = torch.tensor(edge_index, dtype=torch.long).T
# edge_weight = torch.tensor(edge_weight, dtype=torch.float)
#
# # 保存边数据
# np.savez('movie_graph_edges.npz', edge_index=edge_index, edge_weight=edge_weight)
#
# # 打印示例
# print("共现边示例（前10条）：")
# for i in range(min(10, edge_index.shape[1])):
#     item1 = edge_index[0][i].item()
#     item2 = edge_index[1][i].item()
#     print(f"{item1} - {item2} (weight={edge_weight[i].item():.1f})")
#
# # 打印最终的数据统计
# print(f"\n共有 {len(user_sequences)} 个用户，{len(valid_items)} 个物品，{len(edge_weight)} 条共现边")
# print("预处理完成，数据已保存！")
#
# # 检查最终保存的 item_features.npy 和 movie_graph_edges.npz 中的最大物品ID以及是否有为0的物品ID
# # 读取 item_features.npy 文件
# item_features_data = np.load('item_features.npy')
# num_items_in_features = item_features_data.shape[0]
#
# # 读取 movie_graph_edges.npz 文件
# movie_graph_data = np.load('movie_graph_edges.npz')
# edge_index = movie_graph_data['edge_index']
#
# # 获取所有物品的ID
# unique_item_ids = set(edge_index.flatten().tolist())
#
# # 使用 valid_items 中的实际物品ID进行检查
# max_item_feature_id = max(valid_items) if valid_items else 0
# print(f"最大物品ID (item_features.npy): {max_item_feature_id}")
#
# # 检查 item_features.npy 中是否有ID为0的物品
# if 0 in valid_items:
#     print("item_features.npy 中存在物品ID为0的物品")
# else:
#     print("item_features.npy 中不存在物品ID为0的物品")
#
# # 检查 movie_graph_edges.npz 中的物品ID
# movie_graph_data = np.load('movie_graph_edges.npz')
# edge_index = movie_graph_data['edge_index']
# unique_item_ids = set(edge_index.flatten().tolist())
# max_movie_graph_id = max(unique_item_ids) if unique_item_ids else 0
# print(f"最大物品ID (movie_graph_edges.npz): {max_movie_graph_id}")
# print("movie_graph_edges.npz 中不存在物品ID为0的物品" if 0 not in unique_item_ids else "movie_graph_edges.npz 中存在物品ID为0的物品")




"任务2.通过作者提供的映射ID还有我们新的索引ID，映射出新的user_sequence_collaborative_indexing_500_20_sequential.txt，即包括映射用户ID和物品索引ID"
# 定义函数读取用户映射文件，并返回一个字典，键为原始用户ID，值为新的用户ID
def read_user_mapping(file_path):
    user_mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                original_user_id = int(parts[0])
                new_user_id = int(parts[1])
                user_mapping[original_user_id] = new_user_id
    return user_mapping

# 定义函数根据用户映射字典，将user_sequence.txt中的用户ID替换为新的用户ID，并保存为新文件
def remap_user_ids(user_sequence_file, user_mapping, output_file):
    with open(user_sequence_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split()
            if parts:
                original_user_id = int(parts[0])
                if original_user_id in user_mapping:
                    new_user_id = user_mapping[original_user_id]
                    new_line = f"{new_user_id} " + " ".join(parts[1:]) + "\n"
                    f_out.write(new_line)

# 文件路径
user_indexing_file = 'user_indexing.txt'  # 用户映射文件路径
user_sequence_file = 'user_sequence.txt'  # 原始用户交互序列文件路径
output_file = 'user_sequence_map.txt'     # 输出的新用户交互序列文件路径

# 读取用户映射文件
user_mapping = read_user_mapping(user_indexing_file)

# 根据用户映射，将原始用户交互序列中的用户ID替换为新的用户ID，并保存为新文件
remap_user_ids(user_sequence_file, user_mapping, output_file)

print(f"已将用户ID映射为新的ID，并保存到文件: {output_file}")

# 定义函数读取物品映射文件，并返回一个字典，键为原始物品ID，值为新的物品ID（无空格拼接）
import re

def clean_ci_token(token):
    # 1. 先把 <CI00> ➔ <CI0>
    token = re.sub(r'<CI00>', r'<CI0>', token)
    # 2. 再把 <CI0X> ➔ <CIX> （排除 <CI0>）
    token = re.sub(r'<CI0+([1-9]\d*)>', r'<CI\1>', token)
    return token

def read_item_mapping(file_path):
    item_mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                original_item_id = int(parts[0])
                new_item_id_raw = "".join(parts[1:])  # 拼接成 <CIxx><CIyy>
                # ✅ 清洗掉多余0
                new_item_id_cleaned = clean_ci_token(new_item_id_raw)
                item_mapping[original_item_id] = new_item_id_cleaned
    return item_mapping


# 定义函数根据物品映射字典，将user_sequence_new.txt中的物品ID替换为新的物品ID，并保存为新文件
def remap_item_ids(user_sequence_file, item_mapping, output_file):
    with open(user_sequence_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split()
            if parts:
                new_user_id = parts[0]
                new_item_ids = [item_mapping[int(item_id)] for item_id in parts[1:]]
                new_line = f"{new_user_id} {' '.join(new_item_ids)}\n"  # ✅ 这里 join 本身没问题，因为前面已经去掉了多余空格
                f_out.write(new_line)

# 文件路径
k = 30
n = 500
item_collaborative_indexing_file = f'item_collaborative_indexing_{n}_{k}_sequential.txt'  # 物品映射文件路径
user_sequence_new_file = 'user_sequence_map.txt'  # 用户交互序列文件路径
output_file_final = 'user_sequence_final.txt'  # 输出的最终用户交互序列文件路径

# 读取物品映射文件
item_mapping = read_item_mapping(item_collaborative_indexing_file)

# 根据物品映射，将用户交互序列中的物品ID替换为新的物品ID，并保存为最终文件
remap_item_ids(user_sequence_new_file, item_mapping, output_file_final)

print(f"已将物品ID映射为新的ID，并保存到最终文件: {output_file_final}")


# 定义函数将用户交互序列文件按用户ID排序
def sort_user_sequence(input_file, output_file):
    # 读取数据并按用户ID排序
    with open(input_file, 'r') as f_in:
        lines = f_in.readlines()

    # 排序按第一个元素
    sorted_lines = sorted(lines, key=lambda line: int(line.split()[0]))

    # 将排序后的数据写入输出文件
    with open(output_file, 'w') as f_out:
        for line in sorted_lines:
            f_out.write(line)


# 文件路径
input_file = 'user_sequence_final.txt'  # 输入的最终用户交互序列文件路径
output_file_sorted = f'user_sequence_collaborative_indexing_{n}_{k}_sequential.txt'  # 排序后的最终用户交互序列文件路径

# 对用户交互序列文件按用户ID排序
sort_user_sequence(input_file, output_file_sorted)

print(f"已将用户交互序列文件按用户ID排序，并保存到文件: {output_file_sorted}")


