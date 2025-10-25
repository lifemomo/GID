import json
import numpy as np
import torch
from itertools import combinations
from collections import defaultdict
import pandas as pd
# # 检查最终保存的 item_features.npy 和 movie_graph_edges.npz 中的最大物品ID以及是否有为0的物品ID
# # 读取 item_features.npy 文件
# item_features_data = np.load('item_features.npy')
# num_items_in_features = item_features_data.shape[0]
#
# # 打印最大物品ID
# print(f"最大物品ID (item_features.npy): {num_items_in_features - 1}")
#
# # 检查 item_features.npy 中是否有ID为0的物品
# if 0 in range(num_items_in_features):
#     print("item_features.npy 中存在物品ID为0的物品")
# else:
#     print("item_features.npy 中不存在物品ID为0的物品")
#
# # 读取 movie_graph_edges.npz 文件
# movie_graph_data = np.load('movie_graph_edges.npz')
# edge_index = movie_graph_data['edge_index']
#
# # 获取所有物品的ID
# unique_item_ids = set(edge_index.flatten().tolist())
#
# # 打印最大物品ID
# max_item_id = max(unique_item_ids)
# print(f"最大物品ID (movie_graph_edges.npz): {max_item_id}")
#
# # 检查 movie_graph_edges.npz 中是否有ID为0的物品
# if 0 in unique_item_ids:
#     print("movie_graph_edges.npz 中存在物品ID为0的物品")
# else:
#     print("movie_graph_edges.npz 中不存在物品ID为0的物品")
#
# # 读取并检查 item_features.npy 中是否存在物品ID为0
# item_features_data = np.load('item_features.npy')
#
# # 获取最大物品ID
# max_item_id = len(item_features_data) - 1
# print(f"最大物品ID (item_features.npy): {max_item_id}")
#
# # 检查 item_features.npy 中是否有物品ID为0
# if np.any(item_features_data[0] != 0):  # 检查第一行是否全是0
#     print("item_features.npy 中存在物品ID为0的物品")
# else:
#     print("item_features.npy 中不存在物品ID为0的物品")


# def check_continuity_from_user_sequence(file_path):
#     user_ids = set()
#     item_ids = set()
#
#     with open(file_path, 'r') as f:
#         for line in f:
#             tokens = line.strip().split()
#             if len(tokens) < 2:
#                 continue
#             user_id = int(tokens[0])
#             item_list = list(map(int, tokens[1:]))
#             user_ids.add(user_id)
#             item_ids.update(item_list)
#
#     # 检查连续性
#     user_ids_sorted = sorted(user_ids)
#     item_ids_sorted = sorted(item_ids)
#
#     user_continuous = user_ids_sorted == list(range(min(user_ids_sorted), max(user_ids_sorted) + 1))
#     item_continuous = item_ids_sorted == list(range(min(item_ids_sorted), max(item_ids_sorted) + 1))
#
#     print(f"用户ID是否连续: {user_continuous}")
#     print(f"用户ID范围: {min(user_ids_sorted)} ~ {max(user_ids_sorted)}，共 {len(user_ids_sorted)} 个用户")
#
#     print(f"物品ID是否连续: {item_continuous}")
#     print(f"物品ID范围: {min(item_ids_sorted)} ~ {max(item_ids_sorted)}，共 {len(item_ids_sorted)} 个物品")
#
# # 使用示例
# check_continuity_from_user_sequence('user_sequence.txt')


#统计数据集信息
import os, gzip, json, ast
from collections import defaultdict, Counter

# ======== 路径 ========
USER_SEQ_FILE = "user_sequence.txt"

# ======== 属性抽取配置（按需开关） ========
ATTR_CFG = {
    "categories_leaf": True,   # 取 categories 的叶子类目
    "categories_path": False,  # 或者取完整路径 'A>B>C'（与上互不排斥，可同时开启）
    "brand": True,
    "genres": True,            # 'genre' (str) 或 'genres' (list)
    # 你还可以增加：'format','studio','publisher' ... -> 在 extract_attrs() 里按需要扩展
}

# ========= 工具 =========
def parse_maybe_non_json(line: str):
    """优先 json.loads；失败再 ast.literal_eval（适配 2014 版的“松散 JSON”）。"""
    try:
        return json.loads(line)
    except Exception:
        try:
            return ast.literal_eval(line)
        except Exception:
            return None

def gini(values):
    vals = [v for v in values if v > 0]
    if not vals: return 0.0
    vals.sort()
    n = len(vals); s = sum(vals)
    cum = sum((i+1)*x for i,x in enumerate(vals))
    return (2*cum)/(n*s) - (n+1)/n

# ========= 读取交互，得到核心集合 =========
users, items, E_total = set(), set(), 0
user_items = defaultdict(list)
with open(USER_SEQ_FILE, "r", encoding="utf-8") as f:
    for line in f:
        p = line.strip().split()
        if len(p) < 2:
            continue
        u, its = p[0], p[1:]
        users.add(u)
        for it in its:
            items.add(it)
            user_items[u].append(it)
        E_total += len(its)

unique_edges = set()
for u, its in user_items.items():
    for it in its:
        unique_edges.add((u, it))

U, I = len(users), len(items)
E_unique = len(unique_edges)
density = (E_unique / (U*I)) if U and I else 0.0
sparsity_pct = (1 - density) * 100

print("=== Core stats ===")
print(f"#User = {U:,}")
print(f"#Item = {I:,}")
print(f"#Interactions (total, with duplicates) = {E_total:,}")
print(f"#Interactions (unique user-item pairs) = {E_unique:,}")
print(f"Sparsity (%) = {sparsity_pct:.4f}")
print(f"Avg interactions/user (total) = {(E_total/U if U else 0):.3f}")
print(f"Avg interactions/item (total) = {(E_total/I if I else 0):.3f}")
print(f"Avg interactions/user (unique) = {(E_unique/U if U else 0):.3f}")
print(f"Avg interactions/item (unique) = {(E_unique/I if I else 0):.3f}")
