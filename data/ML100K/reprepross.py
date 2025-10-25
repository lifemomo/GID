"任务1.预处理数据，方便后续的GCN处理及聚类"
import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
import torch

# -------------------- 加载路径 --------------------
u_data_path = 'u.data'
u_item_path = 'u.item'

# -------------------- 加载评分和电影数据 --------------------
ratings = pd.read_csv(u_data_path, sep='\t', header=None,
                      names=['user_id', 'item_id', 'rating', 'timestamp'])

item_cols = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL',
             'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
             'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
             'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv(u_item_path, sep='|', header=None, names=item_cols, encoding='latin-1')

# -------------------- K-core 过滤函数 --------------------
def filter_k_core(ratings, user_core, item_core):
    while True:
        before_len = len(ratings)
        user_counts = ratings.groupby('user_id').size()
        item_counts = ratings.groupby('item_id').size()

        valid_users = user_counts[user_counts >= user_core].index
        valid_items = item_counts[item_counts >= item_core].index

        ratings = ratings[ratings['user_id'].isin(valid_users) & ratings['item_id'].isin(valid_items)]

        after_len = len(ratings)
        if after_len == before_len:
            break
    return ratings

# -------------------- K-core 过滤 --------------------
ratings_filtered = filter_k_core(ratings, user_core=5, item_core=5)

# -------------------- 获取有效物品ID --------------------
valid_item_ids = sorted(ratings_filtered['item_id'].unique())
print(f"有效物品数量（原始 ID）：{len(valid_item_ids)}")

# -------------------- 构建 item_features（保持原始ID顺序） --------------------
valid_movies = movies[movies['movie_id'].isin(valid_item_ids)]
item_features = valid_movies.set_index('movie_id').loc[valid_item_ids].iloc[:, 5:24].values.astype(np.float32)

# -------------------- 构建共现图 --------------------
user_groups = ratings_filtered.groupby('user_id')['item_id'].apply(list)
co_occurrence_dict = defaultdict(int)

for items in user_groups:
    for i, j in combinations(set(items), 2):  # 使用 set 去重
        pair = tuple(sorted([i, j]))
        co_occurrence_dict[pair] += 1

edges = list(co_occurrence_dict.keys())
weights = list(co_occurrence_dict.values())

edge_index = torch.tensor(edges, dtype=torch.long).T  # [2, num_edges]
edge_weight = torch.tensor(weights, dtype=torch.float)

# -------------------- 打印示例 --------------------
print("示例共现边：")
print(pd.DataFrame({
    'movie_id_1': edge_index[0][:10].numpy(),
    'movie_id_2': edge_index[1][:10].numpy(),
    'weight': edge_weight[:10].numpy()
}))

# -------------------- 数据统计 --------------------
print(f"最终数据中包含 {ratings_filtered['user_id'].nunique()} 个用户，"
      f"{len(valid_item_ids)} 个物品，{len(ratings_filtered)} 条交互")

# -------------------- 打印有效物品ID（从小到大） --------------------
print("item_features.npy 中的物品ID列表（从小到大排序）：")
for movie_id in valid_item_ids:
    print(movie_id)

# # -------------------- 保存文件 --------------------
# np.save('item_features.npy', item_features)
# np.save('valid_item_ids.npy', valid_item_ids)
# np.savez('movie_graph_edges.npz', edge_index=edge_index, edge_weight=edge_weight)
#
# print("✅ 已保存 item_features.npy 和 movie_graph_edges.npz（保留原始 movie_id）")


"任务2.统计用户的交互序列 user_sequence.txt，并与作者处理的序列进行对比"
# import pandas as pd
# from collections import Counter
#
# # ========== 1. 定义 K-core 过滤函数 ==========
# def filter_k_core(ratings, user_core, item_core):
#     user_counts = ratings.groupby('user_id').size()
#     item_counts = ratings.groupby('item_id').size()
#
#     valid_users = user_counts[user_counts >= user_core].index
#     valid_items = item_counts[item_counts >= item_core].index
#
#     filtered_ratings = ratings[
#         ratings['user_id'].isin(valid_users) &
#         ratings['item_id'].isin(valid_items)
#     ]
#     return filtered_ratings
#
# # ========== 2. 加载数据 + K-core ==========
# input_file = 'u.data'
# ratings = pd.read_csv(input_file, sep='\t', header=None,
#                       names=['user_id', 'item_id', 'rating', 'timestamp'])
#
# user_core = 5
# item_core = 5
# ratings_filtered = filter_k_core(ratings, user_core, item_core)
#
# # ✅ 关键修改：按用户和时间排序
# ratings_sorted = ratings_filtered.sort_values(by=['user_id', 'timestamp'])
#
# # ========== 3. 构建用户交互序列 ==========
# user_interactions = {}
# for user_id, group in ratings_sorted.groupby('user_id'):
#     items = group['item_id'].tolist()
#     user_interactions[user_id] = items
#
# # ========== 4. 写入 user_sequence.txt ==========
# output_file = 'user_sequence_original.txt'
# with open(output_file, 'w') as f:
#     for user_id, items in user_interactions.items():
#         items_str = ' '.join(map(str, items))
#         f.write(f"{user_id} {items_str}\n")
#
# print(f"✅ 用户交互序列已保存到 {output_file}")
#
# # ========== 5. 定义序列读取与对比函数 ==========
# def read_user_sequence(file_path):
#     user_sequence = {}
#     with open(file_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             user_id = int(parts[0])
#             items = set(map(int, parts[1:]))
#             user_sequence[user_id] = items
#     return user_sequence
#
# def compare_user_sequences(seq1, seq2):
#     matched_users = set(seq1.keys()).intersection(set(seq2.keys()))
#     inconsistent_users = []
#     for user_id in matched_users:
#         if Counter(seq1[user_id]) != Counter(seq2[user_id]):
#             inconsistent_users.append(user_id)
#     return inconsistent_users
#
# # ========== 6. 对比两个 user_sequence ==========
# file1 = 'user_sequence_original.txt'
# file2 = 'user_sequence_old.txt'
#
# user_sequence1 = read_user_sequence(file1)
# user_sequence2 = read_user_sequence(file2)
#
# inconsistent_users = compare_user_sequences(user_sequence1, user_sequence2)
#
# if not inconsistent_users:
#     print("✅ 两个交互序列文件中的用户物品交互记录完全一致！")
# else:
#     print("⚠️ 以下用户的交互记录在两个文件中不一致：")
#     for user_id in inconsistent_users:
#         print(f"用户 {user_id}:")
#         print(f"您的交互序列: {list(user_sequence1[user_id])}")
#         print(f"作者的交互序列: {list(user_sequence2[user_id])}")



"任务3.通过作者提供的映射ID还有我们新的索引ID，映射出新的user_sequence_collaborative_indexing_500_20_sequential.txt，即包括映射用户ID和物品索引ID"
import os

# --------------------------- 读取用户映射 --------------------------- #
# def read_user_mapping(file_path):
#     user_mapping = {}
#     with open(file_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) == 2:
#                 original_user_id = int(parts[0])
#                 new_user_id = int(parts[1])
#                 user_mapping[original_user_id] = new_user_id
#     return user_mapping
#
# # --------------------------- 替换用户ID --------------------------- #
# def remap_user_ids(user_sequence_file, user_mapping, output_file):
#     with open(user_sequence_file, 'r') as f_in, open(output_file, 'w') as f_out:
#         for line in f_in:
#             parts = line.strip().split()
#             if parts:
#                 original_user_id = int(parts[0])
#                 if original_user_id in user_mapping:
#                     new_user_id = user_mapping[original_user_id]
#                     new_line = f"{new_user_id} " + " ".join(parts[1:]) + "\n"
#                     f_out.write(new_line)
#
# # --------------------------- 读取物品映射 --------------------------- #
# def read_item_mapping(file_path):
#     item_mapping = {}
#     with open(file_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) >= 2:
#                 original_item_id = parts[0]
#                 new_item_id = "".join(parts[1:])  # 拼接为一个整体，如 <CI0><CI1>
#                 item_mapping[original_item_id] = new_item_id
#     return item_mapping
#
# # --------------------------- 替换物品ID --------------------------- #
# def remap_item_ids(user_sequence_file, item_mapping, output_file):
#     with open(user_sequence_file, 'r') as f_in, open(output_file, 'w') as f_out:
#         for line in f_in:
#             parts = line.strip().split()
#             if parts:
#                 new_user_id = parts[0]
#                 try:
#                     new_item_ids = [item_mapping[item_id] for item_id in parts[1:]]
#                     new_line = f"{new_user_id} " + " ".join(new_item_ids) + "\n"
#                     f_out.write(new_line)
#                 except KeyError as e:
#                     print(f"跳过一行，原因：未找到物品ID {e}")
#
# # --------------------------- 按用户ID排序 --------------------------- #
# def sort_user_sequence(input_file, output_file):
#     with open(input_file, 'r') as f_in:
#         lines = f_in.readlines()
#
#     sorted_lines = sorted(lines, key=lambda line: int(line.split()[0]))
#
#     with open(output_file, 'w') as f_out:
#         for line in sorted_lines:
#             f_out.write(line)
#
# # --------------------------- 主执行流程 --------------------------- #
# k = 50
# n = 1000
#
# # 路径配置（静态文件）
# user_indexing_file = 'user_indexing.txt'                         # 用户映射表
# user_sequence_file = 'user_sequence_original.txt'                         # 原始用户序列
# user_sequence_mapped = 'user_sequence.txt'                   # 映射后（只替换用户）的中间文件
#
# # 路径配置（动态文件，根据 n 和 k）
# item_collaborative_indexing_file = f'item_collaborative_indexing_{n}_{k}_sequential.txt'  # 物品映射表
# user_sequence_final = f'user_sequence_final_{n}_{k}.txt'                                   # 替换物品ID后的最终序列（未排序）
# user_sequence_sorted = f'user_sequence_collaborative_indexing_{n}_{k}_sequential.txt'      # 最终文件
#
# # 1. 用户ID映射
# user_mapping = read_user_mapping(user_indexing_file)
# remap_user_ids(user_sequence_file, user_mapping, user_sequence_mapped)
# print(f"✅ 用户ID已映射并保存至: {user_sequence_mapped}")
#
# # 2. 物品ID映射
# item_mapping = read_item_mapping(item_collaborative_indexing_file)
# remap_item_ids(user_sequence_mapped, item_mapping, user_sequence_final)
# print(f"✅ 物品ID已映射并保存至: {user_sequence_final}")
#
# # 3. 按用户ID排序
# sort_user_sequence(user_sequence_final, user_sequence_sorted)
# print(f"✅ 最终文件已按用户ID排序并保存至: {user_sequence_sorted}")
#
