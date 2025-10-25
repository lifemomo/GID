# 读取 item_collaborative_indexing 文件中的物品ID
def read_item_ids_from_file(file_path):
    item_ids = set()  # 使用集合来存储物品ID，以便后续对比
    with open(file_path, 'r') as file:
        for line in file:
            tokens = line.strip().split()  # 通过空格分隔每一行
            if len(tokens) > 1:
                item_id = int(tokens[0])  # 提取物品ID（文件中的第一个字段）
                item_ids.add(item_id)
    return item_ids

# 检查两个文件中的物品ID是否匹配
def check_items_match(file1, file2):
    # 读取两个文件中的物品ID
    item_ids_file1 = read_item_ids_from_file(file1)
    item_ids_file2 = read_item_ids_from_file(file2)

    # 输出差异
    if item_ids_file1 == item_ids_file2:
        print("两个文件中的物品ID完全匹配！")
    else:
        print("物品ID不匹配！")
        # 找出不同的物品ID
        missing_in_file2 = item_ids_file1 - item_ids_file2
        missing_in_file1 = item_ids_file2 - item_ids_file1
        if missing_in_file2:
            print(f"在{file2}中缺少的物品ID: {missing_in_file2}")
        if missing_in_file1:
            print(f"在{file1}中缺少的物品ID: {missing_in_file1}")


# 调用函数检查两个文件
file1 = 'item_collaborative_indexing_500_20_sequential_old.txt'
file2 = 'item_collaborative_indexing_200_50_sequential.txt'
check_items_match(file1, file2)

# import pandas as pd
#
# # 读取 u.data 文件
# data_path = 'u.data'
# df = pd.read_csv(data_path, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
#
# # 检查用户ID是否连续
# user_ids = sorted(df['user_id'].unique())
# user_continuous = (user_ids == list(range(min(user_ids), max(user_ids) + 1)))
#
# # 检查电影ID是否连续
# item_ids = sorted(df['item_id'].unique())
# item_continuous = (item_ids == list(range(min(item_ids), max(item_ids) + 1)))
#
# # 输出结果
# print(f"用户ID是否连续: {user_continuous}")
# print(f"用户ID范围: {min(user_ids)} ~ {max(user_ids)}，共 {len(user_ids)} 个用户")
#
# print(f"电影ID是否连续: {item_continuous}")
# print(f"电影ID范围: {min(item_ids)} ~ {max(item_ids)}，共 {len(item_ids)} 个物品")

# 两个文件中的物品ID完全匹配！
# 用户ID是否连续: True
# 用户ID范围: 1 ~ 943，共 943 个用户
# 电影ID是否连续: True
# 电影ID范围: 1 ~ 1682，共 1682 个物品


# import numpy as np
#
# # 读取 user_sequence.txt 中所有出现的 item_id
# def extract_items_from_user_sequence(file_path):
#     item_ids = set()
#     with open(file_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             # 跳过用户 ID，第一个是 user
#             for item_id in parts[1:]:
#                 try:
#                     item_ids.add(int(item_id))
#                 except ValueError:
#                     print(f"⚠️ 无法转换为整数的物品ID: {item_id}")
#     return item_ids
#
# # 主程序入口
# user_sequence_file = 'user_sequence.txt'
# item_feature_path = 'item_features.npy'
#
# # 提取 user_sequence 中用到的 item_id
# sequence_item_ids = extract_items_from_user_sequence(user_sequence_file)
# print(f"✅ 从 user_sequence.txt 中提取到 {len(sequence_item_ids)} 个唯一 item_id")
#
# # 加载 item_features 并获取其索引范围
# item_features = np.load(item_feature_path)
# item_feature_count = item_features.shape[0]
# item_index_range = set(range(1, item_feature_count + 1))  # 假设 index 从 1 开始
#
# # 检查是否所有 item_id 都存在于特征中
# missing_items = sequence_item_ids - item_index_range
# if missing_items:
#     print(f"❌ 有 {len(missing_items)} 个物品ID 不在 item_features 中，例如：{list(missing_items)[:10]}")
# else:
#     print("🎉 所有 user_sequence.txt 中的 item_id 都存在于 item_features 中！")
#
# print("user_sequence.txt 中的 item_id 范围:", min(sequence_item_ids), "到", max(sequence_item_ids))
# print("是否连续？", len(sequence_item_ids) == (max(sequence_item_ids) - min(sequence_item_ids) + 1))
#
# print("item_features.npy 中的索引范围:", 1, "到", item_feature_count)
# print("是否连续？", item_feature_count == (max(range(1, item_feature_count + 1)) - min(range(1, item_feature_count + 1)) + 1))


# def extract_item_ids_from_user_sequence(file_path):
#     """从用户序列文件中提取所有交互的物品索引ID（<CIx> 格式）"""
#     item_ids = set()
#     with open(file_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()[1:]  # Skip user_id and get the item indexes
#             item_ids.update(parts)  # Add item indexes (e.g., <CI0><CI1>)
#     return item_ids
#
# def extract_item_ids_from_item_index(file_path):
#     """从物品索引文件中提取所有物品的索引ID（<CIx> 格式）"""
#     item_ids = set()
#     with open(file_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()[1:]  # Skip actual item ID and get the item indexes
#             item_ids.update(parts)  # Add item indexes (e.g., <CI0><CI1>)
#     return item_ids
#
#
# k = 20
# n = 300
# # 文件路径
# user_seq_file = f"user_sequence_collaborative_indexing_{n}_{k}_sequential.txt"
# item_index_file = f"item_collaborative_indexing_{n}_{k}_sequential.txt"
#
# # 提取索引ID集合
# user_seq_item_ids = extract_item_ids_from_user_sequence(user_seq_file)
# item_index_ids = extract_item_ids_from_item_index(item_index_file)
#
# # 比较两个集合
# missing_in_index = user_seq_item_ids - item_index_ids
# extra_in_index = item_index_ids - user_seq_item_ids
#
# # 打印结果
# print(f"用户序列中总共有 {len(user_seq_item_ids)} 个不同的物品索引ID")
# print(f"item_index 文件中总共有 {len(item_index_ids)} 个不同的物品索引ID")
#
# if not missing_in_index and not extra_in_index:
#     print("✅ 两个文件中的物品索引ID完全一致！")
# else:
#     if missing_in_index:
#         print(f"❌ 有 {len(missing_in_index)} 个物品索引ID 在用户序列中出现，但未在 item_index 中找到，例如：{list(missing_in_index)[:10]}")
#     if extra_in_index:
#         print(f"⚠️ 有 {len(extra_in_index)} 个物品索引ID 在 item_index 中存在，但未出现在用户序列中，例如：{list(extra_in_index)[:10]}")


