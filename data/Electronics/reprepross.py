"任务1 处理初始数据，确保与作者的用户商品ID匹配"
import gzip
import ast
import json
# 1. 加载 user_sequence.txt 中的商品 ASIN
# sequence_file = "user_sequence.txt"
# target_asins = set()
#
# with open(sequence_file, 'r', encoding='utf-8') as f:
#     for line in f:
#         parts = line.strip().split()
#         if len(parts) > 1:
#             target_asins.update(parts[1:])
#
# print(f"✅ 目标商品 ASIN 数量: {len(target_asins)}")
#
# # 2. 筛选并保存 meta_Electronics.json.gz 中匹配商品
# meta_input = "meta_Electronics.json.gz"
# meta_output = "filtered_meta_Electronics.json.gz"
#
# with gzip.open(meta_input, 'rt', encoding='utf-8', errors='ignore') as fin, \
#         gzip.open(meta_output, 'wt', encoding='utf-8') as fout:
#     kept = 0
#     for line in fin:
#         try:
#             record = ast.literal_eval(line)
#             if record.get("asin") in target_asins:
#                 fout.write(json.dumps(record) + "\n")
#                 kept += 1
#         except Exception:
#             continue
#
# print(f"✅ 已保存 {kept} 条商品记录至 {meta_output}")
#
# # 1. 提取 user_sequence.txt 中的用户和商品
# target_users = set()
# target_asins = set()
#
# with open(sequence_file, 'r', encoding='utf-8') as f:
#     for line in f:
#         parts = line.strip().split()
#         if parts:
#             target_users.add(parts[0])
#             target_asins.update(parts[1:])
#
# print(f"✅ 目标用户数: {len(target_users)}，商品数: {len(target_asins)}")
#
# # 2. 筛选 reviews_Beauty_5.json.gz
# review_input = "reviews_Electronics_5.json.gz"
# review_output = "filtered_reviews_Electronics_5.json.gz"
#
# with gzip.open(review_input, 'rt', encoding='utf-8', errors='ignore') as fin, \
#      gzip.open(review_output, 'wt', encoding='utf-8') as fout:
#
#     kept = 0
#     for line in fin:
#         try:
#             review = json.loads(line)
#             if review.get("reviewerID") in target_users and review.get("asin") in target_asins:
#                 fout.write(json.dumps(review) + "\n")
#                 kept += 1
#         except Exception:
#             continue
#
# print(f"✅ 已保存 {kept} 条评论至 {review_output}")

"任务1.5 剔除噪声数据，缩小数据集的数量"
from collections import defaultdict
# def load_user_sequence(file_path):
#     user_item_dict = {}
#     with open(file_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) >= 2:
#                 user_id = parts[0]
#                 item_ids = parts[1:]
#                 user_item_dict[user_id] = item_ids
#     return user_item_dict
#
# def summarize(user_item_dict, stage="前"):
#     all_items = set()
#     total_interactions = 0
#     for items in user_item_dict.values():
#         all_items.update(items)
#         total_interactions += len(items)
#     print(f"📊 过滤{stage}统计：")
#     print(f"用户数: {len(user_item_dict)}")
#     print(f"项目数: {len(all_items)}")
#     print(f"总交互数: {total_interactions}\n")
# def k_core_filter(user_item_dict, k_user=5, k_item=5):
#     user_item = defaultdict(set)
#     item_user = defaultdict(set)
#
#     # 构建反向索引
#     for user, items in user_item_dict.items():
#         for item in items:
#             user_item[user].add(item)
#             item_user[item].add(user)
#
#     changed = True
#     while changed:
#         changed = False
#
#         # 过滤用户
#         users_to_remove = [u for u, items in user_item.items() if len(items) < k_user]
#         for u in users_to_remove:
#             for i in user_item[u]:
#                 item_user[i].discard(u)
#             del user_item[u]
#             changed = True
#
#         # 过滤项目
#         items_to_remove = [i for i, users in item_user.items() if len(users) < k_item]
#         for i in items_to_remove:
#             for u in item_user[i]:
#                 user_item[u].discard(i)
#             del item_user[i]
#             changed = True
#
#     # 构造最终过滤后的 user-item 字典
#     filtered_user_item_dict = {
#         u: list(items) for u, items in user_item.items() if len(items) > 0
#     }
#
#     return filtered_user_item_dict
#
# def save_user_sequence(user_item_dict, output_path):
#     with open(output_path, 'w') as f:
#         for user, items in user_item_dict.items():
#             line = user + ' ' + ' '.join(items) + '\n'
#             f.write(line)
#
# # ========== 参数设定 ==========
# input_file = 'user_sequence_all.txt'
# output_file = 'user_sequence.txt'
# k_user = 12
# k_item = 10
#
# # ========== 执行流程 ==========
# user_item_dict =load_user_sequence(input_file)
# summarize(user_item_dict, stage="前")
#
# filtered_user_item_dict = k_core_filter(user_item_dict, k_user=k_user, k_item=k_item)
# summarize(filtered_user_item_dict, stage="后")
#
# save_user_sequence(filtered_user_item_dict, output_file)
# print(f"✅ 已保存过滤后的交互序列至：{output_file}")

"任务1.6 调整过滤后的user_sequence.txt中的用户项目交互顺序"
from collections import defaultdict
# USER_SEQ_FILE = "user_sequence.txt"
# USER_INDEX_FILE = "user_indexing.txt"
# ITEM_COLLAB_FILE_OLD = "item_collaborative_indexing_500_20_sequential-old.txt"
# USER_SEQ_COLLAB_FILE_OLD = "user_sequence_collaborative_indexing_500_20_sequential-old.txt"
# OUTPUT_FILE = "user_sequence_reordered.txt"
#
# def read_user_sequence(path):
#     """返回：list of (orig_user, [asin...])"""
#     seqs = []
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split()
#             if not parts:
#                 continue
#             seqs.append((parts[0], parts[1:]))
#     return seqs
#
# def read_user_index_map(path):
#     """原始用户ID -> 新用户ID"""
#     m = {}
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) == 2:
#                 m[parts[0]] = parts[1]
#     return m
#
# def read_item_collab_map(path):
#     """ASIN -> token（例如 <CI1><CI29><CI91>）"""
#     m = {}
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) >= 2:
#                 asin = parts[0]
#                 token = "".join(parts[1:])
#                 m[asin] = token
#     return m
#
# def invert_item_collab_map(asin2tok):
#     """token -> [ASIN...]（考虑到理论上可能一对多，这里存list）"""
#     tok2asins = defaultdict(list)
#     for a, t in asin2tok.items():
#         tok2asins[t].append(a)
#     return tok2asins
#
# def read_user_seq_collab_old(path):
#     """新用户ID + token…  -> dict: new_user -> [token…]"""
#     d = {}
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split()
#             if not parts:
#                 continue
#             new_uid = parts[0]
#             tokens = parts[1:]
#             d[new_uid] = tokens
#     return d
#
# from collections import Counter
#
# def reorder_user_items_by_tokens(user_items, token_order, tok2asins):
#     """
#     保留重复：同一 ASIN 出现 n 次，就在结果中出现 n 次。
#     策略：
#       1) 第一阶段：按 token_order 逐个 token，在候选 ASIN 中挑“还剩余次数>0”的那个，放入结果并将剩余次数-1；
#          若一个 token 对应多个 ASIN，优先选在该用户原序列里“更靠前出现的那个”（可选优化见注释）。
#       2) 第二阶段：把剩余未匹配的 ASIN，按原序列相对顺序、逐个消耗剩余次数补齐。
#     """
#     # 统计每个 ASIN 的剩余次数
#     remaining = Counter(user_items)
#     result = []
#
#     # 可选优化：预先记录每个 ASIN 在用户序列中的“首次出现位置”，用于 tie-break
#     first_pos = {}
#     for idx, a in enumerate(user_items):
#         if a not in first_pos:
#             first_pos[a] = idx
#
#     # 阶段1：按 token 顺序放置
#     for tok in token_order:
#         cands = tok2asins.get(tok, [])
#         # 在候选中，选“还剩余次数>0”的，若有多个，选 first_pos 最小的那个
#         best = None
#         best_pos = 10**18
#         for a in cands:
#             if remaining[a] > 0:
#                 pos = first_pos.get(a, 10**18)
#                 if pos < best_pos:
#                     best = a
#                     best_pos = pos
#         if best is not None:
#             result.append(best)
#             remaining[best] -= 1
#
#     # 阶段2：把剩余的按原相对顺序补齐（保留重复）
#     for a in user_items:
#         while remaining[a] > 0:
#             result.append(a)
#             remaining[a] -= 1
#
#     # 强校验（长度应与输入一致）
#     assert len(result) == len(user_items), "重排后长度与原序列不一致，检查逻辑！"
#     return result
#     "建立该用户 ASIN -> 出现次数（以便处理重复）,这里假设 user_items 里不常有重复；若有需要，可改成 multiset 逻辑,暂用 set + used 控制一次性匹配"
#     for tok in token_order:
#         candidates = tok2asins.get(tok, [])
#         picked = None
#         for asin in candidates:
#             if asin in remaining and asin not in used:
#                 picked = asin
#                 break
#         if picked is not None:
#             result.append(picked)
#             used.add(picked)
#             # 不从 remaining 移除是为了后续保持原相对顺序时仍能判断
#             # 但我们用 used 控制不要重复加入
#
#     # 将未匹配到 token 的ASIN，按原相对顺序追加（不丢失）
#     for asin in user_items:
#         if asin not in used:
#             result.append(asin)
#             used.add(asin)
#
#     # 保险：长度应与原始相同
#     # 如果你希望强校验，可加 assert len(result) == len(user_items)
#     return result
#
# def main():
#     # 读取数据
#     raw_user_seqs = read_user_sequence(USER_SEQ_FILE)                 # [(orig_user, [asin...])]
#     user_id_map = read_user_index_map(USER_INDEX_FILE)                # orig_user -> new_user
#     asin2tok_old = read_item_collab_map(ITEM_COLLAB_FILE_OLD)         # asin -> token
#     tok2asins_old = invert_item_collab_map(asin2tok_old)              # token -> [asin...]
#     old_user_tok_seq = read_user_seq_collab_old(USER_SEQ_COLLAB_FILE_OLD)  # new_user -> [token...]
#
#     not_found_user = 0
#     reordered_lines = 0
#
#     with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
#         for orig_user, asins in raw_user_seqs:
#             new_user = user_id_map.get(orig_user)
#             if new_user is None:
#                 # 找不到用户映射：保留原顺序写出，也可选择跳过
#                 fout.write(orig_user + " " + " ".join(asins) + "\n")
#                 not_found_user += 1
#                 continue
#
#             token_order = old_user_tok_seq.get(new_user)
#             if token_order is None:
#                 # old 文件里没有这个用户：同样保留原顺序
#                 fout.write(orig_user + " " + " ".join(asins) + "\n")
#                 continue
#
#             reordered = reorder_user_items_by_tokens(asins, token_order, tok2asins_old)
#             fout.write(orig_user + " " + " ".join(reordered) + "\n")
#             reordered_lines += 1
#
#     print(f"✅ 完成重排：{reordered_lines} 行已按 old token 顺序重排。")
#     if not_found_user:
#         print(f"⚠️ {not_found_user} 个用户在 user_indexing.txt 中未找到映射，已按原顺序保留。")
#     print(f"已输出到：{OUTPUT_FILE}")
#
# if __name__ == "__main__":
#     main()


"任务2.预处理数据，方便后续的GCN处理及聚类"
import json
import numpy as np
import torch
from itertools import combinations
from collections import defaultdict
import gzip

# 文件路径
# user_sequence_path = 'user_sequence.txt'
# meta_file_path = 'filtered_meta_Electronics.json.gz'
#
# # ----------------------- Step 1: 读取用户序列 -----------------------
# user_sequences = {}
# all_asins_in_seq = set()
#
# with open(user_sequence_path, 'r') as f:
#     for line in f:
#         tokens = line.strip().split()
#         if len(tokens) < 2:
#             continue
#         user_id = tokens[0]
#         asins = tokens[1:]
#         user_sequences[user_id] = asins
#         all_asins_in_seq.update(asins)
#
# # ----------------------- Step 2: 读取商品元数据 -----------------------
# asin2meta = {}
# brands = set()
# categories = set()
#
# with gzip.open(meta_file_path, 'rt', encoding='utf-8') as f:
#     for line in f:
#         meta = json.loads(line)
#         asin = meta.get('asin')
#         if asin not in all_asins_in_seq:
#             continue
#         asin2meta[asin] = meta
#         brand = meta.get('brand')
#         if brand:
#             brands.add(brand.strip())
#         cat_list = meta.get('categories')
#         if isinstance(cat_list, list) and len(cat_list) > 0:
#             for cat in cat_list[0]:  # 只取第一级列表
#                 categories.add(cat.strip())
#
# valid_asins = sorted(asin2meta.keys())
#
# print(f"✅ 有效商品总数: {len(valid_asins)}")
#
# # ----------------------- Step 3: 构建嵌入索引字典 -----------------------
# brand_to_idx = {b: i for i, b in enumerate(sorted(brands))}
# category_to_idx = {c: i for i, c in enumerate(sorted(categories))}
#
# # 嵌入层设置
# embedding_dim = 64  # 可自定义,若模型欠拟合（聚类不清晰），可提升维度,若过拟合或效率低，可降为 32
# brand_emb = torch.nn.Embedding(len(brand_to_idx), embedding_dim)
# cat_emb = torch.nn.Embedding(len(category_to_idx), embedding_dim)
#
# # ----------------------- Step 4: 构建商品特征矩阵 -----------------------
# asin_to_index = {asin: idx for idx, asin in enumerate(valid_asins)}
# index_to_asin = {idx: asin for asin, idx in asin_to_index.items()}
#
# item_features = []
#
# for asin in valid_asins:
#     meta = asin2meta[asin]
#
#     # 品牌向量
#     brand_vec = torch.zeros(embedding_dim)
#     brand = meta.get('brand')
#     if brand and brand.strip() in brand_to_idx:
#         brand_vec = brand_emb(torch.tensor(brand_to_idx[brand.strip()]))
#
#     # 类别平均向量
#     cat_vecs = []
#     cats = meta.get('categories')
#     if isinstance(cats, list) and len(cats) > 0:
#         for c in cats[0]:
#             c = c.strip()
#             if c in category_to_idx:
#                 cat_vecs.append(cat_emb(torch.tensor(category_to_idx[c])))
#     if cat_vecs:
#         cat_vec = torch.stack(cat_vecs).mean(dim=0)
#     else:
#         cat_vec = torch.zeros(embedding_dim)
#
#     # 合并向量
#     final_vec = torch.cat([brand_vec, cat_vec])
#     item_features.append(final_vec.detach().numpy())
#
# item_features = np.array(item_features, dtype=np.float32)
# np.save('item_features.npy', item_features)
# print(f"📦 item_features.npy 保存完成，形状: {item_features.shape}")
#
# # ----------------------- Step 5: 构建商品共现图 -----------------------
# co_occur_dict = defaultdict(int)
# for asins in user_sequences.values():
#     filtered = [a for a in asins if a in asin_to_index]
#     for i, j in combinations(filtered, 2):
#         if i != j:
#             pair = tuple(sorted([i, j]))
#             co_occur_dict[pair] += 1
#
# edge_index = []
# edge_weight = []
#
# for (a1, a2), w in co_occur_dict.items():
#     edge_index.append([asin_to_index[a1], asin_to_index[a2]])
#     edge_weight.append(w)
#
# edge_index = torch.tensor(edge_index, dtype=torch.long).T
# edge_weight = torch.tensor(edge_weight, dtype=torch.float)
#
# np.savez('movie_graph_edges.npz', edge_index=edge_index, edge_weight=edge_weight)
# print(f"🧩 共现图保存完成，共有边数: {edge_index.shape[1]}")
#
# # ----------------------- Step 6: 保存索引映射字典 -----------------------
# with open('asin_to_index.json', 'w', encoding='utf-8') as f:
#     json.dump(asin_to_index, f, ensure_ascii=False, indent=2)
# with open('index_to_asin.json', 'w', encoding='utf-8') as f:
#     json.dump({str(k): v for k, v in index_to_asin.items()}, f, ensure_ascii=False, indent=2)
#
# print("✅ 所有处理完成！可进行GCN聚类分析。")


"任务3.通过作者提供的映射ID还有我们新的索引ID，映射出新的user_sequence_collaborative_indexing_500_20_sequential.txt，即包括映射用户ID和物品索引ID"
# 定义函数读取用户映射文件，并返回一个字典，键为原始用户ID，值为新的用户ID
def read_user_mapping(file_path):
    user_mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                original_user_id = parts[0]
                new_user_id = parts[1]
                user_mapping[original_user_id] = new_user_id
    return user_mapping

# --------------------------- 替换用户ID --------------------------- #
def remap_user_ids(user_sequence_file, user_mapping, output_file):
    with open(user_sequence_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split()
            if parts:
                original_user_id = parts[0]
                if original_user_id in user_mapping:
                    new_user_id = user_mapping[original_user_id]
                    new_line = f"{new_user_id} " + " ".join(parts[1:]) + "\n"
                    f_out.write(new_line)
                else:
                    print(f"⚠️ 跳过一行：用户 {original_user_id} 未找到映射")

# --------------------------- 读取物品映射 --------------------------- #
def read_item_mapping(file_path):
    item_mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                original_item_id = parts[0]
                new_item_id = "".join(parts[1:])  # 拼接完整路径 <CIx><CIy>
                item_mapping[original_item_id] = new_item_id
    return item_mapping

# --------------------------- 替换物品ID --------------------------- #
def remap_item_ids(user_sequence_file, item_mapping, output_file):
    with open(user_sequence_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split()
            if parts:
                user_id = parts[0]
                try:
                    new_item_ids = [item_mapping[item_id] for item_id in parts[1:]]
                    new_line = f"{user_id} " + " ".join(new_item_ids) + "\n"
                    f_out.write(new_line)
                except KeyError as e:
                    print(f"⚠️ 跳过一行：物品ID未映射 {e}")

# --------------------------- 按用户ID排序 --------------------------- #
def sort_user_sequence(input_file, output_file):
    with open(input_file, 'r') as f_in:
        lines = f_in.readlines()

    # 转换为整数ID排序
    sorted_lines = sorted(lines, key=lambda line: int(line.split()[0]))

    with open(output_file, 'w') as f_out:
        for line in sorted_lines:
            f_out.write(line)

# --------------------------- 主执行流程 --------------------------- #
k = 2
n = 500

# 文件路径配置
user_sequence_file = 'user_sequence.txt'                         # 原始字符串用户序列
user_indexing_file = 'user_indexing.txt'                         # 用户映射文件：字符串ID ➝ 整数ID
user_sequence_mapped = 'user_sequence_mapped.txt'                # 替换用户ID后的中间文件

item_collaborative_indexing_file = f'item_collaborative_indexing_{n}_{k}_sequential.txt'  # 物品映射路径
user_sequence_final = f'user_sequence_final_{n}_{k}.txt'         # 替换物品ID后的文件
user_sequence_sorted = f'user_sequence_collaborative_indexing_{n}_{k}_sequential.txt'  # 最终输出文件

# Step 1：替换用户ID
user_mapping = read_user_mapping(user_indexing_file)
remap_user_ids(user_sequence_file, user_mapping, user_sequence_mapped)
print(f"✅ 用户ID已映射并保存至: {user_sequence_mapped}")

# Step 2：替换物品ID
item_mapping = read_item_mapping(item_collaborative_indexing_file)
remap_item_ids(user_sequence_mapped, item_mapping, user_sequence_final)
print(f"✅ 物品ID已映射并保存至: {user_sequence_final}")

# Step 3：排序输出
sort_user_sequence(user_sequence_final, user_sequence_sorted)
print(f"✅ 最终文件已排序并保存至: {user_sequence_sorted}")


"任务4. 根据过滤后的数据集清除掉作者的协同索引数据中不存在的u用户和项目"
import re

# 文件路径定义
# user_sequence_txt = 'user_sequence.txt'
# user_indexing_txt = 'user_indexing.txt'
# user_seq_old_txt = 'user_sequence_collaborative_indexing_500_20_sequential-old-all.txt'
# item_index_old_txt = 'item_collaborative_indexing_500_20_sequential-old-all.txt'
#
# user_seq_clean_txt = 'user_sequence_collaborative_indexing_500_20_sequential-old.txt'
# item_index_clean_txt = 'item_collaborative_indexing_500_20_sequential-old.txt'
#
# # Step 1: 加载原始用户和项目集合
# def load_user_item_set(user_sequence_file):
#     user_set = set()
#     item_set = set()
#     with open(user_sequence_file, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if parts:
#                 user_set.add(parts[0])
#                 item_set.update(parts[1:])
#     return user_set, item_set
#
# # Step 2: 读取用户ID映射
# def load_user_indexing(file_path):
#     new_to_raw = {}
#     with open(file_path, 'r') as f:
#         for line in f:
#             raw_id, new_id = line.strip().split()
#             new_to_raw[new_id] = raw_id
#     return new_to_raw
#
# # Step 3: 读取 item token → 原始 item 的映射（反向）
# def load_token_to_item(file_path):
#     token_to_item = {}
#     with open(file_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) == 2:
#                 raw_item, token = parts
#                 token_to_item[token] = raw_item
#     return token_to_item
#
# # Step 4: 过滤用户交互文件
# def filter_user_sequence(user_seq_file, new_to_raw_user, valid_users, valid_items, token_to_item, output_file):
#     retained_lines = 0
#     with open(user_seq_file, 'r') as f_in, open(output_file, 'w') as f_out:
#         for line in f_in:
#             parts = line.strip().split()
#             if not parts:
#                 continue
#
#             new_user_id = parts[0]
#             raw_user_id = new_to_raw_user.get(new_user_id)
#             if raw_user_id not in valid_users:
#                 continue
#
#             # 提取原始项目ID，过滤不在 valid_items 中的项目
#             filtered_tokens = []
#             for token in parts[1:]:
#                 if token in token_to_item and token_to_item[token] in valid_items:
#                     filtered_tokens.append(token)
#
#             if filtered_tokens:
#                 f_out.write(f"{new_user_id} {' '.join(filtered_tokens)}\n")
#                 retained_lines += 1
#     print(f"✅ 过滤后用户交互序列保留 {retained_lines} 条记录")
#
# # Step 5: 过滤项目映射文件
# def filter_item_indexing(item_index_file, valid_items, output_file):
#     retained = 0
#     with open(item_index_file, 'r') as f_in, open(output_file, 'w') as f_out:
#         for line in f_in:
#             parts = line.strip().split()
#             if parts and parts[0] in valid_items:
#                 f_out.write(line)
#                 retained += 1
#     print(f"✅ 过滤后保留项目 {retained} 个")
#
# # 执行流程
# valid_users, valid_items = load_user_item_set(user_sequence_txt)
# new_to_raw_user = load_user_indexing(user_indexing_txt)
# token_to_item = load_token_to_item(item_index_old_txt)
#
# filter_user_sequence(user_seq_old_txt, new_to_raw_user, valid_users, valid_items, token_to_item, user_seq_clean_txt)
# filter_item_indexing(item_index_old_txt, valid_items, item_index_clean_txt)



"任务5. 根据过滤后的数据集清除掉作者的数据中随机索引和顺序索引不存在的u用户和项目"
# 文件路径
# user_sequence_txt = 'user_sequence.txt'
# user_indexing_txt = 'user_indexing.txt'
# # 随机索引
# # user_seq_old_txt = 'user_sequence_random_indexing_old.txt'
# # item_index_old_txt = 'item_random_indexing_old.txt'
# # user_seq_clean_txt = 'user_sequence_random_indexing.txt'
# # item_index_clean_txt = 'item_random_indexing.txt'
#
# # 顺序索引
# user_seq_old_txt = 'user_sequence_sequential_indexing_original_old.txt'
# item_index_old_txt = 'item_sequential_indexing_original_old.txt'
# user_seq_clean_txt = 'user_sequence_sequential_indexing_original.txt'
# item_index_clean_txt = 'item_sequential_indexing_original.txt'
#
#
# # Step 1: 读取原始用户序列，获取存在的用户和项目集合
# valid_users = set()
# valid_items = set()
# with open(user_sequence_txt, 'r') as f:
#     for line in f:
#         parts = line.strip().split()
#         if not parts:
#             continue
#         user_id = parts[0]
#         item_ids = parts[1:]
#         valid_users.add(user_id)
#         valid_items.update(item_ids)
#
# # Step 2: 读取用户映射表：原始用户 → 新用户ID
# user_id_map = {}
# with open(user_indexing_txt, 'r') as f:
#     for line in f:
#         orig, new = line.strip().split()
#         user_id_map[orig] = new
#
# # Step 3: 构建反向映射：新用户ID → 原始用户ID
# new_to_orig_user = {v: k for k, v in user_id_map.items()}
#
# # Step 4: 过滤 user_sequence_random_indexing_old.txt
# with open(user_seq_old_txt, 'r') as f_in, open(user_seq_clean_txt, 'w') as f_out:
#     for line in f_in:
#         parts = line.strip().split()
#         if not parts:
#             continue
#         new_user_id = parts[0]
#         orig_user_id = new_to_orig_user.get(new_user_id)
#         if orig_user_id in valid_users:
#             f_out.write(line)
#
# # Step 5: 过滤 item_random_indexing_old.txt
# with open(item_index_old_txt, 'r') as f_in, open(item_index_clean_txt, 'w') as f_out:
#     for line in f_in:
#         parts = line.strip().split(maxsplit=1)
#         if not parts:
#             continue
#         item_id = parts[0]
#         if item_id in valid_items:
#             f_out.write(line)


"任务5.文件路径 过滤掉index中不需要的用户项目"
# user_sequence_txt = 'user_sequence.txt'
# user_indexing_txt = 'user_indexing.txt'
# user_seq_old_txt = 'user_sequence_random_indexing_old.txt'
# item_index_old_txt = 'item_random_indexing_old.txt'
#
# user_seq_clean_txt = 'user_sequence_random_indexing.txt'
# item_index_clean_txt = 'item_random_indexing.txt'
# user_indexing_new_txt = 'user_indexing_new.txt'
#
# # Step 1: 读取原始用户序列，获取存在的用户和项目集合
# valid_users = set()
# valid_items = set()
# with open(user_sequence_txt, 'r') as f:
#     for line in f:
#         parts = line.strip().split()
#         if not parts:
#             continue
#         user_id = parts[0]
#         item_ids = parts[1:]
#         valid_users.add(user_id)
#         valid_items.update(item_ids)
#
# # Step 2: 读取用户映射表：原始用户 → 新用户ID
# user_id_map = {}
# with open(user_indexing_txt, 'r') as f:
#     for line in f:
#         orig, new = line.strip().split()
#         user_id_map[orig] = new
#
# # Step 3: 构建反向映射：新用户ID → 原始用户ID
# new_to_orig_user = {v: k for k, v in user_id_map.items()}
#
# # Step 4: 过滤 user_sequence_random_indexing_old.txt
# with open(user_seq_old_txt, 'r') as f_in, open(user_seq_clean_txt, 'w') as f_out:
#     for line in f_in:
#         parts = line.strip().split()
#         if not parts:
#             continue
#         new_user_id = parts[0]
#         orig_user_id = new_to_orig_user.get(new_user_id)
#         if orig_user_id in valid_users:
#             f_out.write(line)
#
# # Step 5: 过滤 item_random_indexing_old.txt
# with open(item_index_old_txt, 'r') as f_in, open(item_index_clean_txt, 'w') as f_out:
#     for line in f_in:
#         parts = line.strip().split(maxsplit=1)
#         if not parts:
#             continue
#         item_id = parts[0]
#         if item_id in valid_items:
#             f_out.write(line)
#
# # Step 6: 删除 user_indexing.txt 中不存在于 user_sequence.txt 的用户
# with open(user_indexing_txt, 'r') as f_in, open(user_indexing_new_txt, 'w') as f_out:
#     for line in f_in:
#         orig, new = line.strip().split()
#         if orig in valid_users:
#             f_out.write(line)




