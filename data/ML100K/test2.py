import json
from collections import defaultdict

# ---------- 文件路径 ----------
item_old_file = "item_collaborative_indexing_500_20_sequential_old.txt"
item_new_file = "item_collaborative_indexing_500_20_sequential.txt"
user_seq_old_file = "user_sequence_collaborative_indexing_500_20_sequential_old.txt"
user_seq_new_file = "user_sequence_collaborative_indexing_500_20_sequential.txt"

# ---------- 读取 item 映射（token -> 原始itemID） ----------
def load_item_mapping(filepath):
    mapping = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) != 2:
                print(f"[WARN] 映射行格式异常 @ {filepath}:{ln} -> {line.strip()}")
                continue
            item_id, token = parts
            mapping[token] = item_id
    return mapping

old_map = load_item_mapping(item_old_file)
new_map = load_item_mapping(item_new_file)

# ---------- 将用户序列用映射表还原为“原始itemID序列” ----------
def load_user_sequences(filepath, token2orig):
    seqs = {}
    miss_cnt = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue
            user_id, tokens = parts[0], parts[1:]
            orig = []
            missing = []
            for t in tokens:
                if t in token2orig:
                    orig.append(token2orig[t])
                else:
                    missing.append(t)
                    miss_cnt += 1
            if missing:
                # 打印前几条缺失，避免刷屏
                if miss_cnt <= 10:
                    print(f"[WARN] 缺失映射 @ {filepath}:{ln} user={user_id} tokens={missing[:5]} ...")
            seqs[user_id] = orig
    return seqs, miss_cnt

user_seq_old, miss_old = load_user_sequences(user_seq_old_file, old_map)
user_seq_new, miss_new = load_user_sequences(user_seq_new_file, new_map)

# ---------- 汇总覆盖 ----------
users_old = set(user_seq_old.keys())
users_new = set(user_seq_new.keys())
common_users = users_old & users_new

print(f"旧文件用户数: {len(users_old)}")
print(f"新文件用户数: {len(users_new)}")
print(f"公共用户数  : {len(common_users)}")
print(f"旧->还原 缺失item映射: {miss_old}")
print(f"新->还原 缺失item映射: {miss_new}")

only_in_old = users_old - users_new
only_in_new = users_new - users_old
if only_in_old:
    print(f"[WARN] 仅旧文件存在的用户: {len(only_in_old)}（示例: {list(sorted(only_in_old))[:5]}）")
if only_in_new:
    print(f"[WARN] 仅新文件存在的用户: {len(only_in_new)}（示例: {list(sorted(only_in_new))[:5]}）")

# ---------- 逐用户比对顺序 ----------
mismatch_users = []
first_diffs = {}  # 记录每个不一致用户第一个不同的位置

for u in sorted(common_users, key=lambda x: (len(x), x)):  # 排序仅为稳定输出
    seq_old = user_seq_old[u]
    seq_new = user_seq_new[u]
    if seq_old != seq_new:
        mismatch_users.append(u)
        # 找到第一个不同位置
        L = min(len(seq_old), len(seq_new))
        pos = None
        for i in range(L):
            if seq_old[i] != seq_new[i]:
                pos = i
                break
        if pos is None:
            # 前缀相同但长度不同
            pos = L
        first_diffs[u] = pos

print(f"\n总公共用户: {len(common_users)}")
print(f"顺序不一致的用户数: {len(mismatch_users)}")

if mismatch_users:
    print("\n示例差异（前5个）：")
    for u in mismatch_users[:5]:
        pos = first_diffs[u]
        s_old = user_seq_old[u]
        s_new = user_seq_new[u]
        print(f"\n用户 {u}：第一个差异位置 idx={pos}")
        print("旧 len=", len(s_old), "片段：", s_old[max(0, pos-3):pos+3])
        print("新 len=", len(s_new), "片段：", s_new[max(0, pos-3):pos+3])
else:
    print("✅ 所有公共用户的项目顺序一致（按原始itemID还原后比较）")




# -*- coding: utf-8 -*-
import json, re

# ==== 路径（按需改名；你当前产物名里带 _2_，如果只是兼容别的脚本，先保留也行）====
# USER_SEQ_RAW = "user_sequence.txt"
# ITEM_MAP_FILE = "item_collaborative_indexing_500_2_sequential.txt"  # 原始ASIN -> token
# USER_SEQ_NEW  = "user_sequence_collaborative_indexing_500_2_sequential-from-raw.txt"
#
# # ==== 1) 读取 item 映射并自检 ====
# orig2tok = {}
# tok_set = set()
# with open(ITEM_MAP_FILE, "r", encoding="utf-8") as f:
#     for ln, line in enumerate(f, 1):
#         parts = line.strip().split()
#         if len(parts) != 2:
#             raise ValueError(f"[{ITEM_MAP_FILE}:{ln}] 行格式应为: <ASIN> <token-path>")
#         asin, token = parts
#         if asin in orig2tok:
#             raise ValueError(f"重复的 ASIN: {asin}")
#         orig2tok[asin] = token
#         tok_set.add(token)
#
# print(f"[MAP] 条目数={len(orig2tok)}, token唯一数={len(tok_set)}")
# if len(orig2tok) != len(tok_set):
#     print("⚠️ 警告：存在重复 token（整条路径应唯一），请检查 assign_paths 逻辑")
#
# # 逆映射（用于还原）
# tok2orig = {v:k for k,v in orig2tok.items()}
#
# # ==== 2) 纯替换生成新用户序列（不排序、不去重、不重映射）====
# with open(USER_SEQ_RAW, "r", encoding="utf-8") as fin, open(USER_SEQ_NEW, "w", encoding="utf-8") as fout:
#     miss_item = 0
#     total_lines = 0
#     for line in fin:
#         parts = line.strip().split()
#         if not parts:
#             continue
#         u, items = parts[0], parts[1:]
#         total_lines += 1
#         toks = []
#         for it in items:
#             if it not in orig2tok:
#                 miss_item += 1
#                 # 跳过缺失的 item（或改成直接报错也行）
#                 continue
#             toks.append(orig2tok[it])
#         fout.write(u + " " + " ".join(toks) + "\n")
# print(f"[BUILD] 写出: {USER_SEQ_NEW} | 缺失item映射={miss_item}, 行数={total_lines}")
#
# # ==== 3) 把新序列还原回 ASIN，与原文件逐行对比，验证“顺序未变”====
# mismatch = 0
# with open(USER_SEQ_RAW, "r", encoding="utf-8") as f_raw, open(USER_SEQ_NEW, "r", encoding="utf-8") as f_new:
#     for idx, (l_raw, l_new) in enumerate(zip(f_raw, f_new), 1):
#         pr = l_raw.strip().split()
#         pn = l_new.strip().split()
#         if not pr or not pn:
#             continue
#         ur, ir = pr[0], pr[1:]
#         un, tn = pn[0], pn[1:]
#         # 用户ID必须相同（我们没有改用户ID）
#         if ur != un:
#             mismatch += 1
#             print(f"[用户ID不一致 @ 行{idx}] raw={ur} new={un}")
#             continue
#         # 用新token逆映射回 ASIN
#         ir_back = []
#         missing_back = False
#         for t in tn:
#             if t not in tok2orig:
#                 print(f"[缺逆映射 @ 行{idx}] token={t}")
#                 missing_back = True
#                 break
#             ir_back.append(tok2orig[t])
#         if missing_back:
#             mismatch += 1
#             continue
#         # 对比原始 ASIN 序列
#         if ir != ir_back:
#             mismatch += 1
#             # 打印一个简短片段看差异位置
#             k = min(5, len(ir), len(ir_back))
#             print(f"[顺序不一致 @ 行{idx}]")
#             print("  原始:", ir[:k])
#             print("  还原:", ir_back[:k])
#
# print(f"[CHECK] 顺序不一致行数 = {mismatch}")
# if mismatch == 0:
#     print("✅ 交互顺序保持一致（仅做了逐项替换）")
# else:
#     print("❌ 仍有不一致，请返回检查映射或生成逻辑")
