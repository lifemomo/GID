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


# -*- coding: utf-8 -*-
from collections import Counter, defaultdict

# ---------- 文件 ----------
# USER_SEQ_FILE = "user_sequence.txt"  # 原始: <orig_user> <ASIN...>
# USER_INDEX_FILE = "user_indexing.txt"  # orig -> new
# ITEM_COLL_OLD = "item_collaborative_indexing_500_20_sequential.txt"  # <ASIN> <token...>
# ITEM_COLL_NEW = "item_collaborative_indexing_500_2_sequential.txt"      # <ASIN> <token...>
# USER_TOK_OLD = "user_sequence_collaborative_indexing_500_20_sequential.txt"  # <new_user> <token...>
# USER_TOK_NEW = "user_sequence_collaborative_indexing_500_2_sequential.txt"       # <new_user> <token...>
#
# def load_user_index(path):
#     m = {}
#     with open(path, "r", encoding="utf-8") as f:
#         for ln in f:
#             p = ln.strip().split()
#             if len(p)==2:
#                 m[p[0]] = p[1]
#     return m
#
# def load_asin2tok(path):
#     asin2tok = {}
#     with open(path, "r", encoding="utf-8") as f:
#         for ln in f:
#             p = ln.strip().split()
#             if len(p) >= 2:
#                 asin = p[0]
#                 token = "".join(p[1:])  # 关键：把第2列起拼接
#                 asin2tok[asin] = token
#     return asin2tok
#
# def invert_tok(asin2tok):
#     t2a = defaultdict(list)
#     for a,t in asin2tok.items():
#         t2a[t].append(a)
#     return t2a
#
# def load_user_tokens(path):
#     d = {}
#     with open(path, "r", encoding="utf-8") as f:
#         for ln in f:
#             p = ln.strip().split()
#             if p:
#                 d[p[0]] = p[1:]
#     return d
#
# def load_user_asins(path):
#     seqs = []
#     with open(path, "r", encoding="utf-8") as f:
#         for ln in f:
#             p = ln.strip().split()
#             if len(p) > 1:
#                 seqs.append((p[0], p[1:]))
#     return seqs
#
# def reorder_by_tokens(user_items, token_order, tok2asins):
#     """
#     保留重复次数：先按 token_order 消耗能对上的 ASIN（若1个token对应多个ASIN，选原序列里更靠前的）；
#     再把剩余 ASIN 按原相对顺序补回。
#     """
#     remaining = Counter(user_items)
#     result = []
#     first_pos = {}
#     for i,a in enumerate(user_items):
#         if a not in first_pos:
#             first_pos[a] = i
#     # 按用户 token 顺序放置
#     for tok in token_order:
#         cands = tok2asins.get(tok, [])
#         best, best_pos = None, 10**18
#         for a in cands:
#             if remaining[a] > 0:
#                 pos = first_pos.get(a, 10**18)
#                 if pos < best_pos:
#                     best, best_pos = a, pos
#         if best is not None:
#             result.append(best)
#             remaining[best] -= 1
#     # 剩余按原序补齐（保重复）
#     for a in user_items:
#         while remaining[a] > 0:
#             result.append(a)
#             remaining[a] -= 1
#     assert len(result) == len(user_items)
#     return result
#
# def adjacent_order_agreement(a, b):
#     """
#     计算相邻对一致率：a、b 为相同多重集的两个序列。
#     统计 a 中的相邻有序对是否也以相同相对顺序出现在 b 中（考虑重复，用索引对 disambiguate）。
#     """
#     # 给每个元素加出现序号，消除重复歧义
#     def with_occ(seq):
#         cnt = Counter()
#         out = []
#         for x in seq:
#             cnt[x] += 1
#             out.append((x, cnt[x]))
#         return out
#     A = with_occ(a)
#     B = with_occ(b)
#     posB = {val:i for i,val in enumerate(B)}  # 唯一键: (asin,occ)
#     if len(A) < 2:
#         return 1.0
#     total = 0
#     agree = 0
#     for i in range(len(A)-1):
#         total += 1
#         u, v = A[i], A[i+1]
#         if posB[u] < posB[v]:
#             agree += 1
#     return agree / total
#
# # 加载
# user_map = load_user_index(USER_INDEX_FILE)
# asin2tok_old = load_asin2tok(ITEM_COLL_OLD)
# asin2tok_new = load_asin2tok(ITEM_COLL_NEW)
# tok2asins_old = invert_tok(asin2tok_old)
# tok2asins_new = invert_tok(asin2tok_new)
# user_tok_old = load_user_tokens(USER_TOK_OLD)
# user_tok_new = load_user_tokens(USER_TOK_NEW)
# raw_users = load_user_asins(USER_SEQ_FILE)
#
# # 评估
# n = 0
# exact_same = 0
# adj_agreements = []
# for orig_u, asins in raw_users:
#     new_u = user_map.get(orig_u)
#     if new_u is None:
#         continue
#     tok_order_old = user_tok_old.get(new_u)
#     tok_order_new = user_tok_new.get(new_u)
#     if tok_order_old is None or tok_order_new is None:
#         continue
#
#     seq_old = reorder_by_tokens(asins, tok_order_old, tok2asins_old)
#     seq_new = reorder_by_tokens(asins, tok_order_new, tok2asins_new)
#
#     n += 1
#     if seq_old == seq_new:
#         exact_same += 1
#     adj_agreements.append(adjacent_order_agreement(seq_old, seq_new))
#
# print(f"覆盖评估的用户数: {n}")
# print(f"完全相同的用户序列: {exact_same}/{n}")
# if adj_agreements:
#     adj_agreements.sort()
#     mean = sum(adj_agreements)/len(adj_agreements)
#     median = adj_agreements[len(adj_agreements)//2]
#     print(f"相邻对一致率: mean={mean:.3f}, median={median:.3f}, min={adj_agreements[0]:.3f}, max={adj_agreements[-1]:.3f}")



from collections import Counter
import statistics as stats

# ---------- 文件路径 ----------
# USER_SEQ_REORDERED = "user_sequence.txt"
# USER_INDEX_FILE = "user_indexing.txt"
# USER_SEQ_COLLAB_OLD = "user_sequence_collaborative_indexing_500_20_sequential-old.txt"
# ITEM_COLLAB_FILE = "item_collaborative_indexing_500_20_sequential-old.txt"
#
# # ---------- 读映射 ----------
# # 原始用户 -> 新用户
# user_map = {}
# with open(USER_INDEX_FILE, "r", encoding="utf-8") as f:
#     for ln in f:
#         p = ln.strip().split()
#         if len(p) == 2:
#             user_map[p[0]] = p[1]
#
# # ASIN -> token（把第2列起全部拼接）
# asin2tok = {}
# with open(ITEM_COLLAB_FILE, "r", encoding="utf-8") as f:
#     for ln in f:
#         p = ln.strip().split()
#         if len(p) >= 2:
#             asin2tok[p[0]] = "".join(p[1:])
#
# # 新用户 -> gold token 序列
# gold_by_user = {}
# with open(USER_SEQ_COLLAB_OLD, "r", encoding="utf-8") as f:
#     for ln in f:
#         p = ln.strip().split()
#         if p:
#             gold_by_user[p[0]] = p[1:]
#
# # ---------- 工具 ----------
# def intersection_order_equal(mapped_tokens, gold_tokens):
#     """gold 中按顺序取出 mapped 的多重交集，检查顺序是否一致。"""
#     need = Counter(mapped_tokens)
#     filtered = []
#     for t in gold_tokens:
#         if need[t] > 0:
#             filtered.append(t)
#             need[t] -= 1
#     return filtered == mapped_tokens, filtered
#
# # ---------- 校验 ----------
# tot_users = 0
# strict_equal = 0
# intersect_equal = 0
# len_mismatch_users = 0
# order_mismatch_users = 0
# missing_user = 0
# missing_item_tokens = 0
#
# len_diffs = []      # gold_len - mapped_len
# ratios = []         # mapped_len / gold_len
#
# PRINT_N = 3
# printed_len = 0
# printed_ord = 0
#
# with open(USER_SEQ_REORDERED, "r", encoding="utf-8") as f:
#     for ln in f:
#         p = ln.strip().split()
#         if not p:
#             continue
#         orig_u, asins = p[0], p[1:]
#         tot_users += 1
#
#         new_u = user_map.get(orig_u)
#         if new_u is None or new_u not in gold_by_user:
#             missing_user += 1
#             continue
#
#         # map ASIN -> token（保留重复与长度）
#         mapped = []
#         ok = True
#         for a in asins:
#             t = asin2tok.get(a)
#             if t is None:
#                 mapped.append(None)  # 标记缺失
#                 ok = False
#                 missing_item_tokens += 1
#             else:
#                 mapped.append(t)
#
#         gold = gold_by_user[new_u]
#
#         # 记录长度差
#         len_diffs.append(len(gold) - len(mapped))
#         if len(gold) > 0:
#             ratios.append(len(mapped) / len(gold))
#         else:
#             ratios.append(1.0)
#
#         if len(mapped) != len(gold):
#             len_mismatch_users += 1
#
#         # 严格全等
#         if mapped == gold:
#             strict_equal += 1
#             intersect_equal += 1  # 严格全等必然交集一致
#             continue
#
#         # 交集顺序一致
#         eq, gold_filtered = intersection_order_equal(mapped, gold)
#         if eq:
#             intersect_equal += 1
#         else:
#             order_mismatch_users += 1
#             if printed_ord < PRINT_N:
#                 print(f"[Order mismatch] user {orig_u} -> new {new_u}")
#                 # 找第一个不等位置
#                 idx = next((i for i,(a,b) in enumerate(zip(mapped, gold_filtered)) if a != b), None)
#                 if idx is not None:
#                     lo, hi = max(0, idx-3), min(len(mapped), idx+4)
#                     print("  mapped:", mapped[lo:hi])
#                     print("  gold∩mapped:", gold_filtered[lo:hi])
#                 printed_ord += 1
#
# # ---------- 汇总 ----------
# def fmt_stats(vals):
#     if not vals: return "n=0"
#     return f"n={len(vals)}, mean={stats.mean(vals):.3f}, median={stats.median(vals):.3f}, min={min(vals)}, max={max(vals)}"
#
# print("-"*60)
# print(f"用户总数: {tot_users}")
# print(f"严格全等用户数: {strict_equal}")
# print(f"交集顺序一致用户数: {intersect_equal}")
# print(f"长度不一致用户数: {len_mismatch_users}")
# print(f"顺序不一致用户数（基于交集）: {order_mismatch_users}")
# print(f"缺少用户映射/旧文件缺失: {missing_user}")
# print(f"缺少物品token条数: {missing_item_tokens}")
# print(f"gold_len - mapped_len 统计: {fmt_stats(len_diffs)}")
# print(f"mapped_len / gold_len 统计: {fmt_stats(ratios)}")


#统计数据集信息
import os, gzip, json, ast
from collections import defaultdict, Counter

# ======== 路径 ========
# USER_SEQ_FILE = "user_sequence.txt"
# META_FILE = "filtered_meta_Movies_and_TV.json.gz"
#
# # ======== 属性抽取配置（按需开关） ========
# ATTR_CFG = {
#     "categories_leaf": True,   # 取 categories 的叶子类目
#     "categories_path": False,  # 或者取完整路径 'A>B>C'（与上互不排斥，可同时开启）
#     "brand": True,
#     "genres": True,            # 'genre' (str) 或 'genres' (list)
#     # 你还可以增加：'format','studio','publisher' ... -> 在 extract_attrs() 里按需要扩展
# }
#
# # ========= 工具 =========
# def parse_maybe_non_json(line: str):
#     """优先 json.loads；失败再 ast.literal_eval（适配 2014 版的“松散 JSON”）。"""
#     try:
#         return json.loads(line)
#     except Exception:
#         try:
#             return ast.literal_eval(line)
#         except Exception:
#             return None
#
# def gini(values):
#     vals = [v for v in values if v > 0]
#     if not vals: return 0.0
#     vals.sort()
#     n = len(vals); s = sum(vals)
#     cum = sum((i+1)*x for i,x in enumerate(vals))
#     return (2*cum)/(n*s) - (n+1)/n
#
# # ========= 读取交互，得到核心集合 =========
# users, items, E_total = set(), set(), 0
# user_items = defaultdict(list)
# with open(USER_SEQ_FILE, "r", encoding="utf-8") as f:
#     for line in f:
#         p = line.strip().split()
#         if len(p) < 2:
#             continue
#         u, its = p[0], p[1:]
#         users.add(u)
#         for it in its:
#             items.add(it)
#             user_items[u].append(it)
#         E_total += len(its)
#
# unique_edges = set()
# for u, its in user_items.items():
#     for it in its:
#         unique_edges.add((u, it))
#
# U, I = len(users), len(items)
# E_unique = len(unique_edges)
# density = (E_unique / (U*I)) if U and I else 0.0
# sparsity_pct = (1 - density) * 100
#
# print("=== Core stats ===")
# print(f"#User = {U:,}")
# print(f"#Item = {I:,}")
# print(f"#Interactions (total, with duplicates) = {E_total:,}")
# print(f"#Interactions (unique user-item pairs) = {E_unique:,}")
# print(f"Sparsity (%) = {sparsity_pct:.4f}")
# print(f"Avg interactions/user (total) = {(E_total/U if U else 0):.3f}")
# print(f"Avg interactions/item (total) = {(E_total/I if I else 0):.3f}")
# print(f"Avg interactions/user (unique) = {(E_unique/U if U else 0):.3f}")
# print(f"Avg interactions/item (unique) = {(E_unique/I if I else 0):.3f}")
#
# # ========= 从 meta 提取项目属性 =========
# def normalize_text(x):
#     if x is None: return None
#     s = str(x).strip()
#     return s if s else None
#
# def extract_attrs(rec):
#     """
#     返回该 item 的属性集合（set[str]）。
#     这里统一以 'field:value' 形式编码，避免同名值跨字段冲突。
#     """
#     attrs = set()
#     # categories: list[list[str]]
#     if ATTR_CFG.get("categories_leaf") or ATTR_CFG.get("categories_path"):
#         cats = rec.get("categories") or rec.get("category")
#         if isinstance(cats, list):
#             for path in cats:
#                 if isinstance(path, list) and path:
#                     if ATTR_CFG.get("categories_leaf"):
#                         leaf = normalize_text(path[-1])
#                         if leaf: attrs.add(f"cat:{leaf}")
#                     if ATTR_CFG.get("categories_path"):
#                         path_str = ">".join([str(x).strip() for x in path if normalize_text(x)])
#                         if path_str:
#                             attrs.add(f"catpath:{path_str}")
#         # 某些数据是单层 list[str]
#         elif isinstance(cats, str):
#             if ATTR_CFG.get("categories_leaf"):
#                 v = normalize_text(cats)
#                 if v: attrs.add(f"cat:{v}")
#
#     # brand
#     if ATTR_CFG.get("brand"):
#         b = normalize_text(rec.get("brand"))
#         if b: attrs.add(f"brand:{b}")
#
#     # genres
#     if ATTR_CFG.get("genres"):
#         gs = rec.get("genres") or rec.get("genre")
#         if isinstance(gs, list):
#             for g in gs:
#                 g = normalize_text(g)
#                 if g: attrs.add(f"genre:{g}")
#         else:
#             g = normalize_text(gs)
#             if g: attrs.add(f"genre:{g}")
#
#     # 你可以按需开启更多字段：
#     # for key in ["format","studio","publisher"]:
#     #     if ATTR_CFG.get(key):
#     #         v = rec.get(key)
#     #         if isinstance(v, list):
#     #             for t in v:
#     #                 t = normalize_text(t)
#     #                 if t: attrs.add(f"{key}:{t}")
#     #         else:
#     #             v = normalize_text(v)
#     #             if v: attrs.add(f"{key}:{v}")
#
#     return attrs
#
# item2attrs = {}
# in_meta, parsed_ok = 0, 0
# with gzip.open(META_FILE, "rt", encoding="utf-8", errors="ignore") as fin:
#     for line in fin:
#         rec = parse_maybe_non_json(line)
#         if not isinstance(rec, dict):
#             continue
#         asin = rec.get("asin")
#         if asin in items:  # 只关心出现在交互里的项目
#             in_meta += 1
#             attrs = extract_attrs(rec)
#             item2attrs[asin] = attrs
#             parsed_ok += 1
#
# covered_items = sum(1 for it in items if item2attrs.get(it))
# coverage_pct = (covered_items / I * 100) if I else 0.0
#
# # 属性全集与非零计数
# all_attrs = set()
# for s in item2attrs.values():
#     all_attrs |= s
# A = len(all_attrs)
# IA = sum(len(s) for s in item2attrs.values())  # item-attr 非零对
#
# attr_density = (IA / (I*A)) if (I and A) else 0.0
# attr_sparsity_pct = (1 - attr_density) * 100 if A else 100.0
# avg_attrs_per_item = (IA / I) if I else 0.0
# avg_items_per_attr = (IA / A) if A else 0.0
#
# # 属性流行度分布
# attr_pop = Counter()
# for s in item2attrs.values():
#     for a in s:
#         attr_pop[a] += 1
# gini_attr = gini(attr_pop.values())
# topk = max(1, int(0.01 * max(1, len(attr_pop))))  # Top 1%
# topk_share = (sum(sorted(attr_pop.values(), reverse=True)[:topk]) / (sum(attr_pop.values()) or 1))
#
# # 用户属性多样性（按用户交互到的唯一属性数）
# user_attr_div = []
# for u, its in user_items.items():
#     aset = set()
#     for it in its:
#         aset |= item2attrs.get(it, set())
#     user_attr_div.append(len(aset))
# avg_user_attr_div = (sum(user_attr_div)/len(user_attr_div)) if user_attr_div else 0.0
#
# print("\n=== Attribute stats (from filtered_meta_Movies_and_TV.json.gz) ===")
# print(f"Items found in meta = {in_meta:,} / {I:,} (coverage of meta parse: {coverage_pct:.2f}%)")
# print(f"#Attribute = {A:,}")
# print(f"#(item,attr) non-zeros = {IA:,}")
# print(f"Attribute sparsity (%) = {attr_sparsity_pct:.4f}")
# print(f"Avg attributes per item = {avg_attrs_per_item:.3f}")
# print(f"Avg items per attribute = {avg_items_per_attr:.3f}")
# print(f"Attribute popularity Gini = {gini_attr:.3f}")
# print(f"Top-1% attributes' share = {topk_share*100:.2f}%")
# print(f"Avg attribute diversity per user = {avg_user_attr_div:.3f}")





