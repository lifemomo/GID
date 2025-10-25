#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Downsample training interactions per user to simulate data sparsity.
- Input : *_train.json  (either a flat list of dicts, or {user: [dicts]})
- Output: *_train_keep{ratio}_seed{S}.json  (same schema as input)
"""

import argparse, json, os, random
from collections import defaultdict

# ---------- I/O ----------

def load_train(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize to dict[user] -> list[dict]
    if isinstance(data, list):
        # Try common field names
        # Fall back to 'user'/'item'/'time' keys; adjust here if你的字段名不同
        ukey = next((k for k in ("user","user_id","uid","u") if k in data[0]), None)
        if ukey is None:
            raise ValueError("Cannot find user key in list schema. Please rename keys.")
        by_user = defaultdict(list)
        for rec in data:
            by_user[rec[ukey]].append(rec)
        schema = ("list", {"ukey": ukey})
        return by_user, schema
    elif isinstance(data, dict):
        # Assume {user: [records]} schema
        # records 可以是 item id 或 dict；都原样保留
        schema = ("dict", {})
        # 强制把 key 转成 str，防止后面 json dump 时混型
        by_user = {str(u): v for u, v in data.items()}
        return by_user, schema
    else:
        raise ValueError("Unsupported JSON top-level type.")

def save_train(path, by_user, schema, time_key="time"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if schema[0] == "list":
        ukey = schema[1]["ukey"]
        flat = []
        for u, lst in by_user.items():
            for rec in lst:
                # 还原 user 字段（某些库 dump 时会丢失）：
                if isinstance(rec, dict):
                    rec[ukey] = u
                flat.append(rec)
        # 尽量按时间排序（若有 time 字段）
        if flat and isinstance(flat[0], dict) and time_key in flat[0]:
            flat.sort(key=lambda r: r[time_key])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(flat, f, ensure_ascii=False)
    else:  # "dict"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(by_user, f, ensure_ascii=False)

# ---------- Core ----------

def set_seed(s):
    random.seed(s)

def downsample_user_list(user_list, keep_ratio, seed, mode="random", time_key="time"):
    """
    user_list: list of interactions (dict or scalar). Returned list keeps same element type.
    keep_ratio: 0<r<=1
    mode: "random" (默认), or "earliest"
    """
    set_seed(seed)
    m = len(user_list)
    if m == 0:
        return user_list
    k = max(1, int(round(m * keep_ratio)))

    # 如果是顺序模型，先按时间排序以便 'earliest' 语义
    seq = sort_by_time(user_list, time_key=time_key)

    if mode == "earliest":
        sel = seq[:k]
    else:  # random
        # random.sample 不能直接抽 dict 的话，我们对 index 采样
        idxs = list(range(m))
        pick = set(random.sample(idxs, k))
        sel = [seq[i] for i in pick]

    # 训练前通常需要时间升序
    sel = sort_by_time(sel, time_key=time_key)
    return sel

def downsample_train(by_user, keep_ratio, seed, mode="random", time_key="time", min_keep=1):
    new_user = {}
    for u, lst in by_user.items():
        # 如果 lst 是 item id（非 dict），也能工作
        new_user[str(u)] = downsample_user_list(lst, keep_ratio, seed, mode, time_key)
        # 可选：如果你的训练需要至少 L 条历史，可在这里过滤用户
        # if len(new_user[str(u)]) < L: del new_user[str(u)]
    return new_user

def to_list(user_inter):
    """Ensure per-user interactions are a list."""
    if isinstance(user_inter, list):
        return user_inter
    if isinstance(user_inter, dict):
        # 若是编号字典，尽量按数字键排序；否则按插入顺序
        try:
            keys = sorted(user_inter.keys(), key=lambda x: int(x))
            return [user_inter[k] for k in keys]
        except Exception:
            return list(user_inter.values())
    # 其它类型（单条标量）也转成列表
    return [user_inter]

def sort_by_time(lst, time_key="time"):
    """Safely sort a list of interactions by time if可行."""
    lst = to_list(lst)
    if lst and isinstance(lst[0], dict) and (time_key in lst[0]):
        lst = sorted(lst, key=lambda r: r[time_key])
    return lst

def downsample_user_list(user_list, keep_ratio, seed, mode="random", time_key="time"):
    """
    user_list: list/dict of interactions (dict or scalar).
    Return: list after downsampling (保持元素类型不变).
    """
    import random
    random.seed(seed)

    seq = sort_by_time(user_list, time_key=time_key)  # 先统一成 list
    m = len(seq)
    if m == 0:
        return seq
    k = max(1, int(round(m * keep_ratio)))

    if mode == "earliest":
        sel = seq[:k]
    else:
        idxs = list(range(m))
        pick = set(random.sample(idxs, k))
        sel = [seq[i] for i in pick]

    sel = sort_by_time(sel, time_key=time_key)
    return sel

def downsample_train(by_user, keep_ratio, seed, mode="random", time_key="time", min_keep=1):
    new_user = {}
    for u, lst in by_user.items():
        sel = downsample_user_list(lst, keep_ratio, seed, mode, time_key)
        new_user[str(u)] = sel  # 统一存回 list（与大多数加载器兼容）
    return new_user


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--train_json", required=True, help="Path to *_train.json")
    ap.add_argument("--train_json", required=False,
                    default=r"E:\python-3.8.2\LLM\LLM-ID\LastFM\LastFM_sequential,straightforward_collaborative_500_20_sequential_train.json")

    ap.add_argument("--ratios", type=float, nargs="+", default=[0.8, 0.6, 0.4],
                    help="Keep ratios per user, e.g., 0.8 0.6 0.4")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
    ap.add_argument("--mode", choices=["random","earliest"], default="random",
                    help="random=用户内随机保留; earliest=保留最早的 k 条")
    ap.add_argument("--time_key", default="time", help="Timestamp field name if exists")
    args = ap.parse_args()

    by_user, schema = load_train(args.train_json)

    base, ext = os.path.splitext(args.train_json)
    for r in args.ratios:
        assert 0 < r <= 1, "keep ratio must be in (0,1]"
        for s in args.seeds:
            ds = downsample_train(by_user, keep_ratio=r, seed=s,
                                  mode=args.mode, time_key=args.time_key)
            out = f"{base}_keep{int(r*100)}_seed{s}{ext}"
            save_train(out, ds, schema, time_key=args.time_key)
            print(f"Saved: {out}")

if __name__ == "__main__":
    main()
