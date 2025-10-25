import json
from collections import defaultdict, deque

# 对所有节点重新排序
with open("cluster_tree.json", "r") as f:
    example = json.load(f)

# Step 1: 构造树结构：parent -> children
tree = defaultdict(set)
for path in example.keys():
    parts = path.split('-')
    for i in range(1, len(parts)):
        parent = '-'.join(parts[:i])
        child = '-'.join(parts[:i + 1])
        tree[parent].add(child)

# Step 2: 层序遍历，从 root 开始，按层编号（从第二层开始）
new_id_map = {}  # 原始路径 → 新编号（仅从第二层开始）
queue = deque(['root'])
current_id = 0
level = 0

while queue:
    level_size = len(queue)
    next_queue = deque()

    for _ in range(level_size):
        node = queue.popleft()
        children = sorted(tree.get(node, []))  # 从左到右

        for child in children:
            if level >= 0:  # 第二层及以后才开始编号
                new_id_map[child] = str(current_id % 25)
                current_id += 1
            next_queue.append(child)

    queue = next_queue
    level += 1

# Step 3: 替换路径为新编号路径
renamed_paths = {}

for path in example.keys():
    parts = path.split('-')[1:]  # 去掉 root
    full_path = "root"
    new_parts = []

    for i in range(1, len(parts) + 1):
        sub_path = "root-" + "-".join(parts[:i])
        if sub_path in new_id_map:
            new_parts.append(new_id_map[sub_path])
        else:
            new_parts.append(parts[i - 1])  # 原样保留（第一层）

    renamed_paths[path] = "-".join(new_parts)

#保存结果为 JSON 文件
with open("renamed_paths.json", "w") as f:
    json.dump(renamed_paths, f, indent=2, ensure_ascii=False)

# 输出结果
print(json.dumps(renamed_paths, indent=2))


# 只对叶子节点重新排序
# # 加载 cluster_tree.json 文件
# with open("cluster_tree.json", "r") as f:
#     cluster_tree = json.load(f)
#
# # 存放叶子节点对应的新路径
# leaf_path_map = {}
#
# for key, leaf_nodes in cluster_tree.items():
#     # 拆掉 "root-"，得到路径部分
#     path_parts = key.replace("root-", "").split("-")
#     base_path = "-".join(path_parts)
#
#     for idx, node in enumerate(leaf_nodes):
#         # 对 25 取模的编号作为最后一段
#         leaf_suffix = str(idx % 25)
#         full_path = f"{base_path}-{leaf_suffix}"
#         leaf_path_map[str(node)] = full_path
#
# # 保存为 JSON 文件
# with open("leaf_node_renamed.json", "w") as f:
#     json.dump(leaf_path_map, f, indent=4)