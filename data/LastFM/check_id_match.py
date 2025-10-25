import re

def clean_item_id(item_id):
    # 去除无效的0，保留一个0
    item_id = re.sub(r"<CI0+([1-9])>", r"<CI\1>", item_id)  # 去除无效的0，例如 <CI001> -> <CI1>
    item_id = re.sub(r"<CI0+0>", "<CI0>", item_id)  # 保留一个0，例如 <CI00> -> <CI0>

    # 处理大于10的数字，保留数字并移除无效的0
    item_id = re.sub(r"<CI0*(\d{2,})>", r"<CI\1>", item_id)  # <CI010> -> <CI10>

    return item_id


def process_item_index_file(input_file, output_file):
    # 读取原始文件并处理
    with open(input_file, 'r') as f:
        lines = f.readlines()

    with open(output_file, 'w') as f:
        for line in lines:
            parts = line.split()
            item_id = parts[0]
            item_path = " ".join(parts[1:])

            # 清洗ID
            cleaned_item_path = clean_item_id(item_path)

            # 保存处理后的结果
            f.write(f"{item_id} {cleaned_item_path}\n")

    print(f"处理后的文件已保存到 {output_file}")


# # 输入文件路径和输出文件路径
# input_file = 'item_collaborative_indexing_original.txt'
# output_file = 'item_collaborative_indexing.txt'
#
# # 处理文件,去除无效的0
# process_item_index_file(input_file, output_file)

# 读取文件并提取物品ID
def read_item_ids(file_path):
    item_ids = set()  # 使用集合来存储物品ID，以便后续对比
    with open(file_path, 'r') as file:
        for line in file:
            # 假设每行的格式是：物品ID <CI...> <CI...>
            item_id = line.split()[0]  # 提取物品ID
            item_ids.add(item_id)
    return item_ids


# 检查两个文件中的物品ID是否匹配，并打印物品ID
def check_items_match(file1, file2):
    # 读取两个文件中的物品ID
    item_ids_file1 = read_item_ids(file1)
    item_ids_file2 = read_item_ids(file2)

    # 排序后打印物品ID
    print(f"文件 {file1} 中的物品ID（按升序排序）：")
    for item_id in sorted(item_ids_file1, key=int):
        print(item_id)

    print(f"\n文件 {file2} 中的物品ID（按升序排序）：")
    for item_id in sorted(item_ids_file2, key=int):
        print(item_id)

    # 输出差异
    if item_ids_file1 == item_ids_file2:
        print("\n两个文件中的物品ID完全匹配！")
    else:
        print("\n物品ID不匹配！")
        # 找出不同的物品ID
        missing_in_file2 = item_ids_file1 - item_ids_file2
        missing_in_file1 = item_ids_file2 - item_ids_file1
        if missing_in_file2:
            print(f"\n在{file2}中缺少的物品ID: {sorted(missing_in_file2, key=int)}")
        if missing_in_file1:
            print(f"\n在{file1}中缺少的物品ID: {sorted(missing_in_file1, key=int)}")


# # 调用函数检查两个文件 即经过过滤后是否和作者的物品ID对得上号，保证我们过滤后的数据是一样的
# file1 = 'item_collaborative_indexing.txt'
# file2 = 'item_collaborative_indexing_500_20_sequential_old.txt'
# check_items_match(file1, file2)


