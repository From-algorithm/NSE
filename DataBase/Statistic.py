# 读取文件
with open('data.txt', 'r') as file:
    lines = file.readlines()

# 初始化节点集合
node_set_1 = set()
node_set_2 = set()

# 遍历每一行
for line in lines:
    elements = line.split()
    node_set_1.add(elements[0])
    node_set_2.add(elements[1])

# 输出结果
print("不同的第一种元素节点数量：", len(node_set_1))
print("不同的第二种元素节点数量：", len(node_set_2))