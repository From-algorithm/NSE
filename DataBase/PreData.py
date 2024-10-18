import os.path as osp
import random
import numpy as np


import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/OGB_MAG')


def random_svd(data, target_dim, num_iterations):
    # 随机初始化投影矩阵
    projection_matrix = np.random.rand(data.shape[1], target_dim)

    for i in range(num_iterations):
        # 使用幂次迭代法更新投影矩阵
        projection_matrix = data.T @ data @ projection_matrix

        # 对投影矩阵进行QR分解
        projection_matrix, _ = np.linalg.qr(projection_matrix)

    # 对数据进行降维
    reduced_data = np.round(data @ projection_matrix, 3)

    return np.round(reduced_data, 3)


target_dim = 64
num_iterations = 5


dataset = OGB_MAG(path)
data = dataset[0]
data_edge = (data[('paper', 'has_topic', 'field_of_study')]['edge_index']).numpy()
data_edge = data_edge.T
# Count the number of papers related to each topic
paper_counts = {}
fea = (data['paper']['x']).numpy()
reduced_data = random_svd(fea, target_dim, num_iterations)
features_list = [arr for arr in reduced_data]


for row in data_edge:
    paper = row[0]
    if paper in paper_counts:
        paper_counts[paper] += 1
    else:
        paper_counts[paper] = 1


filtered_papers = {paper: count for paper, count in paper_counts.items() if count <  101 and count>1}
filtered_papers = dict(random.sample(filtered_papers.items(), 2500))
filtered_data_edge = [row for row in data_edge if row[0] in filtered_papers]


result_dict ={}
for sonlist in filtered_data_edge:
    index = int(sonlist[0])
    value = features_list[index].tolist()
    result_dict[f'P{index}'] = value

with open('result_dict.txt', 'w') as f:
    f.write(str(result_dict))


output_file = "../../Distance/HDE/DataBase/OGB_MAG2500/PF.txt"
with open(output_file, "w") as file:
    for row in filtered_data_edge:
        line = f"P{row[0]} F{row[1]}"
        file.write(line + "\n")