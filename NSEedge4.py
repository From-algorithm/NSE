import json

import networkx as nx
import random
import numpy as np
import tensorflow as tf
from itertools import combinations

seed_id = 0
np.random.seed(seed_id)  # 确保各个算法的 数据，邻居等信息一致，保证公平
USE_DE = 1
###################################################################
# 每次修改数据集，需变更PATH、type2idx、type_list、feature_data读取这四部分
###################################################################
PATH = 'DataBase/PubMed_Mini/SG.txt'
type2idx = {'S': 0, 'G': 1}
# Struct Encoder#######################################################################
type_list = ['S', 'G']
# Feature Injection#######################################################################
# batch_data有一处必须修改，即更改edge_list结点顺序
EPOCH = 400
NUM_NEIGHBOR = 5
K_HOP = 2  # 聚合K_HOP的邻居
BATCH_SIZE = 32
EMB_DIM = 64


def load_dataset():
    test_ratio = 0.2
    G = nx.Graph()
    sp = 1 - test_ratio * 2
    NODE_TYPE = len(type2idx)
    edge_list = []

    with open(PATH, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            G.add_edge(line[0], line[1])
            edge_list.append(line)

    print('PATH_Add Already!')

    num_edge = len(edge_list)
    sp1 = int(num_edge * sp)
    sp2 = int(num_edge * test_ratio)
    sp3 = num_edge - sp1 - sp2
    print("总共边数量：" + str(num_edge) + "训练边数量：" + str(sp1) + "验证边数量：" + str(sp3) + "测试边数量：" + str(sp2))
    G_train = nx.Graph()
    G_val = nx.Graph()
    G_test = nx.Graph()

    G_train.add_edges_from(edge_list[:sp1])
    G_val.add_edges_from(edge_list[sp1:sp1 + sp2])
    G_test.add_edges_from(edge_list[sp1 + sp2:])
    print(
        f"all edge: {len(G.edges)}, train edge: {len(G_train.edges)}, val edge: {len(G_val.edges)}, test edge: {len(G_test.edges)}")
    return G_train, G_val, G_test, NODE_TYPE, type2idx


# ######################################## 加载数据 ########################################
G_train, G_val, G_test, NODE_TYPE, type2idx = load_dataset()

mini_batch = []
fea_batch = []

if USE_DE:
    NUM_FEA = 4 * 2 + NODE_TYPE
else:
    NUM_FEA = NODE_TYPE
# 根据输入和输出的节点数量自适应地调整权重的初始化范围.uniform=False 表示初始化器不使用均匀分布，而是使用正态分布进行初始化。
initializer = tf.contrib.layers.xavier_initializer(uniform=False)
# L2 正则化是一种常用的正则化方法用于解决过拟合问题，它会在损失函数中增加一个正则化项，以减少模型的过拟合。这里的参数 0.0 表示正则化项的系数为 0，即不进行正则化。
regularizer = tf.contrib.layers.l2_regularizer(0.0)


# 这段代码的思想是计算从源节点到目标节点的最短路径在各个节点类型下的最少出现次数。
def dist_encoder(src, dest, G, K_HOP):
    # 这个是起点到终点的状态编码，若src=dest，则[0.,0.]；若src到得了dest，则[0.,1.]；若src到得了dest，
    # 但是他们的最近路途长度大于5个节点，即src->node1->node2->node3->node4->dest，则编码[1.,0.]；若无法抵达，则[1.,1.].
    # 计算在各个类型下的SPD=最少出现次数
    # 代码首先使用nx.all_simple_paths函数获取从源节点到目标节点的所有简单路径，其中路径的最大长度为K_HOP + 2。
    # 如果是空的就是[]; 如果不是就起码有好几个但是只要取第一个[link1, link2, link3]; link用paths[0]表示，
    paths = list(nx.all_simple_paths(G, src, dest, cutoff=K_HOP + 2))
    sorted_paths = sorted(paths, key=len)
    array = np.array([0.0, 0.0, 0.0, 0.0])
    if sorted_paths:
        for i in range(min(len(sorted_paths[0]) - 1, 2)):
            if sorted_paths[0][i + 1][0] == type_list[0]:
                array[2 * i] = 1.0
                array[2 * i + 1] = 0.0
            else:
                array[2 * i] = 0.0
                array[2 * i + 1] = 1.0
    return array


# nx.all_simple_paths() 提取节点的字母，比如M293就是M，那么M这个计数值+1
def type_encoder(node):
    res = [0.0] * NODE_TYPE
    res[type2idx[node[0]]] = 1.0
    return res


def gen_fea_batch1(G, root, des, fea_dict, K_HOP):
    fea_batch = []
    mini_batch.append([root])
    # 两个相对位置的onehot
    if USE_DE:
        # a这个向量的最后两个值，如果是A型节点则为1 0 ，M型节点为0 1
        a = [0.0] * 4 * 2 + type_encoder(root)
    else:
        a = type_encoder(root)
    # np.asarray(a, dtype=np.float32) 将列表 a 转换为一个 NumPy 数组 ；使用 .reshape(-1, NUM_FEA) 其中 -1 表示自动计算该维度的大小，
    fea_batch.append(np.asarray(a, dtype=np.float32).reshape(-1, NUM_FEA))

    # 1-ord
    ns_1 = []
    for node in mini_batch[-1]:
        # 获取节点的邻居节点列表，并添加自身到邻居列表中
        neighbors = list(G.neighbors(node)) + [node]
        # 从邻居列表中随机选择 NUM_NEIGHBOR 个节点，允许重复选择
        chosen_neighbors = np.random.choice(neighbors, NUM_NEIGHBOR, replace=True)
        # 将选择的邻居节点列表添加到 ns_1 列表中
        ns_1.append(list(chosen_neighbors))
    mini_batch.append(ns_1[0])

    # 创建一个空的 de_1 列表
    de_1 = []
    if USE_DE:
        # 遍历 ns_1[0] 中的每个节点
        for dest in ns_1[0]:
            # 获取目标节点 dest 的特征向量和类型编码，并将它们拼接在一起
            dest_features = fea_dict[dest]
            dest_type_encoding = np.asarray(type_encoder(dest))
            dest_combined = np.concatenate([dest_features, dest_type_encoding], axis=0)
            # 将拼接后的向量添加到 de_1 列表中
            de_1.append(dest_combined)
    else:
        # 遍历 ns_1[0] 中的每个节点
        for dest in ns_1[0]:
            # 获取目标节点 dest 的类型编码，并将其转换为 NumPy 数组
            dest_type_encoding = np.asarray(type_encoder(dest))
            # 将类型编码添加到 de_1 列表中
            de_1.append(dest_type_encoding)
    # 将 de_1 列表转换为 NumPy 数组，并添加到 fea_batch 列表中
    fea_batch.append(np.asarray(de_1, dtype=np.float32).reshape(1, -1))

    # 2-order 根据1-order生成的目标节点随机五个邻居节点，再来生成这五个邻居节点的各自五个邻居节点向量
    ns_2 = [list(np.random.choice(list(G.neighbors(node)) + [node], NUM_NEIGHBOR, replace=True)) for node in
            mini_batch[-1]]
    de_2 = []
    for i in range(len(ns_2)):
        tmp = []
        for j in range(len(ns_2[0])):
            # fea_dict[ns_2[i][j]] + type_encoder(ns_2[i][j])
            if USE_DE:
                tmp.append(np.concatenate([fea_dict[ns_2[i][j]], np.asarray(type_encoder(ns_2[i][j]))], axis=0))
            else:
                tmp.append(np.asarray(type_encoder(ns_2[i][j])))
        de_2.append(tmp)
    fea_batch.append(np.asarray(de_2, dtype=np.float32).reshape(1, -1))
    return np.concatenate(fea_batch, axis=1)


def gen_fea_batch2(G, root, fea_dict, K_HOP):
    fea_batch = []
    mini_batch.append([root])
    # 两个相对位置的onehot
    if USE_DE:
        # a这个向量的最后两个值，如果是M型节点则为1.0 ，A型节点为0.0
        a = [0.0] * 4 * 2 + type_encoder(root)
    else:
        a = type_encoder(root)
    # np.asarray(a, dtype=np.float32) 将列表 a 转换为一个 NumPy 数组 ；使用 .reshape(-1, NUM_FEA) 其中 -1 表示自动计算该维度的大小，
    fea_batch.append(np.asarray(a, dtype=np.float32).reshape(-1, NUM_FEA))

    # 1-ord
    # if len(G.neighbors(node)) < 1:+
    #     print(node)
    # 邻居集合补上自己，因为subG可能有孤立点
    # 遍历 mini_batch[-1] 中的每个节点
    ns_1 = []
    for node in mini_batch[-1]:
        # 获取节点的邻居节点列表，并添加自身到邻居列表中
        neighbors = list(G.neighbors(node)) + [node]
        # 从邻居列表中随机选择 NUM_NEIGHBOR 个节点，允许重复选择
        chosen_neighbors = np.random.choice(neighbors, NUM_NEIGHBOR, replace=True)
        # 将选择的邻居节点列表添加到 ns_1 列表中
        ns_1.append(list(chosen_neighbors))
    mini_batch.append(ns_1[0])
    # 创建一个空的 de_1 列表
    de_1 = []
    if USE_DE:
        # 遍历 ns_1[0] 中的每个节点
        for dest in ns_1[0]:
            # 获取目标节点 dest 的特征向量和类型编码，并将它们拼接在一起
            dest_features = fea_dict[dest]
            dest_type_encoding = np.asarray(type_encoder(dest))
            dest_combined = np.concatenate([dest_features, dest_type_encoding], axis=0)
            # 将拼接后的向量添加到 de_1 列表中
            de_1.append(dest_combined)
    else:
        # 遍历 ns_1[0] 中的每个节点
        for dest in ns_1[0]:
            # 获取目标节点 dest 的类型编码，并将其转换为 NumPy 数组
            dest_type_encoding = np.asarray(type_encoder(dest))
            # 将类型编码添加到 de_1 列表中
            de_1.append(dest_type_encoding)
    # 将 de_1 列表转换为 NumPy 数组，并添加到 fea_batch 列表中
    fea_batch.append(np.asarray(de_1, dtype=np.float32).reshape(1, -1))

    # 2-order 根据1-order生成的目标节点随机五个邻居节点，再来生成这五个邻居节点的各自五个邻居节点向量
    ns_2 = [list(np.random.choice(list(G.neighbors(node)) + [node], NUM_NEIGHBOR, replace=True)) for node in
            mini_batch[-1]]
    de_2 = []
    for i in range(len(ns_2)):
        tmp = []
        for j in range(len(ns_2[0])):
            # fea_dict[ns_2[i][j]] + type_encoder(ns_2[i][j])
            if USE_DE:
                tmp.append(np.concatenate([fea_dict[ns_2[i][j]], np.asarray(type_encoder(ns_2[i][j]))], axis=0))
            else:
                tmp.append(np.asarray(type_encoder(ns_2[i][j])))
        de_2.append(tmp)
    fea_batch.append(np.asarray(de_2, dtype=np.float32).reshape(1, -1))

    return np.concatenate(fea_batch, axis=1)


# v2
# print(G.degree('A1'), G.degree('A3'))
# 函数的目的是从图G中获取包含给定节点对node_pair的子图。它首先使用nx.ego_graph函数分别获取以节点A和节点B为中心、半径为K_HOP的邻域子图A_ego和B_ego。

def subgraph_sampling_with_DE_node_pair(G, node_pair, K_HOP=2):
    # print('edge_DE .... ')
    [A, B] = node_pair
    # 生成以节点 A 为中心、半径为 K_HOP 的邻域子图 A_ego
    A_ego = nx.ego_graph(G, A, radius=K_HOP)
    # plt.figure()
    # pos = nx.spring_layout(A_ego)
    # nx.draw(A_ego, pos=pos, with_labels=True, font_size=6)
    # plt.show()
    # print(nx.shortest_path_length(A_ego, A))
    B_ego = nx.ego_graph(G, B, radius=K_HOP)
    # plt.figure()
    # pos = nx.spring_layout(B_ego)
    # nx.draw(B_ego, pos=pos, with_labels=True, font_size=6)
    # plt.show()

    # 然后，使用nx.compose函数将两个邻域子图合并成一个子图sub_G_for_AB。
    # 接下来，使用sub_G_for_AB.remove_edges_from(combinations(node_pair, 2))移除子图中节点对之间的边
    sub_G_for_AB = nx.compose(A_ego, B_ego)

    sub_G_for_AB.remove_edges_from(combinations(node_pair, 2))

    sub_G_nodes = sub_G_for_AB.nodes

    # 绘制子图 sub_G_for_AB
    # plt.figure()
    # pos = nx.spring_layout(sub_G_for_AB)
    # nx.draw(sub_G_for_AB, pos=pos, with_labels=True, font_size=6)
    # plt.show()

    # 下面这段代码主要功能是计算子图中所有节点到给定节点对的距离，并生成与节点对相关的特征。
    # 创建空字典SPD_based_on_node_pair，用于存储每个节点到给定节点对的距离信息。
    SPD_based_on_node_pair = {}
    # 子图中的每个节点，代码调用dist_encoder函数计算它与节点对之间的距离编码。其中，dist_encoder函数会根据节点类型统计最短路径在不同节点类型下的最少出现次数，并返回距离编码结果。
    if USE_DE:
        for node in sub_G_nodes:
            tmpA = dist_encoder(A, node, sub_G_for_AB, K_HOP)
            tmpB = dist_encoder(B, node, sub_G_for_AB, K_HOP)
            SPD_based_on_node_pair[node] = np.concatenate([tmpA, tmpB], axis=0)

    # A he B 的聚合图
    # 为什么区分gen_fea_batch1和gen_fea_batch2，因为node_pair的A一定是数据集的第一种类型节点，而node_pair的B由于随机负采样可能会有第一种或第二种两种节点，
    # 但是我们输入的节点特征只有一种，故部分情况无法计算。 能够输出相似度矩阵的必然是node_pair A为第一种类型，B为第二种类型。
    # 他会拿出A的一阶邻居和B来计算相似度
    A_fea_batch = gen_fea_batch1(sub_G_for_AB, A, B,
                                 SPD_based_on_node_pair, K_HOP)
    B_fea_batch = gen_fea_batch2(sub_G_for_AB, B,
                                 SPD_based_on_node_pair, K_HOP)
    return A_fea_batch, B_fea_batch


def batch_data(G, batch_size=3):
    edge = list(G.edges)
    nodes = list(G.nodes)
    num_batch = int(len(edge) / batch_size)
    random.shuffle(edge)
    # #################必须要改的地方 ###############################
    edge = [(x, y) if x.startswith(list(type2idx.keys())[0]) else (y, x) for x, y in edge]
    for idx in range(num_batch):
        # TODO add shuffle and random sample
        batch_edge = edge[idx * batch_size:(idx + 1) * batch_size]
        batch_label = [1.0] * batch_size
        # label[idx * batch_size:(idx + 1) * batch_size]

        batch_A_fea = []
        batch_B_fea = []
        batch_y = []
        #
        # neg_batch_A_fea = []
        # neg_batch_B_fea = []
        # neg_batch_y = []

        # for (edge, label) in zip(batch_edge, batch_label):
        for (bx, by) in zip(batch_edge, batch_label):
            # print(bx, by)

            # 正样本对
            posA, posB = subgraph_sampling_with_DE_node_pair(G, bx, K_HOP=K_HOP)
            batch_A_fea.append(posA)
            batch_B_fea.append(posB)
            batch_y.append(np.asarray(by, dtype=np.float32))

            # 负样本对
            # batch_A_fea.append(tmpA)
            # TODO do not consider sampling pos as neg
            neg_tmpB_id = random.choice(nodes)
            negA, negB = subgraph_sampling_with_DE_node_pair(G, [bx[0], neg_tmpB_id], K_HOP=K_HOP)
            batch_A_fea.append(negA)
            batch_B_fea.append(negB)
            batch_y.append(np.asarray(0.0, dtype=np.float32))

        yield np.asarray(np.squeeze(batch_A_fea)), np.asarray(np.squeeze(batch_B_fea)), np.asarray(batch_y).reshape(
            batch_size * 2, 1)


# split data
def split(G, split=0.8):
    edge_list = list(G.edges)
    num_edge = len(edge_list)
    sp = int(num_edge * split)
    train_edge = edge_list[:sp]
    train_label = [1.0] * sp  # np.ones(sp)

    test_edge = edge_list[sp:]
    test_label = [1.0] * (num_edge - sp)  # np.ones(sp)
    # train_data = (train_edge, train_label]
    # test_data = [test_edge, test_label]
    return train_edge, train_label, test_edge, test_label
    # return train_edge, test_edge


def ESGNN(fea, model='meirec'):
    """
    :param fea: fea_batch, [[0, 0, 4], [[0, 1, 1], [0, 1, 4], [0, 1, 4], [0, 1, 1], [1, 0, 1]]]
    :return:
    """
    with tf.variable_scope(name_or_scope='ESGNN', reuse=tf.AUTO_REUSE):
        # 提取node、1阶邻居特征、2阶邻居特征
        node = fea[:, :NUM_FEA]
        neigh1 = fea[:, NUM_FEA:NUM_FEA * (NUM_NEIGHBOR + 1)]
        neigh1 = tf.reshape(neigh1, [-1, NUM_NEIGHBOR, NUM_FEA])
        neigh2 = fea[:, NUM_FEA * (NUM_NEIGHBOR + 1):]
        neigh2 = tf.reshape(neigh2, [-1, NUM_NEIGHBOR, NUM_NEIGHBOR, NUM_FEA])
        if model == 'meirec':
            # 五个二级节点相加求平均
            neigh2_agg = tf.reduce_mean(neigh2, axis=2)
            tmp = tf.concat([neigh1, neigh2_agg], axis=2)
            tmp = tf.layers.dense(tmp, EMB_DIM,
                                  activation=tf.nn.elu,
                                  use_bias=True,
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer,
                                  name='tmp_proj'
                                  )
            # tmp再次聚合5个个1级节点求平均再和节点本身相连接
            # tmp再次聚合5个个1级节点求平均再和节点本身相连接
            order_nodemean = tf.reduce_mean(tmp, axis=1)
            emb = tf.concat([node, order_nodemean], axis=1)
        emb = tf.layers.dense(emb, EMB_DIM,
                              activation=tf.nn.elu,
                              use_bias=True,
                              kernel_initializer=initializer,
                              kernel_regularizer=regularizer,
                              name='emb_proj'
                              )
        emb = tf.layers.dense(emb, EMB_DIM,
                              activation=tf.nn.elu,
                              use_bias=True,
                              kernel_initializer=initializer,
                              kernel_regularizer=regularizer,
                              name='emb_proj_2'
                              )
        return emb


# Link Prediction用于计算节点对之间的连接概率，并返回预测结果、损失值、AUC值、节点1和节点2的嵌入向量
def LinkPrediction(n1, n2, label):
    # 通过调用函数ESGNN(n1)和ESGNN(n2)，分别对节点n1和n2进行图神经网络（ESGNN）的嵌入操作
    n1_emb = ESGNN(n1)
    n2_emb = ESGNN(n2)
    # 定义了一个隐藏层的全连接层，然后定义了输出层的全连接层
    # 全连接层tf.layers.dense对特征向量进行预测，得到输出向量pred。该全连接层具有32个神经元，激活函数为ELU
    pred = tf.layers.dense(tf.concat([n1_emb, n2_emb], axis=1),
                           32,
                           activation=tf.nn.elu,
                           use_bias=True,
                           kernel_initializer=initializer,
                           kernel_regularizer=regularizer,
                           name='pred_layer'
                           )
    pred = tf.layers.dense(pred,
                           16,
                           # leaky_relu尝试到了0.2-0.4比较好，这里选用0.4
                           activation=tf.nn.leaky_relu,
                           use_bias=True,
                           kernel_initializer=initializer,
                           kernel_regularizer=regularizer,
                           name='pred_layer_1'
                           )
    pred = tf.layers.dense(pred,
                           1,
                           activation=None,
                           use_bias=True,
                           kernel_initializer=initializer,
                           kernel_regularizer=regularizer,
                           name='pred_layer_2'
                           )
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=pred))
    auc, auc_op = tf.metrics.auc(labels=label, predictions=tf.nn.sigmoid(pred))
    # 准确率指标
    _, accuracy = tf.metrics.accuracy(labels=label, predictions=tf.round(tf.nn.sigmoid(pred)))
    # tf.metrics.ac
    return pred, loss, auc, auc_op, n1_emb, n2_emb, accuracy


# shape=(None, (NUM_NEIGHBOR + 1) * NUM_FEA)
# 定义A、B节点、Y标签的占位符，他们都是558维
A_holder = tf.placeholder(tf.float32, shape=(None, (NUM_NEIGHBOR * NUM_NEIGHBOR + NUM_NEIGHBOR + 1) * NUM_FEA),
                          name='a')
B_holder = tf.placeholder(tf.float32, shape=(None, (NUM_NEIGHBOR * NUM_NEIGHBOR + NUM_NEIGHBOR + 1) * NUM_FEA),
                          name='b')
y_holder = tf.placeholder(tf.float32, shape=(None, 1), name='y')

# 调用LP函数获取预测结果、损失、A_emb和B_emb
pred, loss, auc, auc_op, A_emb, B_emb, acc = LinkPrediction(A_holder, B_holder, y_holder)

op = tf.train.AdamOptimizer(0.001).minimize(loss)

init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()

plot_x = []
plot_y = []

with tf.Session() as sess:
    sess.run(init_op)
    sess.run(local_init_op)

    train_losses = []  # 记录训练集的损失值
    train_aucs = []  # 记录训练集的AUC值
    val_losses = []  # 记录验证集的损失值
    val_aucs = []  # 记录验证集的AUC值
    test_losses = []  # 记录测试集的损失值
    test_aucs = []  # 记录测试集的AUC值
    train_accs = []
    val_accs = []
    test_accs = []
    for ep in range(EPOCH):
        # train
        batch_A_fea, batch_B_fea, batch_y = batch_data(G_train, BATCH_SIZE).__next__()
        fetches = [A_emb, B_emb, op, pred, loss, auc_op, auc, acc]
        feed_dict = {
            A_holder: batch_A_fea,
            B_holder: batch_B_fea,
            y_holder: batch_y,
        }
        tra_A_emb, tra_B_emb, _, tra_pred, tra_loss, tra_auc_op, tra_auc, tra_acc = sess.run(fetches, feed_dict)
        print("Epoch:", ep, "train:", 'train_loss:', tra_loss, 'train_auc:', tra_auc, 'train_acc:', tra_acc)
        train_losses.append(tra_loss)
        train_aucs.append(tra_auc)
        train_accs.append(tra_acc)

        # val
        val_batch_A_fea, val_batch_B_fea, val_batch_y = batch_data(G_val, BATCH_SIZE).__next__()
        fetches = [A_emb, B_emb, pred, loss, auc_op, auc, acc]
        feed_dict = {
            A_holder: val_batch_A_fea,
            B_holder: val_batch_B_fea,
            y_holder: val_batch_y
        }
        val_A_emb, val_B_emb, val_pred, val_loss, val_auc_op, val_auc, val_acc = sess.run(fetches, feed_dict)
        print("Epoch:", ep, "val:", "val_loss:", val_loss, "val_auc:", val_auc, 'val_acc:', val_acc)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        val_accs.append(val_acc)

        # test
        test_batch_A_fea, test_batch_B_fea, test_batch_y = batch_data(G_test, BATCH_SIZE).__next__()
        fetches = [A_emb, B_emb, pred, loss, auc_op, auc, acc]
        feed_dict = {
            A_holder: test_batch_A_fea,
            B_holder: test_batch_B_fea,
            y_holder: test_batch_y
        }
        test_A_emb, test_B_emb, test_pred, test_loss, test_auc_op, test_auc, test_acc = sess.run(fetches, feed_dict)
        print("Epoch:", ep, "test:", "test_loss:", test_loss, "test_auc:", test_auc, "test_acc:", test_acc)
        test_losses.append(test_loss)
        test_aucs.append(test_auc)
        test_accs.append(test_acc)

import matplotlib.pyplot as plt

# 绘制图表
plt.figure()
plt.plot(range(EPOCH), train_losses, label='Train Loss')
plt.plot(range(EPOCH), val_losses, label='Validation Loss')
plt.plot(range(EPOCH), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs Epoch')
plt.show()

plt.figure()
plt.plot(range(EPOCH), train_aucs, label='Train AUC')
plt.plot(range(EPOCH), val_aucs, label='Validation AUC')
plt.plot(range(EPOCH), test_aucs, label='Test AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.title('AUC vs Epoch')
plt.show()
