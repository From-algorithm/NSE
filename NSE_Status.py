import json

import networkx as nx
import random
import numpy as np
import tensorflow as tf
from itertools import combinations

seed_id = 0
np.random.seed(seed_id)  
USE_DE = 1
PATH = 'DataBase/PubMed_Mini/DG.txt'
type2idx = {'D': 0, 'G': 1}
type_list = ['D', 'G']
EPOCH = 400
NUM_NEIGHBOR = 5
K_HOP = 2 
BATCH_SIZE = 32
EMB_DIM = 32


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
    print( str(num_edge) + str(sp1) + str(sp3)  + str(sp2))
    G_train = nx.Graph()
    G_val = nx.Graph()
    G_test = nx.Graph()

    G_train.add_edges_from(edge_list[:sp1])
    G_val.add_edges_from(edge_list[sp1:sp1 + sp2])
    G_test.add_edges_from(edge_list[sp1 + sp2:])
    print(
        f"all edge: {len(G.edges)}, train edge: {len(G_train.edges)}, val edge: {len(G_val.edges)}, test edge: {len(G_test.edges)}")
    return G_train, G_val, G_test, NODE_TYPE, type2idx


G_train, G_val, G_test, NODE_TYPE, type2idx = load_dataset()

mini_batch = []
fea_batch = []

if USE_DE:
    NUM_FEA = (8 + 2) * 2 + NODE_TYPE
else:
    NUM_FEA = NODE_TYPE

initializer = tf.contrib.layers.xavier_initializer(uniform=False)

regularizer = tf.contrib.layers.l2_regularizer(0.0)


def dist_encoder(src, dest, G, K_HOP):
    paths = list(nx.all_simple_paths(G, src, dest, cutoff=K_HOP + 2))
    sorted_paths = sorted(paths, key=len)
    array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    if sorted_paths:
        if len(sorted_paths[0]) == 1:
            status_encoder = np.array([0.0, 0.0])
        elif len(sorted_paths[0]) > 5:
            status_encoder = np.array([1.0, 0.0])
            sorted_paths[0] = sorted_paths[0][:5]
        else:
            status_encoder = np.array([0.0, 1.0])

        for i in range(len(sorted_paths[0]) - 1):
            if sorted_paths[0][i + 1][0] == type_list[0]:
                array[2 * i] = 1.0
                array[2 * i + 1] = 0.0
            else:
                array[2 * i] = 0.0
                array[2 * i + 1] = 1.0
    else:
        status_encoder = np.array([1.0, 1.0])
    return array, status_encoder


def type_encoder(node):
    res = [0.0] * NODE_TYPE
    res[type2idx[node[0]]] = 1.0
    return res


def gen_fea_batch1(G, root, des, fea_dict, K_HOP):
    fea_batch = []
    mini_batch.append([root])
    if USE_DE:
        a = [0.0] * (8 + 2) * 2 + type_encoder(root)
    else:
        a = type_encoder(root)
    fea_batch.append(np.asarray(a, dtype=np.float32).reshape(-1, NUM_FEA))

    # 1-ord-
    ns_1 = []
    for node in mini_batch[-1]:
        neighbors = list(G.neighbors(node)) + [node]
        chosen_neighbors = np.random.choice(neighbors, NUM_NEIGHBOR, replace=True)
        ns_1.append(list(chosen_neighbors))
    mini_batch.append(ns_1[0])

    de_1 = []
    if USE_DE:
        for dest in ns_1[0]:
            dest_features = fea_dict[dest]
            dest_type_encoding = np.asarray(type_encoder(dest))
            dest_combined = np.concatenate([dest_features, dest_type_encoding], axis=0)
            de_1.append(dest_combined)
    else:
        for dest in ns_1[0]:
            dest_type_encoding = np.asarray(type_encoder(dest))
            de_1.append(dest_type_encoding)

    fea_batch.append(np.asarray(de_1, dtype=np.float32).reshape(1, -1))

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
    if USE_DE:
        a = [0.0] * (8 + 2) * 2 + type_encoder(root)
    else:
        a = type_encoder(root)
    fea_batch.append(np.asarray(a, dtype=np.float32).reshape(-1, NUM_FEA))


    ns_1 = []
    for node in mini_batch[-1]:
        neighbors = list(G.neighbors(node)) + [node]
        chosen_neighbors = np.random.choice(neighbors, NUM_NEIGHBOR, replace=True)
        ns_1.append(list(chosen_neighbors))
    mini_batch.append(ns_1[0])
    de_1 = []
    if USE_DE:
        for dest in ns_1[0]:
            dest_features = fea_dict[dest]
            dest_type_encoding = np.asarray(type_encoder(dest))
            dest_combined = np.concatenate([dest_features, dest_type_encoding], axis=0)
            de_1.append(dest_combined)
    else:
        for dest in ns_1[0]:
            dest_type_encoding = np.asarray(type_encoder(dest))
            de_1.append(dest_type_encoding)
    fea_batch.append(np.asarray(de_1, dtype=np.float32).reshape(1, -1))


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



def subgraph_sampling_with_DE_node_pair(G, node_pair, K_HOP=2):
    # print('edge_DE .... ')
    [A, B] = node_pair

    A_ego = nx.ego_graph(G, A, radius=K_HOP)

    B_ego = nx.ego_graph(G, B, radius=K_HOP)

    sub_G_for_AB = nx.compose(A_ego, B_ego)

    sub_G_for_AB.remove_edges_from(combinations(node_pair, 2))

    sub_G_nodes = sub_G_for_AB.nodes



  
    SPD_based_on_node_pair = {}
    if USE_DE:
        for node in sub_G_nodes:
            tmpA, tmpA_status = dist_encoder(A, node, sub_G_for_AB, K_HOP)
            tmpB, tmpB_status = dist_encoder(B, node, sub_G_for_AB, K_HOP)
            SPD_based_on_node_pair[node] = np.concatenate([tmpA, tmpA_status, tmpB, tmpB_status], axis=0)

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
    edge = [(x, y) if x.startswith(list(type2idx.keys())[0]) else (y, x) for x, y in edge]
    for idx in range(num_batch):
        # TODO add shuffle and random sample
        batch_edge = edge[idx * batch_size:(idx + 1) * batch_size]
        batch_label = [1.0] * batch_size
        # label[idx * batch_size:(idx + 1) * batch_size]

        batch_A_fea = []
        batch_B_fea = []
        batch_y = []


        for (bx, by) in zip(batch_edge, batch_label):
            # print(bx, by)

            posA, posB = subgraph_sampling_with_DE_node_pair(G, bx, K_HOP=K_HOP)
            batch_A_fea.append(posA)
            batch_B_fea.append(posB)
            batch_y.append(np.asarray(by, dtype=np.float32))


            neg_tmpB_id = random.choice(nodes)
            negA, negB = subgraph_sampling_with_DE_node_pair(G, [bx[0], neg_tmpB_id], K_HOP=K_HOP)
            batch_A_fea.append(negA)
            batch_B_fea.append(negB)
            batch_y.append(np.asarray(0.0, dtype=np.float32))

        yield np.asarray(np.squeeze(batch_A_fea)), np.asarray(np.squeeze(batch_B_fea)), np.asarray(batch_y).reshape(
            batch_size * 2, 1)


def split(G, split=0.8):
    edge_list = list(G.edges)
    num_edge = len(edge_list)
    sp = int(num_edge * split)
    train_edge = edge_list[:sp]
    train_label = [1.0] * sp 

    test_edge = edge_list[sp:]
    test_label = [1.0] * (num_edge - sp)  
    return train_edge, train_label, test_edge, test_label


def ESGNN(fea, model='meirec'):
    """
    :param fea: fea_batch, [[0, 0, 4], [[0, 1, 1], [0, 1, 4], [0, 1, 4], [0, 1, 1], [1, 0, 1]]]
    :return:
    """
    with tf.variable_scope(name_or_scope='ESGNN', reuse=tf.AUTO_REUSE):

        node = fea[:, :NUM_FEA]
        neigh1 = fea[:, NUM_FEA:NUM_FEA * (NUM_NEIGHBOR + 1)]
        neigh1 = tf.reshape(neigh1, [-1, NUM_NEIGHBOR, NUM_FEA])
        neigh2 = fea[:, NUM_FEA * (NUM_NEIGHBOR + 1):]
        neigh2 = tf.reshape(neigh2, [-1, NUM_NEIGHBOR, NUM_NEIGHBOR, NUM_FEA])
        if model == 'meirec':

            neigh2_agg = tf.reduce_mean(neigh2, axis=2)
            tmp = tf.concat([neigh1, neigh2_agg], axis=2)
            tmp = tf.layers.dense(tmp, EMB_DIM,
                                  activation=tf.nn.elu,
                                  use_bias=True,
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer,
                                  name='tmp_proj'
                                  )
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


def LinkPrediction(n1, n2, label):
    n1_emb = ESGNN(n1)
    n2_emb = ESGNN(n2)

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
                           # leaky_relu == 0.4
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

    _, accuracy = tf.metrics.accuracy(labels=label, predictions=tf.round(tf.nn.sigmoid(pred)))

    return pred, loss, auc, auc_op, n1_emb, n2_emb, accuracy



A_holder = tf.placeholder(tf.float32, shape=(None, (NUM_NEIGHBOR * NUM_NEIGHBOR + NUM_NEIGHBOR + 1) * NUM_FEA),
                          name='a')
B_holder = tf.placeholder(tf.float32, shape=(None, (NUM_NEIGHBOR * NUM_NEIGHBOR + NUM_NEIGHBOR + 1) * NUM_FEA),
                          name='b')
y_holder = tf.placeholder(tf.float32, shape=(None, 1), name='y')



pred, loss, auc, auc_op, A_emb, B_emb, acc = LinkPrediction(A_holder, B_holder, y_holder)

op = tf.train.AdamOptimizer(0.001).minimize(loss)

init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()

plot_x = []
plot_y = []

with tf.Session() as sess:
    sess.run(init_op)
    sess.run(local_init_op)

    train_losses = []  
    train_aucs = [] 
    val_losses = [] 
    val_aucs = []  
    test_losses = [] 
    test_aucs = [] 
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


