import json
import networkx as nx
import random
import numpy as np
import tensorflow as tf
from itertools import combinations
from typing import Tuple, List, Dict, Any


class Config:
    def __init__(self):
        self.SEED = 0
        self.USE_DISTANCE_ENCODING = 1
        self.DATA_PATH = '**'
        self.NODE_TYPE_MAP = {'**': 0, '**': 1}
        self.NODE_TYPE_LIST = ['**', '**']
        
        self.EPOCHS = 2000
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 0.001
        self.WEIGHT_DECAY = 0.0
        
        self.NEIGHBOR_NUM = 5
        self.HOP_NUM = 2
        self.TEST_RATIO = 0.2
        
        self.EMBEDDING_DIM = 32
        self.FEATURE_DIM = self._calc_feature_dim()
        self.TOTAL_FEATURE_DIM = self._calc_total_feature_dim()

    def _calc_feature_dim(self) -> int:
        if self.USE_DISTANCE_ENCODING:
            return (8 + 2) * 2 + len(self.NODE_TYPE_MAP)
        return len(self.NODE_TYPE_MAP)

    def _calc_total_feature_dim(self) -> int:
        node_feat = self.FEATURE_DIM
        hop1_feat = self.FEATURE_DIM * self.NEIGHBOR_NUM
        hop2_feat = self.FEATURE_DIM * self.NEIGHBOR_NUM * self.NEIGHBOR_NUM
        return node_feat + hop1_feat + hop2_feat


class GraphDataLoader:
    def __init__(self, config: Config):
        self.config = config
        self._init_seed()

    def _init_seed(self) -> None:
        np.random.seed(self.config.SEED)
        random.seed(self.config.SEED)
        tf.set_random_seed(self.config.SEED)

    def load_and_split_data(self) -> Tuple[nx.Graph, nx.Graph, nx.Graph, int]:
        full_graph = nx.Graph()
        edge_list = []
        with open(self.config.DATA_PATH, 'r') as f:
            for line in f.readlines():
                nodes = line.strip().split(' ')
                full_graph.add_edge(nodes[0], nodes[1])
                edge_list.append(nodes)

        total_edges = len(edge_list)
        train_edge_cnt = int(total_edges * (1 - self.config.TEST_RATIO * 2))
        val_edge_cnt = int(total_edges * self.config.TEST_RATIO)

        train_graph = self._build_subgraph(edge_list[:train_edge_cnt])
        val_graph = self._build_subgraph(edge_list[train_edge_cnt:train_edge_cnt + val_edge_cnt])
        test_graph = self._build_subgraph(edge_list[train_edge_cnt + val_edge_cnt:])

        self._print_data_info(full_graph, train_graph, val_graph, test_graph)

        return train_graph, val_graph, test_graph, len(self.config.NODE_TYPE_MAP)

    @staticmethod
    def _build_subgraph(edges: List[List[str]]) -> nx.Graph:
        sub_graph = nx.Graph()
        sub_graph.add_edges_from(edges)
        return sub_graph

    @staticmethod
    def _print_data_info(full: nx.Graph, train: nx.Graph, val: nx.Graph, test: nx.Graph) -> None:
        print(f"Data split results:")


class FeatureEncoder:
    def __init__(self, config: Config):
        self.config = config

    def encode_node_type(self, node: str) -> List[float]:
        type_encoding = [0.0] * len(self.config.NODE_TYPE_MAP)
        type_encoding[self.config.NODE_TYPE_MAP[node[0]]] = 1.0
        return type_encoding

    def encode_distance(self, source: str, target: str, graph: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
        paths = list(nx.all_simple_paths(graph, source, target, cutoff=self.config.HOP_NUM + 2))
        sorted_paths = sorted(paths, key=len) if paths else []

        dist_array = np.array([0.0] * 8)
        status_array = np.array([0.0, 0.0])

        if sorted_paths:
            shortest_path = sorted_paths[0]
            if len(shortest_path) == 1:
                status_array = np.array([0.0, 0.0])
            elif len(shortest_path) > 5:
                status_array = np.array([1.0, 0.0])
                shortest_path = shortest_path[:5]
            else:
                status_array = np.array([0.0, 1.0])

            for i in range(len(shortest_path) - 1):
                next_node_type = shortest_path[i + 1][0]
                if next_node_type == self.config.NODE_TYPE_LIST[0]:
                    dist_array[2 * i] = 1.0
                else:
                    dist_array[2 * i + 1] = 1.0
        else:
            status_array = np.array([1.0, 1.0])

        return dist_array, status_array

    def gen_node_feature_dict(self, subgraph: nx.Graph, node_pair: List[str]) -> Dict[str, np.ndarray]:
        node_x, node_y = node_pair
        feature_dict = {}

        if self.config.USE_DISTANCE_ENCODING:
            for node in subgraph.nodes:
                dist_x, status_x = self.encode_distance(node_x, node, subgraph)
                dist_y, status_y = self.encode_distance(node_y, node, subgraph)
                feature_dict[node] = np.concatenate([dist_x, status_x, dist_y, status_y], axis=0)

        return feature_dict


class SubgraphSampler:
    def __init__(self, config: Config, encoder: FeatureEncoder):
        self.config = config
        self.encoder = encoder

    def sample_subgraph(self, node_pair: List[str], graph: nx.Graph) -> Tuple[nx.Graph, Dict[str, np.ndarray]]:
        node_x, node_y = node_pair
        ego_x = nx.ego_graph(graph, node_x, radius=self.config.HOP_NUM)
         ***
        combined_subgraph = nx.compose(ego_x, ego_y)
        combined_subgraph.remove_edges_from(combinations(node_pair, 2))
        feature_dict = self.encoder.gen_node_feature_dict(combined_subgraph, node_pair)
        return combined_subgraph, feature_dict

    def sample_neighbors(self, graph: nx.Graph, node: str) -> List[str]:
        neighbors = list(graph.neighbors(node)) + [node]
        return list(np.random.choice(neighbors, self.config.NEIGHBOR_NUM, replace=True))

    def gen_node_feature_batch(self, subgraph: nx.Graph, root_node: str, feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
        feature_list = []

        if self.config.USE_DISTANCE_ENCODING:
            root_feat = [0.0] * (8 + 2) * 2 + self.encoder.encode_node_type(root_node)
        else:
            root_feat = self.encoder.encode_node_type(root_node)
        feature_list.append(np.asarray(root_feat, dtype=np.float32).reshape(-1, self.config.FEATURE_DIM))

        hop1_nodes = self.sample_neighbors(subgraph, root_node)
        hop1_feats = self._gen_hop_features(hop1_nodes, feature_dict)
        feature_list.append(np.asarray(hop1_feats, dtype=np.float32).reshape(1, -1))

        hop2_nodes = [self.sample_neighbors(subgraph, n) for n in hop1_nodes]
        hop2_feats = []
        for nodes in hop2_nodes:
            hop2_feats.append(self._gen_hop_features(nodes, feature_dict))
        feature_list.append(np.asarray(hop2_feats, dtype=np.float32).reshape(1, -1))

        return np.concatenate(feature_list, axis=1)

    def _gen_hop_features(self, nodes: List[str], feature_dict: Dict[str, np.ndarray]) -> List[np.ndarray]:
        hop_feats = []
        for node in nodes:
            if self.config.USE_DISTANCE_ENCODING:
                node_feat = feature_dict[node]
                type_feat = np.asarray(self.encoder.encode_node_type(node))
                combined_feat = np.concatenate([node_feat, type_feat], axis=0)
                hop_feats.append(combined_feat)
            else:
                hop_feats.append(np.asarray(self.encoder.encode_node_type(node)))
        return hop_feats


class BatchGenerator:
    def __init__(self, config: Config, sampler: SubgraphSampler):
        self.config = config
        self.sampler = sampler

    def generate_batch(self, graph: nx.Graph) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        edges = list(graph.edges)
        nodes = list(graph.nodes)
        batch_edge_cnt = int(len(edges) / self.config.BATCH_SIZE)
        random.shuffle(edges)

        standard_type = list(self.config.NODE_TYPE_MAP.keys())[0]
        edges = [(x, y) if x.startswith(standard_type) else (y, x) for x, y in edges]

        for batch_idx in range(batch_edge_cnt):
            batch_edges = edges[batch_idx * self.config.BATCH_SIZE : (batch_idx + 1) * self.config.BATCH_SIZE]
            batch_feat_a, batch_feat_b, batch_labels = self._gen_batch_data(batch_edges, graph, nodes)
            yield batch_feat_a, batch_feat_b, batch_labels

    def _gen_batch_data(self, edges: List[Tuple[str, str]], graph: nx.Graph, nodes: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        feat_a_list = []
        feat_b_list = []
        label_list = []

        for edge in edges:
            node_x, node_y = edge
            subgraph, feat_dict = self.sampler.sample_subgraph([node_x, node_y], graph)
            feat_a = self.sampler.gen_node_feature_batch(subgraph, node_x, feat_dict)
            feat_b = self.sampler.gen_node_feature_batch(subgraph, node_y, feat_dict)
            feat_a_list.append(feat_a)
            feat_b_list.append(feat_b)
            label_list.append(np.asarray(1.0, dtype=np.float32))

            neg_node = random.choice(nodes)
            subgraph_neg, feat_dict_neg = self.sampler.sample_subgraph([node_x, neg_node], graph)
            feat_a_neg = self.sampler.gen_node_feature_batch(subgraph_neg, node_x, feat_dict_neg)
            feat_b_neg = self.sampler.gen_node_feature_batch(subgraph_neg, neg_node, feat_dict_neg)
            feat_a_list.append(feat_a_neg)
            feat_b_list.append(feat_b_neg)
            label_list.append(np.asarray(0.0, dtype=np.float32))

        return (
            np.asarray(np.squeeze(feat_a_list)),
            np.asarray(np.squeeze(feat_b_list)),
            np.asarray(label_list).reshape(-1, 1)
        )


class NSEGNNModel:
    def __init__(self, config: Config):
        self.config = config
        self.weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
        self.weight_reg = tf.contrib.layers.l2_regularizer(config.WEIGHT_DECAY)
        self._build_placeholders()
        self._build_model()

    def _build_placeholders(self) -> None:
        self.feat_a = tf.placeholder(
            tf.float32, 
            shape=(None, self.config.TOTAL_FEATURE_DIM), 
            name='input_feat_a'
        )
        self.feat_b = tf.placeholder(
            tf.float32, 
            shape=(None, self.config.TOTAL_FEATURE_DIM), 
            name='input_feat_b'
        )
        self.labels = tf.placeholder(
            tf.float32, 
            shape=(None, 1), 
            name='input_labels'
        )

    def _build_NSEGNN_encoder(self, input_feat: tf.Tensor) -> tf.Tensor:
        with tf.variable_scope('NSEGNN_Encoder', reuse=tf.AUTO_REUSE):
            root_feat = input_feat[:, :self.config.FEATURE_DIM]
            hop1_feat = tf.reshape(
                input_feat[:, self.config.FEATURE_DIM : self.config.FEATURE_DIM * (self.config.NEIGHBOR_NUM + 1)],
                [-1, self.config.NEIGHBOR_NUM, self.config.FEATURE_DIM]
            )
            hop2_feat = tf.reshape(
                input_feat[:, self.config.FEATURE_DIM * (self.config.NEIGHBOR_NUM + 1):],
                [-1, self.config.NEIGHBOR_NUM, self.config.NEIGHBOR_NUM, self.config.FEATURE_DIM]
            )

            hop2_agg = tf.reduce_mean(hop2_feat, axis=2)
            neighbor_combined = tf.concat([hop1_feat, hop2_agg], axis=2)
            neighbor_proj = tf.layers.dense(
                inputs=neighbor_combined,
                units=self.config.EMBEDDING_DIM,
                activation=tf.nn.elu,
                kernel_initializer=self.weight_init,
                kernel_regularizer=self.weight_reg,
                name='neighbor_projection'
            )

            neighbor_agg = tf.reduce_mean(neighbor_proj, axis=1)
            node_combined = tf.concat([root_feat, neighbor_agg], axis=1)

            embedding = tf.layers.dense(
                inputs=node_combined,
                units=self.config.EMBEDDING_DIM,
                activation=tf.nn.elu,
                kernel_initializer=self.weight_init,
                kernel_regularizer=self.weight_reg,
                name='embedding_proj_1'
            )

            embedding = tf.layers.dense(
                inputs=embedding,
                units=self.config.EMBEDDING_DIM,
                activation=tf.nn.elu,
                kernel_initializer=self.weight_init,
                kernel_regularizer=self.weight_reg,
                name='embedding_proj_2'
            )

            return embedding

    def _build_model(self) -> None:
        self.emb_a = self._build_NSEGNN_encoder(self.feat_a)
        self.emb_b = self._build_NSEGNN_encoder(self.feat_b)

        combined_emb = tf.concat([self.emb_a, self.emb_b], axis=1)
        
        hidden = tf.layers.dense(
            inputs=combined_emb,
            units=32,
            activation=tf.nn.elu,
            kernel_initializer=self.weight_init,
            kernel_regularizer=self.weight_reg,
            name='pred_hidden_1'
        )
        
        hidden = tf.layers.dense(
            inputs=hidden,
            units=16,
            activation=tf.nn.leaky_relu,
            kernel_initializer=self.weight_init,
            kernel_regularizer=self.weight_reg,
            name='pred_hidden_2'
        )
        
        self.logits = tf.layers.dense(
            inputs=hidden,
            units=1,
            activation=None,
            kernel_initializer=self.weight_init,
            kernel_regularizer=self.weight_reg,
            name='pred_output'
        )
        
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.logits
        ))
        
        self.auc, self.auc_op = tf.metrics.auc(
            labels=self.labels, predictions=tf.nn.sigmoid(self.logits)
        )
        
        self.acc, _ = tf.metrics.accuracy(
            labels=self.labels, predictions=tf.round(tf.nn.sigmoid(self.logits))
        )
        
        self.optimizer = tf.train.AdamOptimizer(self.config.LEARNING_RATE).minimize(self.loss)


class Trainer:
    def __init__(self, config: Config, model: NSEGNNModel, batch_gen: BatchGenerator, 
                 train_graph: nx.Graph, val_graph: nx.Graph, test_graph: nx.Graph):
        self.config = config
        self.model = model
        self.batch_gen = batch_gen
        self.train_graph = train_graph
        self.val_graph = val_graph
        self.test_graph = test_graph
        
        self.train_losses = []
        self.train_aucs = []
        self.train_accs = []
        self.val_losses = []
        self.val_aucs = []
        self.val_accs = []
        self.test_losses = []
        self.test_aucs = []
        self.test_accs = []

    def run(self) -> None:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            for epoch in range(self.config.EPOCHS):
                self._train_step(sess, epoch)
                self._eval_step(sess, epoch, 'val')
                self._eval_step(sess, epoch, 'test')

    def _train_step(self, sess: tf.Session, epoch: int) -> None:
        batch_a, batch_b, batch_labels = next(self.batch_gen.generate_batch(self.train_graph))
        
        feed_dict = {
            self.model.feat_a: batch_a,
            self.model.feat_b: batch_b,
            self.model.labels: batch_labels
        }
        
        _, loss_val, _, auc_val, acc_val = sess.run(
            [self.model.optimizer, self.model.loss, self.model.auc_op, self.model.auc, self.model.acc],
            feed_dict=feed_dict
        )
        
        self.train_losses.append(loss_val)
        self.train_aucs.append(auc_val)
        self.train_accs.append(acc_val)
        print(f"Epoch: {epoch} | Train - Loss: {loss_val:.4f}, AUC: {auc_val:.4f}, Acc: {acc_val:.4f}")

    def _eval_step(self, sess: tf.Session, epoch: int, mode: str) -> None:
        graph = self.val_graph if mode == 'val' else self.test_graph
        batch_a, batch_b, batch_labels = next(self.batch_gen.generate_batch(graph))
        
        feed_dict = {
            self.model.feat_a: batch_a,
            self.model.feat_b: batch_b,
            self.model.labels: batch_labels
        }
        
        loss_val, _, auc_val, acc_val = sess.run(
            [self.model.loss, self.model.auc_op, self.model.auc, self.model.acc],
            feed_dict=feed_dict
        )
        
        if mode == 'val':
            self.val_losses.append(loss_val)
            self.val_aucs.append(auc_val)
            self.val_accs.append(acc_val)
        else:
            self.test_losses.append(loss_val)
            self.test_aucs.append(auc_val)
            self.test_accs.append(acc_val)
            
        print(f"Epoch: {epoch} | {mode.capitalize()} - Loss: {loss_val:.4f}, AUC: {auc_val:.4f}, Acc: {acc_val:.4f}")


def main():
    config = Config()
    data_loader = GraphDataLoader(config)
    train_graph, val_graph, test_graph, _ = data_loader.load_and_split_data()
    
    encoder = FeatureEncoder(config)
    sampler = SubgraphSampler(config, encoder)
    batch_gen = BatchGenerator(config, sampler)
    
    model = NSEGNNModel(config)
    trainer = Trainer(config, model, batch_gen, train_graph, val_graph, test_graph)
    trainer.run()


if __name__ == "__main__":
    main()
