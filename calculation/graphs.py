#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""Class to manage network graphs."""

__author__ = "Inga Franzen"
__created__ = "2022"
__date_modified__ = "2023/05/31"
__version__ = "1.0"

import random
from abc import ABCMeta, abstractmethod

import igraph as ig
import networkx as nx
import numpy as np
from line_profiler_pycharm import profile
from numpy import random as rm


class Graph(metaclass=ABCMeta):
    @abstractmethod
    def extract_agents_neighbours(self, agents_id: list) -> list:
        pass


class GraphSmall(Graph):
    """
    Graph - No Graph - only a list of neighbours
    """

    # @profile
    def __init__(self, my_agents, mean_contacts, net_id):
        self.id = net_id
        self.node_num = len(my_agents)
        self.mean_contacts = min(mean_contacts, self.node_num - 1)
        self.members_list = my_agents

    # @profile
    def extract_agents_neighbours(self, agents_id):
        nbr_ids = self.members_list.copy()
        try:
            nbr_ids.remove(agents_id)
        except ValueError:
            print(
                f"In {type(self).__name__}.extract_agents_neighbours: agent"
                f" {agents_id} not in graph {self.id}"
            )
            return []
        return nbr_ids


class GraphMedium(Graph):
    """
    Graph - Node label = Index, Member = Node attribute
    """

    # @profile
    def __init__(self, my_agents, mean_contacts, net_id, seed):
        self.seed = seed
        self.id = net_id
        self.node_num = len(my_agents)
        self.mean_contacts = min(mean_contacts, self.node_num - 1)
        self.mbr_idx = dict(zip(my_agents, np.arange(self.node_num)))
        self.members_graph = self._initialize_graph(my_agents)
        assert abs(self._get_edge_target() - len(self.members_graph.es)) < 1

    def _initialize_graph(self, my_agents):
        m = self._get_edge_target()
        random.seed(self.seed.integers(1, 100))
        G = ig.Graph.Erdos_Renyi(n=self.node_num, m=m)
        G.vs["id"] = my_agents
        return G

    def _get_edge_target(self):
        return int(round(self.mean_contacts * self.node_num / 2))

    # @profile
    # only meant for unlocalized networks
    def shuffle_member(self):
        m = self._get_edge_target()
        random.seed(self.seed.integers(1, 100))
        self.members_graph = ig.Graph.Erdos_Renyi(n=self.node_num, m=m)
        my_agents = list(self.mbr_idx.keys())
        self.members_graph.vs["id"] = my_agents

    # @profile
    def extract_agents_neighbours(self, agents_id):
        try:
            idx = self.mbr_idx[agents_id]
        except KeyError:
            print(
                f"In {type(self).__name__}.extract_agents_neighbours: agent"
                f" {agents_id} not in graph {self.id}"
            )
            return []
        nbr = self.members_graph.neighbors(idx)
        nbr_ids = self.members_graph.vs[nbr]["id"]
        return nbr_ids


class GraphLarge(Graph):
    """
    Graph - Node label = Index, Member = Node attribute
    """

    edge_count = 0
    num_edges = 0
    randint = 1
    count = 0
    num_graphs = 0

    # @profile

    def __init__(self, my_agents, mean_contacts, net_id, seed):
        self.seed = seed
        self.id = net_id
        self.node_num = len(my_agents)
        self.mean_contacts = min(mean_contacts, self.node_num - 1)
        self.mbr_idx = dict(zip(my_agents, np.arange(self.node_num)))
        self.idx_mbr = dict(zip(np.arange(self.node_num), my_agents))
        self.members_graph = self._initialize_graph()

    # @profile
    def _initialize_graph(self):
        p = self._calculate_prob()
        G = nx.fast_gnp_random_graph(self.node_num, p, self.seed)
        nx.set_node_attributes(G, self.idx_mbr, "id")
        assert len(self.idx_mbr) == len(self.mbr_idx)
        return G

    def _calculate_prob(self):
        # one node can happen at overlap of interventions (e.g. Home Office)
        if self.node_num < 2:
            return 0
        else:
            # max possible num of edges --> Gaussian formular
            edge_max = self._get_edge_max()
            edge_target = self._get_edge_target()
            return edge_target / edge_max

    def _get_edge_max(self):
        return (self.node_num**2 - self.node_num) / 2

    def _get_edge_target(self):
        return int(round(self.mean_contacts * self.node_num / 2))

    @staticmethod
    def _get_new_randint():
        GraphLarge.randint += 10
        return GraphLarge.randint

    # @profile
    def remove_member_from_graph(self, agents_id):
        try:
            # drop agent and remember idx
            idx = self.mbr_idx.pop(agents_id)
            # drop last idx and remember agent on last idx
            agent_last = self.idx_mbr.pop(self.node_num - 1)
            # swap last agent to free idx
            if agent_last != agents_id:
                self.mbr_idx[agent_last] = idx
                self.idx_mbr[idx] = agent_last
                self.members_graph.nodes[idx]["id"] = agent_last
            self.members_graph.remove_node(self.node_num - 1)
            self.node_num -= 1
        except KeyError:
            print(f"agent {agents_id} not in graph {self.id}")
        assert len(self.idx_mbr) == len(self.mbr_idx)
        assert agents_id not in self.mbr_idx

    # @profile
    def add_member_to_graph(self, agents_id):
        if agents_id in self.mbr_idx:
            print(f"agent {agents_id} already in graph {self.id}")
            return
        # reorganize indices
        self.node_num += 1
        # add agent to graph
        idx = self.node_num - 1
        self.mbr_idx[agents_id] = idx
        self.idx_mbr[idx] = agents_id
        self.members_graph.add_node(idx)
        self.members_graph.nodes[idx]["id"] = agents_id
        # add one contact
        if self.node_num > 2:
            index = GraphLarge._get_new_randint() % (idx - 1)
        # node_num = 2 --> idx = 1 --> connect to 0
        elif self.node_num == 2:
            index = 0
        else:
            return
        self.members_graph.add_edge(index, idx)
        assert len(self.idx_mbr) == len(self.mbr_idx)
        assert agents_id in self.mbr_idx

    # @profile
    def extract_agents_neighbours(self, agents_id):
        try:
            idx = self.mbr_idx[agents_id]
        except KeyError:
            print(f"agent {agents_id} not in graph {self.id}")
            return []
        nbr = list(self.members_graph.adj[idx])
        nbr_ids = [self.idx_mbr[x] for x in nbr]
        return nbr_ids
