#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 21:27:14 2017

@author: klanna
"""

import numpy as np
import networkx as nx
import graphviz as gv
import matplotlib.pyplot as plt


class causal_graph(object):
    # defines causal structure of species    
    def __init__(self, species_names, causal = None, stoich = None, blocks = None, save_to_file = None, title_name = 'causal_graph'):
        self.n_nodes = len(species_names)
        self.nodes_names = species_names        
        self.file_name = save_to_file
        self.title_name = title_name
        
        if blocks == None:
            self.blocks = [1]*self.n_nodes
        else:
            self.blocks = blocks
        
        if stoich is None:            
            if causal is None:
                self.causal = np.ones(self.n_nodes) - np.eye(self.n_nodes)
            else:
                self.causal = causal
            self.stoich_from_causal()
        else:
            self.stoich = stoich
            self.causal_from_stoich()
        
        self.num_reactions = len(self.stoich.transpose())
        self.num_causal_edges = int(sum(sum(self.causal)))
        
        print('num_reactions = ', self.num_reactions, '\nnum_causal_edges = ', self.num_causal_edges)
        self.plot_causal()
        
    def causal_from_stoich(self):                
        # create causal adjacency matrix from the stoichiometry
        S = self.stoich
        adj_mat = np.zeros([self.n_nodes, self.n_nodes])
        for re, stoich_vec in enumerate(S.transpose()):
            parents = np.where(stoich_vec == -1)[0]
            children = np.where(stoich_vec == 1)[0]
            for p in parents:
                for c in children:
                    adj_mat[p, c] = 1
                for p2 in [i for i in parents if i != p]:
                    adj_mat[p, p2] = 1
        self.causal = adj_mat
        
    def stoich_from_causal(self):
        # create possible stoichiometry matrix from all possible causal combination of parents    
        adj_mat = np.transpose(self.causal)
        self.stoich = np.empty([self.n_nodes, 0])
        for child, vec in enumerate(adj_mat):
            # find parents
            parents = np.where(vec > 0)[0]
            for pa_1 in parents:
#                stoich_vector = np.zeros(shape=(self.n_nodes,1))
#                stoich_vector[child] = 1
#                stoich_vector[pa_1] = -1            
#                self.stoich = np.append(self.stoich, stoich_vector, axis=1)
                
                for pa_2 in [j for j in parents if j > pa_1]:
                    stoich_vector = np.zeros(shape=(self.n_nodes,1))
                    stoich_vector[child] = 1
                    if adj_mat[pa_1, pa_2] and adj_mat[pa_2, pa_1]:
                        stoich_vector[pa_1] = -1            
                        stoich_vector[pa_2] = -1
                        self.stoich = np.append(self.stoich, stoich_vector, axis=1)
                    
        for pa, vec in enumerate(self.causal):
           # find children
            children = np.where(vec > 0)[0]
            for ch_1 in children:                
                for ch_2 in [j for j in children if j > ch_1]:
                    stoich_vector = np.zeros(shape=(self.n_nodes,1))
                    stoich_vector[pa] = -1
                    stoich_vector[ch_1] = 1            
                    stoich_vector[ch_2] = 1
                    self.stoich = np.append(self.stoich, stoich_vector, axis=1)
                        
    def plot_causal(self):
        Gref=nx.DiGraph(self.causal, forcelabels='true')
#        Gref.nodes() = self.nodes_names
        d = {}
        for i, s in enumerate(self.nodes_names):
            d[i] = s
        plt.figure(figsize=(8, 8))
        H=nx.relabel_nodes(Gref,d)
        nx.draw_networkx(H, arrows = True, with_labels=True, fontsize = 12)
        plt.title(self.title_name)
        plt.axis('off')
        if self.file_name:            
            plt.savefig(self.file_name + self.title_name + '.pdf')
        plt.show()
            
#        nodecolorsmap = ['#bebada', '#8dd3c7', '#ffffb3', '#fb8072', '#80b1d3', '#fdb462']
#        tpclr = 'black'
#        G = gv.Digraph()
#
#        if len(self.blocks) != 0:
#            for i in range(0, self.n_nodes):
#                G.node(self.nodenames[i, 0], color = nodecolorsmap[self.blocks[0, i]-1], style='filled')
#        w = 4
#        for i in range(self.n_nodes):
#            for j in range(self.n_nodes):
#                if (self.ntype[i, j] == 3) & (i != j):
#                    #     prior knowledge
#                    G.edge(self.nodenames[i, 0], self.nodenames[j, 0], color='#bdbdbd')
#                if (self.ntype[i, j] == 2) & (i != j):
#                    #     prior knowledge
#                    G.edge(self.nodenames[i, 0], self.nodenames[j, 0], color='black', penwidth=str(abs(self.adj[i, j])))
#                if (self.ntype[i, j] == 1) & (i != j):
#                    # true positive
#                    G.edge(str(self.nodenames[i, 0]), self.nodenames[j, 0], color=tpclr, penwidth=str(w*abs(self.adj[i, j])))
#                if (self.ntype[i, j] == -2) & (i != j):
#                    # false positive hard
#                    G.edge(self.nodenames[i, 0], self.nodenames[j, 0], color='blue', penwidth=str(w*abs(self.adj[i, j])))
#                if  (self.ntype[i, j] == -3) & (i != j):
#                    # false positive soft
#                    G.edge(self.nodenames[i, 0], self.nodenames[j, 0], color='green', penwidth=str(w*abs(self.adj[i, j])))
#                if (self.ntype[i, j] == -1) & (i != j):
#                    # false negative
#                    G.edge(self.nodenames[i, 0], self.nodenames[j, 0], color='pink')
#
#        if len(self.blocks) == 0:

    
                
