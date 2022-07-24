import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from scipy.sparse import csr_matrix
# imports TestbedDataset, rmse, mse, pearson, spearman, ci
from EmbedDTI.utils import *
from collections import defaultdict
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem import AllChem


# 处理SMILES序列，转化为原子的图结构
# Process SMILES sequences into graph structures of atoms
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + # 返回原子上隐式Hs的数量
                    one_of_k_encoding_unk(atom.GetTotalValence(),[0,1,2,3,4,5,6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetFormalCharge(),[0,1,2,3,4,5,6,7,8,9,10]) +
                    [atom.GetIsAromatic()] +
                    [atom.IsInRing()]
                    )

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()  # 原子数 Atomic number
    
    features = [] 
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    # 构建有向图 Building a directed graph
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index


# # 处理SMILES序列，转化为基团的图结构 Process SMILES sequences into graph structures of groups

MST_MAX_WEIGHT = 100 

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol)
    return mol

# 构建邻接矩阵和特征矩阵
# Build Adjacency Matrix and Feature Matrix
# the "idx" argument is not used in the cluster_graph function
# originally: def cluster_graph(mol,idx):
def cluster_graph(mol,idx=None):
	n_atoms = mol.GetNumAtoms()
	# if n_atoms == 1: #special case
 #    	return [[0]], []
	cliques = []
	for bond in mol.GetBonds():
		a1 = bond.GetBeginAtom().GetIdx()
		a2 = bond.GetEndAtom().GetIdx()
		if not bond.IsInRing():
			cliques.append([a1,a2])

	ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
	cliques.extend(ssr)

	# nei_list为原子属于哪个基团
	nei_list = [[] for i in range(n_atoms)]
	for i in range(len(cliques)):
	    for atom in cliques[i]:
	        nei_list[atom].append(i)

	#Merge Rings with intersection > 2 atoms
	for i in range(len(cliques)):
		if len(cliques[i]) <= 2: continue
		for atom in cliques[i]:
			for j in nei_list[atom]:
				if i >= j or len(cliques[j]) <= 2: 
					continue
				inter = set(cliques[i]) & set(cliques[j])
				if len(inter) > 2:
					cliques[i].extend(cliques[j])
					cliques[i] = list(set(cliques[i]))
					cliques[j] = []
    
	cliques = [c for c in cliques if len(c) > 0]
	nei_list = [[] for i in range(n_atoms)]
	for i in range(len(cliques)):
		for atom in cliques[i]:
			nei_list[atom].append(i)

	# Build edges and add singleton cliques
	edges = defaultdict(int)
	for atom in range(n_atoms):
		if len(nei_list[atom]) <= 1: 
			continue
		cnei = nei_list[atom]
		bonds = [c for c in cnei if len(cliques[c]) == 2]
		rings = [c for c in cnei if len(cliques[c]) > 4]
		
		for i in range(len(cnei)):
			for j in range(i + 1, len(cnei)):
				c1,c2 = cnei[i],cnei[j]
				inter = set(cliques[c1]) & set(cliques[c2])
				if edges[(c1,c2)] < len(inter):
					edges[(c1,c2)] = len(inter) # cnei[i] < cnei[j] by construction
					edges[(c2,c1)] = len(inter)

	edges = [u + (MST_MAX_WEIGHT-v,) for u,v in edges.items()]
	row,col,data = zip(*edges)
	data = list(data)
	for i in range(len(data)):
		data[i] = 1
	data = tuple(data)
	n_clique = len(cliques)
	clique_graph = csr_matrix( (data,(row,col)), shape=(n_clique,n_clique) )
	edges = [[row[i],col[i]] for i in range(len(row))]
	
	return cliques, edges



def clique_features(clique,edges,clique_idx,smile):
    NumAtoms = len(clique) # 基团中除去氢原子的原子数
    NumEdges = 0  # 与基团所连的边数
    for edge in edges:
        if clique_idx == edge[0] or clique_idx == edge[1]:
            NumEdges += 1
    mol = Chem.MolFromSmiles(smile)
    NumHs = 0 # 基团中氢原子的个数
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() in clique:
            NumHs += atom.GetTotalNumHs()
    # 基团中是否包含环
    IsRing = 0
    if len(clique) > 2:
        IsRing = 1
    # 基团中是否有键
    IsBond = 0
    if len(clique) == 2:
        IsBond = 1

    return np.array(one_of_k_encoding_unk(NumAtoms,[0,1,2,3,4,5,6,7,8,9,10]) + 
        one_of_k_encoding_unk(NumEdges,[0,1,2,3,4,5,6,7,8,9,10]) + 
        one_of_k_encoding_unk(NumHs,[0,1,2,3,4,5,6,7,8,9,10]) + 
        [IsRing] + 
        [IsBond]
        )

"""# Call embedDTI_smile_preprocess(...) to get the atom graph and substructures graph"""

def embedDTI_smile_preprocess(smile_to_process, test_MolToSmiles):
    # atom representation
    if (test_MolToSmiles):
        # do we need to put SMILES into canonical representation or not?
        smile_to_process = Chem.MolToSmiles(Chem.MolFromSmiles(smile_to_process),isomericSmiles=True)
    atom_result = smile_to_graph(smile_to_process)

    # substructure representation
    our_mol = get_mol(smile_to_process)
    our_clique, our_edge = cluster_graph(our_mol)
    c_features = []
    for idx in range(len(our_clique)):
        # TODO check if it has to be canonical SMILES or not
        cq_features = clique_features(our_clique[idx],our_edge,idx,smile_to_process)
        c_features.append( cq_features / sum(cq_features) )
    clique_size = len(our_clique)
    graph = (clique_size, c_features, our_edge)

    return atom_result, graph