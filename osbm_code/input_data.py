import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import scipy.io as sio

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset):

    if dataset == 'gen_z':
        adj, features, feature_presence = create_adj_z()
        return adj, features, feature_presence
    elif dataset.startswith('nips12'):
        adj, features, feature_presence = load_nips_mat(dataset)
        return adj, features, feature_presence
    
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]))))
    x, tx, allx, graph = tuple(objects)

    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil() # convert to linked list
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features, 1
    
def load_nips_mat(dataset):

    mat_data = sio.loadmat('data/nips12.mat')
    # print mat_data.keys()

    adj = mat_data['B']
    
    adj = sp.csr_matrix(adj)
    features = sp.identity((adj.shape[0]))

    return adj, features, 0    

def create_adj_z(n = 100, z = 10):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    assert n % z == 0, 'Num Communities should be a factor of N'

    nz = np.zeros((n,z))

    nz[0:20, 0] = 1
    nz[20:35, 1] = 1
    nz[35:50, 2] = 1
    nz[50:65, 3] = 1
    nz[65:75, 4] = 1
    nz[75:85, 5] = 1
    nz[85:90, 6] = 1
    nz[90:95, 7] = 1
    nz[90:95, 9] = 1
    nz[95:100, 8] = 1

    # overlap
    nz[25:60, 7] = 1
    nz[60:80, 8] = 1
    
    W = np.eye(z)
    adj = sigmoid(np.matmul(np.matmul(nz, W), np.transpose(nz)))
    adj = np.round(adj)
    
    adj = sp.csr_matrix(adj)
    features = sp.identity((adj.shape[0]))

    return adj, features, 0

def load_data_split(dataset_str, split_idx):
    
    data_path = 'data/all_edge_idx_' + dataset_str + '.npy'
    all_edge_idx_array = np.load(data_path)

    return all_edge_idx_array[split_idx]

# not being used
def load_masked_test_edges_for_kfold(dataset_str, k_fold=5, split_idx=0):

    data_path = 'data/' + dataset_str + '/' + str(k_fold) + '-fold/split_' + str(split_idx) + '.npz'
    data = np.load(data_path)

    return data['k_adj_train'], data['k_train_edges'], data['k_val_edges'], data['k_val_edges_false'], data['test_edges'], data['test_edges_false']

def load_masked_test_edges(dataset_str, split_idx=0):

    data_path = 'data/' + dataset_str + '/split_' + str(split_idx) + '.npz'
    data = np.load(data_path)
    
    return data['adj_train'], data['train_edges'], data['val_edges'], data['val_edges_false'], data['test_edges'], data['test_edges_false']
    
