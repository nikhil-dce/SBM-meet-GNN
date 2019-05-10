import numpy as np
import scipy.sparse as sp


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

# Calculate result = D^(-1/2).A.D^(-1/2)
# Return coords, value and shape for coo matrix of result
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj, features, placeholders):

    # adj_normalized, adj and features are tuple with coords, value, and shape. From coo matrix
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


def mask_test_edges(adj, all_edge_idx=None, test_precent=10., val_precent=5.):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    #  coords, value, shape
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]

    print ('Unique Edges: ' + str(edges.shape[0]))

    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] * test_precent / 100.))
    num_val = int(np.floor(edges.shape[0] *val_precent / 100.)) 

    if all_edge_idx is None:
        all_edge_idx = range(edges.shape[0])
        np.random.shuffle(all_edge_idx)
        
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    # Checks if there is a relationship b/w nodes in `a` stored in `b` 
    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return (np.all(np.any(rows_close, axis=-1), axis=-1) and
                np.all(np.any(rows_close, axis=0), axis=0))

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false: # already included
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def mask_train_edges(adj):
    
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    
    adj_compl = 1 - adj.todense()
    adj_compl = sp.csr_matrix(adj_compl)
    adj_compl.eliminate_zeros()
    non_edges = sparse_to_tuple(adj_compl)[0]

    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    #adj_triu = sp.triu(adj)
    train_edges = sparse_to_tuple(sp.triu(adj))[0]
    edges_all = sparse_to_tuple(adj)[0]
    train_edges_false = [] 

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return (np.all(np.any(rows_close, axis=-1), axis=-1) and
                np.all(np.any(rows_close, axis=0), axis=0))
    

    #shuffle(non_edges)
    #train_edges_false = non_edges[:len(train_edges)]
    
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        train_edges_false.append([idx_i, idx_j])
    
    # NOTE: these edge lists only contain single direction of edge!
    return adj, train_edges, train_edges_false

def mask_test_edges_for_kfold (adj, k=4, all_edge_idx = None):
    # Function to build test set with 20% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.
    
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    #  coords, value, shape
    adj_tuple = sparse_to_tuple(adj_triu)

    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]

    number_of_edges = edges.shape[0]
    num_test = int(np.floor(number_of_edges / 5.)) 
    num_val = int(np.floor( (number_of_edges - num_test) / k))

    if all_edge_idx is None:
        all_edge_idx = range(number_of_edges)
        np.random.shuffle(all_edge_idx)

    k_val_edges = []
    k_train_edges = []
    k_val_edges_false = []
    
    test_edge_idx = all_edge_idx[k*num_val:]
    test_edges = edges[test_edge_idx]
    test_edges_false = []

    # Checks if there is a relationship b/w nodes in `a` stored in `b` 
    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return (np.all(np.any(rows_close, axis=-1), axis=-1) and
                np.all(np.any(rows_close, axis=0), axis=0))

    for k_idx in range(k):

        val_edge_idx_start = k_idx*num_val
        val_edge_idx_end = (k_idx+1)*num_val
        
        val_edge_idx = all_edge_idx[val_edge_idx_start:val_edge_idx_end]
        train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
        
        k_val_edges.append(edges[val_edge_idx])
        k_train_edges.append(train_edges)

    
    while len(test_edges_false) < len(test_edges):

        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])

        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue

        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
                
        test_edges_false.append([idx_i, idx_j])

    for k_idx in range(k):
        val_edges_false = []

        while len(val_edges_false) < len(k_val_edges[k_idx]):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], k_train_edges[k_idx]):
                continue
            if ismember([idx_j, idx_i], k_train_edges[k_idx]):
                continue
            if ismember([idx_i, idx_j], k_val_edges[k_idx]):
                continue
            if ismember([idx_j, idx_i], k_val_edges[k_idx]):
                continue
            if len(k_val_edges_false) > k_idx:
                if ismember([idx_j, idx_i], np.array(k_val_edges_false[k_idx])):
                    continue
                if ismember([idx_i, idx_j], np.array(k_val_edges_false[k_idx])):
                    continue
            val_edges_false.append([idx_i, idx_j])

        k_val_edges_false.append(val_edges_false)

        # Sanity Checks
        assert ~ismember(k_val_edges_false[k_idx], edges_all)
        assert ~ismember(k_val_edges_false[k_idx], k_train_edges[k_idx])
        assert ~ismember(test_edges, k_train_edges[k_idx])
        assert ~ismember(k_val_edges[k_idx], k_train_edges[k_idx])
        assert ~ismember(k_val_edges[k_idx], test_edges)

    assert ~ismember(test_edges_false, edges_all)

    k_adj_train = []
    # create adj_train for k-fold setting
    for k_idx in range(k):

        data = np.ones(k_train_edges[k_idx].shape[0])
        train_edges = k_train_edges[k_idx]

        # Re-build train adj matrix
        adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        adj_train = adj_train + adj_train.T
        k_adj_train.append(adj_train)

    # NOTE: these edge lists only contain single direction of edge!
    return k_adj_train, k_train_edges, k_val_edges, k_val_edges_false, test_edges, test_edges_false

def mask_test_edges_randomly (adj, k=5, all_edge_idx=None):
    # Function to build test set with 20% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.
    
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    #  coords, value, shape
    adj_tuple = sparse_to_tuple(adj_triu)

    edges = adj_tuple[0] # lets work
    
    edges_all = sparse_to_tuple(adj)[0]

    number_of_edges = edges.shape[0]

    num_nodes = adj_tuple[2][0]
    total_pairs = num_nodes*num_nodes
    test_data_size = int(np.floor(total_pairs/5.))

    # Checks if there is a relationship b/w nodes in `a` stored in `b` 
    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return (np.all(np.any(rows_close, axis=-1), axis=-1) and
                np.all(np.any(rows_close, axis=0), axis=0))


    k_val_edges = []
    k_train_edges = []
    k_val_edges_false = []
    
    test_cases = []
    test_edges = []
    
    while len(test_cases) < test_data_size:

        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])

        if idx_i == idx_j:
            continue

        test_cases.append([idx_i, idx_j])
        if ismember([idx_i, idx_j], edges_all):
            test_edges.append(1)
        else:
            test_edges.append(0)
            
    for k_idx in range(k):

        val_edge_idx_start = k_idx*num_val + test_data_size
        val_edge_idx_end = (k_idx+1)*num_val + test_data_size
        
        val_edge_idx = all_edge_idx[val_edge_idx_start:val_edge_idx_end]
        train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
        
        k_val_edges.append(edges[val_edge_idx])
        k_train_edges.append(train_edges)

    

    #-----------------------
    
    all_pairs_order = range(total_pairs)
    np.random.shuffle(all_pairs_order)
    all_pairs_order[0:test_data_size]
    sys.exit()

    


    
    num_tes = int(np.floor(adj_tuple[2]))
    num_test = int(np.floor(number_of_edges / 5.)) 
    num_val = int(np.floor( (number_of_edges - num_test) / k))

    if all_edge_idx is None:
        all_edge_idx = range(number_of_edges)
        np.random.shuffle(all_edge_idx)

    k_val_edges = []
    k_train_edges = []
    k_val_edges_false = []
    
    test_edge_idx = all_edge_idx[k*num_val:]
    test_edges = edges[test_edge_idx]
    test_edges_false = []

    #--------------

    for k_idx in range(k):
        val_edges_false = []

        while len(val_edges_false) < len(k_val_edges[k_idx]):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], k_train_edges[k_idx]):
                continue
            if ismember([idx_j, idx_i], k_train_edges[k_idx]):
                continue
            if ismember([idx_i, idx_j], k_val_edges[k_idx]):
                continue
            if ismember([idx_j, idx_i], k_val_edges[k_idx]):
                continue
            if len(k_val_edges_false) > k_idx:
                if ismember([idx_j, idx_i], np.array(k_val_edges_false[k_idx])):
                    continue
                if ismember([idx_i, idx_j], np.array(k_val_edges_false[k_idx])):
                    continue
            val_edges_false.append([idx_i, idx_j])

        k_val_edges_false.append(val_edges_false)

        # Sanity Checks
        assert ~ismember(k_val_edges_false[k_idx], edges_all)
        assert ~ismember(k_val_edges_false[k_idx], k_train_edges[k_idx])
        assert ~ismember(test_edges, k_train_edges[k_idx])
        assert ~ismember(k_val_edges[k_idx], k_train_edges[k_idx])
        assert ~ismember(k_val_edges[k_idx], test_edges)

    assert ~ismember(test_edges_false, edges_all)

    k_adj_train = []
    # create adj_train for k-fold setting
    for k_idx in range(k):

        data = np.ones(k_train_edges[k_idx].shape[0])
        train_edges = k_train_edges[k_idx]

        # Re-build train adj matrix
        adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        adj_train = adj_train + adj_train.T
        k_adj_train.append(adj_train)

    # NOTE: these edge lists only contain single direction of edge!
    return k_adj_train, k_train_edges, k_val_edges, k_val_edges_false, test_edges, test_edges_false
