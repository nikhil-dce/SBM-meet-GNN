from __future__ import division
from __future__ import print_function

import time
import os

import tensorflow as tf
import numpy as np
import sys

tf.set_random_seed(1234)
np.random.seed(1234)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Max number of epochs to train. Training may stop early if validation-error is not decreasing')

flags.DEFINE_string('hidden', '64_50', 'Number of units in hidden layers')
flags.DEFINE_integer('g_hidden', 32, 'Number of units in generator hidden layer')
flags.DEFINE_integer('deep_decoder', 1, 'Use deep decoder')


flags.DEFINE_float('weight_decay', 0.0, 'Weight for L2/L1 loss on embedding matrix')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('model', 'dglfrm', 'Model string: lfrm, dglfrm, dglfrm_b')
flags.DEFINE_string('dataset', 'cora', 'Dataset string: cora, citeseer, pubmed, yeast, nips12, nips234')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

flags.DEFINE_float('alpha0', 10., 'Prior Alpha for Beta')
flags.DEFINE_float('temp_prior', 0.5, 'Prior temperature for concrete distribution')
flags.DEFINE_float('temp_post', 1., 'Posterior temperature for concrete distribution')

flags.DEFINE_integer('use_k_fold', 0, 'Whether to use k-fold cross validation')
flags.DEFINE_integer('k', 5, 'how many folds for cross validation.')
flags.DEFINE_integer('early_stopping', 0, 'how many epochs to train after last best validation')

flags.DEFINE_integer('split_idx', 0, 'Dataset split (Total:10) 0-9')
flags.DEFINE_integer('weighted_ce', 1, 'Weighted Cross Entropy: For class balance')

flags.DEFINE_integer('test', 0, 'Load model and run on test')
flags.DEFINE_integer('reconstruct_x', 1, 'Whether to separately reconstruct x')
flags.DEFINE_string('gpu_to_use','0','Which GPU to use. Leave blank to use None')

flags.DEFINE_integer('log_results', 1, 'Load model and run on test')

import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

from optimizer import OptimizerDGLFRM, OptimizerLFRM
from input_data import load_data, load_masked_test_edges
from model import DGLFRM, DGLFRM_B
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges, mask_test_edges_randomly 

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_to_use
print ('Using gpus: ' + FLAGS.gpu_to_use)
print ('---------------------------------')
print ('Dataset: ' + FLAGS.dataset + '_' + str(FLAGS.split_idx))
print ('Alpha0: ' + str(FLAGS.alpha0))
print ('WeightedCE: ' + str(FLAGS.weighted_ce))
print ('ReconstructX: ' + str(FLAGS.reconstruct_x))

model_str = FLAGS.model
dataset_str = FLAGS.dataset

print (model_str)
if (model_str == 'dglfrm' or model_str == 'dglfrm_b'):
    if (len(FLAGS.hidden.split('_')) < 2):
        sys.exit("The truncation parameter missing. Specify '--hidden <layer_1>_<truncation_parameter>'")
save_dir = './data/' + dataset_str +'/split_'+ str(FLAGS.split_idx) + '/' + model_str + "/" + FLAGS.hidden + "/"
if not os.path.exists(save_dir):
        os.makedirs(save_dir)

# Load data. Raw adj is NxN Matrix and Features is NxF Matrix. Using sparse matrices here (See scipy docs). 
adj, features, feature_presence = load_data(dataset_str)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

print ("Adj Original Matrix: " + str(adj_orig.shape))
print ("Features Shape: " + str(features.shape))

features_shape = features.shape[0]
if FLAGS.features == 0:
        features = sp.identity(features_shape)  # featureless

pos_weight_feats = float(features.shape[0] * features.shape[1] - features.sum()) / features.sum() # (N) / P
norm_feats = features.shape[0] * features.shape[1] / float((features.shape[0] * features.shape[1] - features.sum()) * 2) # (N+P) / (N)
        
# feature sparse matrix to tuples 
features = sparse_to_tuple(features.tocoo())     

def get_score_matrix(sess, placeholders, feed_dict, model, S=5, save_qual=False):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['is_training']: False})
        
    if  model_str == 'dglfrm_b':
        # not using total_bias for now (The model.total_bias parameter is not being trained for now)
        outs = sess.run([model.logit_post, model.total_bias, model.w_gen_1, model.b_gen_1, model.w_gen_2, model.b_gen_2], feed_dict)
        adj_rec, z_activated = model.monte_carlo_sample(outs[0], FLAGS.temp_post, S, sigmoid, outs[2], outs[3], outs[4], outs[5])
        adj_rec = adj_rec
    elif model_str == 'dglfrm':
        # not using the total bias for now (The model.total_bias parameter is not being trained for now)
        outs = sess.run([model.logit_post, model.z_mean, model.z_log_std, model.total_bias, model.w_gen_1, model.b_gen_1, model.w_gen_2, model.b_gen_2], feed_dict)
        adj_rec, z_activated = model.monte_carlo_sample(outs[0], outs[1], outs[2], FLAGS.temp_post, S, sigmoid, outs[4], outs[5], outs[6], outs[7])
        adj_rec = adj_rec 
    else:
        emb = sess.run(model.z_mean, feed_dict=feed_dict)
        adj_rec = np.dot(emb, emb.T)
        z_activated = model.hidden[model.num_hidden_layers-1]

    return adj_rec, z_activated

"""
Get ROC score and average precision
"""
def get_roc_score(adj_rec, edges_pos, edges_neg, emb=None):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    # Compute precision recall curve 
    precision, recall, _ = precision_recall_curve(labels_all, preds_all)
    auc_pr = auc(recall, precision)
    
    return roc_score, ap_score, auc_pr

# create_model 
def create_model(placeholders, adj, features):

    num_nodes = adj.shape[0]
    num_features = features[2][1]
    features_nonzero = features[1].shape[0] # Can be used for dropouts. See GraphConvolutionSparse

    # Create model
    model = None

    if model_str == 'lfrm':
        model = LFRM(placeholders, num_features, num_nodes, features_nonzero)
    elif model_str == 'dglfrm':
        model = DGLFRM(placeholders, num_features, num_nodes, features_nonzero)
    elif model_str == 'dglfrm_b':
        model = DGLFRM_B(placeholders, num_features, num_nodes, features_nonzero)

    edges_for_loss = placeholders['edges_for_loss']
    num_train = placeholders['num_train']

        # Optimizer
    with tf.name_scope('optimizer'):

        if model_str == 'dglfrm':
                opt = OptimizerDGLFRM(labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                              validate_indices=False), [-1]),
                                       model=model, num_nodes=num_nodes,
                                       features = tf.reshape(tf.sparse_tensor_to_dense(placeholders['features'], validate_indices = False), [-1]),
                                       pos_weight=placeholders['pos_weight'],
                                       norm=placeholders['norm'],
                                       weighted_ce=FLAGS.weighted_ce,
                                       edges_for_loss=edges_for_loss, num_train=num_train,
                                       pos_weight_feats = pos_weight_feats, norm_feats = norm_feats
                                       )

        elif model_str == 'lfrm' or model_str == 'dglfrm_b':
                opt = OptimizerLFRM(labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                              validate_indices=False), [-1]),
                                       model=model, num_nodes=num_nodes,
                                       features = tf.reshape(tf.sparse_tensor_to_dense(placeholders['features'], validate_indices = False), [-1]),
                                       pos_weight=placeholders['pos_weight'],
                                       norm=placeholders['norm'],
                                       weighted_ce=FLAGS.weighted_ce,
                                       edges_for_loss=edges_for_loss, num_train=num_train,                                    
                                       pos_weight_feats = pos_weight_feats)

    return model, opt

def train(placeholders, model, opt, adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, features, sess, name="single_fold"):

    adj = adj_train
    
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum() # N/P
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2) # (N+P) x (N+P) / (2N)

    print (adj_train.shape)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # Some preprocessing. adj_norm is D^(-1/2) x adj x D^(-1/2)
    adj_norm = preprocess_graph(adj)

    # session initialize
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    val_roc_score = []
    best_validation = 0.0

    num_nodes = adj.shape[0]

    edges_for_loss = np.ones((num_nodes*num_nodes), dtype=np.float32)

    ignore_edges = []
    edges_to_ignore = np.concatenate((val_edges, val_edges_false, test_edges, test_edges_false), axis=0)
    for e in edges_to_ignore:
        ignore_edges.append(e[0]*num_nodes+e[1])
    edges_for_loss[ignore_edges] = 0
    num_train = num_nodes * num_nodes - len(ignore_edges)

    last_best_epoch = 0

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        feed_dict.update({placeholders['is_training']: True})
        feed_dict.update({placeholders['norm']: norm})
        feed_dict.update({placeholders['pos_weight']: pos_weight})
        feed_dict.update({placeholders['edges_for_loss']: edges_for_loss})
        feed_dict.update({placeholders['num_train']: num_train})
        
        avg_x_cost = 0
        
        if model_str  == 'dglfrm':
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy, opt.x_loss, model.a, model.b, model.z_real, model.z_discrete], feed_dict=feed_dict)
            # a, b are global parameters
            a, b = np.log(1 + np.exp(outs[4])), np.log(1 + np.exp(outs[5]))
            a = np.mean(a)
            b = np.mean(b)
            #regularization = round(outs[3], 2)
            regularization = 0
            z_discrete = outs[7]
            z_real = outs[6]
            avg_x_cost = outs[3]
            W = None

        elif model_str == 'dglfrm_b':
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy, opt.x_loss, model.a, model.b, model.z], feed_dict=feed_dict)
            regularization = 0
            z_discrete = outs[6]
            z_real = None
            avg_x_cost = outs[3]
            W = None
                        
        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]

        adj_rec, z_activated = get_score_matrix(sess, placeholders, feed_dict, model, S=1)
        roc_curr, ap_curr, _  = get_roc_score(adj_rec, val_edges, val_edges_false)
        
        print("Epoch:", '%03d' % (epoch + 1), "cost=", "{:.3f}".format(avg_cost), 
              "x_recon_loss=", "{:.2f}".format(avg_x_cost),
              "val_roc=", "{:.3f}".format(roc_curr), "val_ap=", "{:.3f}".format(ap_curr), 
              'activated_z=', "{:.1f}".format(z_activated), "time=", "{:.2f}".format(time.time() - t))

        roc_curr = round(roc_curr, 3)
        val_roc_score.append(roc_curr)

        # Look-ahead epochs: (We may need to train for some more epochs due to nested stochastic nature of the framework.) 
        if FLAGS.early_stopping != 0 and roc_curr > best_validation: 
            # save model
            print ('Saving model')
            saver.save(sess=sess, save_path=save_dir+name)
            best_validation = roc_curr
            last_best_epoch = 0

        if FLAGS.early_stopping != 0 and last_best_epoch > FLAGS.early_stopping:
            break
        else:
            last_best_epoch += 1

    print("Optimization Finished!")
    
    val_max_index = np.argmax(val_roc_score)
    
    print ('---------------------------------')

    print('Validation ROC Max: {:.3f} at Epoch: {:04d}'.format(val_roc_score[val_max_index], val_max_index))
            
    qual_file = 'data/qual_' + dataset_str + '_' + model_str

    if model_str == 'dglfrm':
        np.savez(qual_file, z_discrete=np.asarray(z_discrete), z_real=np.asarray(z_real), z_out=np.asarray(np.multiply(np.round(z_discrete), z_real)), adj_rec=adj_rec)
    elif model_str == 'dglfrm_b':
        np.savez(qual_file, z_discrete=np.asarray(z_discrete), adj_rec=adj_rec)
    
    if FLAGS.early_stopping != 0:
        saver.restore(sess=sess, save_path=(save_dir+name))

    adj_score, z_activated = get_score_matrix(sess, placeholders, feed_dict, model)

    return adj_score, z_activated

def load_model(placeholders, model, opt, adj_train, test_edges, test_edges_false, features, sess, name="single_fold"):

        adj = adj_train
        # This will be calculated for every fold
        # pos_weight and norm should be tensors
        print ('----------------')
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum() # N/P
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2) # (N+P) x (N+P) / (N)

        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = sparse_to_tuple(adj_label)

        # Some preprocessing. adj_norm is D^(-1/2) x adj x D^(-1/2)
        adj_norm = preprocess_graph(adj)
    
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        feed_dict.update({placeholders['is_training']: True})
        feed_dict.update({placeholders['norm']: norm})
        feed_dict.update({placeholders['pos_weight']: pos_weight})

        # Some preprocessing. adj_norm is D^(-1/2) x adj x D^(-1/2)
        adj_norm = preprocess_graph(adj)
        saver = tf.train.Saver()
        
        saver.restore(sess=sess, save_path=(save_dir+name))
        print ('Model restored')

        # Decrease MC samples for pubmed 
        if (dataset_str == 'pubmed'): 
                S = 5
        else:
                S = 15
        
        adj_score, z_activated = get_score_matrix(sess, placeholders, feed_dict, model, S=S, save_qual=True)

        return adj_score, z_activated

def main():

    num_nodes = adj_orig.shape[0]
    print ("Model is " + model_str)

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'is_training': tf.placeholder(tf.bool),
        'norm': tf.placeholder(tf.float32),
        'pos_weight': tf.placeholder(tf.float32),
        'edges_for_loss': tf.placeholder(tf.float32),
        'num_train': tf.placeholder(tf.float32),
        'epoch': tf.placeholder(tf.float32)        
    }

    model, opt = create_model(placeholders, adj, features)
    sess = tf.Session()
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = load_masked_test_edges(dataset_str, FLAGS.split_idx)
    #adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj, None)
        
    if FLAGS.test:
        adj_score, z_activated = load_model(placeholders, model, opt, adj_train[0], test_edges,
                                            test_edges_false, features, sess)
    else:
        adj_score, z_activated = train(placeholders, model, opt, adj_train[0], train_edges, val_edges,
                                        val_edges_false, test_edges, test_edges_false, features, sess)

    roc_score, ap_score, auc_pr = get_roc_score(adj_score, test_edges, test_edges_false)
    # all_adj_scores = adj_score

    result_log = FLAGS.model + '_' + FLAGS.dataset + '_results'
    result_log += '_alpha:' + str(FLAGS.alpha0) 
    result_log += '_hidden:' + FLAGS.hidden 
    result_log += '_es:' + str(FLAGS.early_stopping)
    result_log += '_maxEpochs:' + str(FLAGS.epochs) + '.txt'

    with open(result_log, 'a') as f:
        f.write ('---------------------------------\n')
        f.write('Split Idx: ' + str(FLAGS.split_idx) + '\n')
        f.write('Test ROC score: ' + str(roc_score) + '\n')
        f.write('Test AP score: ' + str(ap_score) + '\n')
        f.write('Test Z Activated: ' + str(z_activated) + '\n')
    f.close()

    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))
    # print('Test AUC PR Curve: ' + str(auc_pr))
    print('Test Z Activated: ' + str(z_activated))

    print ('---------------------------------')
    print ('Dataset: ' + FLAGS.dataset + '_' + str(FLAGS.split_idx))
    print ('Train set size: ' + str(len(train_edges)*2))
    print ('Val set size: ' + str(len(val_edges)*2))
    print ('Test set size: ' + str(len(test_edges)*2))
    print ('Alpha0: ' + str(FLAGS.alpha0))
    print ('WeightedCE: ' + str(FLAGS.weighted_ce))
    print ('ReconstructX: ' + str(FLAGS.reconstruct_x))

if __name__ == '__main__':
    main()

