import tensorflow as tf
from utils import *

flags = tf.app.flags
FLAGS = flags.FLAGS

class OptimizerDGLFRM(object):
    def __init__(self, labels, model, num_nodes, features, pos_weight, norm, weighted_ce, edges_for_loss, num_train,
                pos_weight_feats, norm_feats, node_labels = None,
                node_labels_mask = None):

        preds_sub = model.reconstructions
        labels_sub = labels

        if weighted_ce == 0:
            # Loss not weighted
            norm = 1
            pos_weight = 1
            
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        if edges_for_loss is not None:
            self.all_cost = tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight)
            self.cost = norm * tf.reduce_sum(tf.multiply(self.all_cost, edges_for_loss)) / num_train
        else:
            self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        
        self.kl_zreal = kl_real(model.z_log_std, model.z_mean) / num_nodes 
        self.cost += self.kl_zreal
        
        self.kl_discrete = kl_discrete(model.logit_post, logit(tf.exp(model.log_prior)), model.y_sample, FLAGS.temp_post, FLAGS.temp_prior) / num_nodes
        self.kl_v = kl_kumar_beta(model.beta_a, model.beta_b, FLAGS.alpha0, log_beta_prior=np.log(1./FLAGS.alpha0)) / num_nodes

        self.cost += (self.kl_discrete + self.kl_v)

        self.x_loss = tf.constant(0.)

        if (FLAGS.reconstruct_x):
            self.x_loss = tf.nn.weighted_cross_entropy_with_logits(logits=model.x_hat, targets=features, pos_weight=pos_weight_feats)
            self.x_loss = tf.reduce_mean(self.x_loss) #* norm_feats
            self.cost += self.x_loss

        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        self.opt_op = self.optimizer.apply_gradients(self.grads_vars)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
class OptimizerLFRM(object):
    
    def __init__(self, labels, model, num_nodes, features, pos_weight, norm, weighted_ce, edges_for_loss, num_train,
                pos_weight_feats):

        preds_sub = model.reconstructions
        labels_sub = labels

        if weighted_ce == 0:
            # Loss not weighted
            norm = 1
            pos_weight = 1

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) # Adam_Optimizer

        if edges_for_loss is not None:
            self.cost = tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight)
            self.cost = norm * tf.reduce_sum(tf.multiply(self.cost, edges_for_loss)) / num_train
        else:
            self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))

        self.kl_discrete = kl_discrete(model.logit_post, logit(tf.exp(model.log_prior)), model.y_sample, FLAGS.temp_post, FLAGS.temp_prior) / num_nodes
        self.kl_v = kl_kumar_beta(model.beta_a, model.beta_b, FLAGS.alpha0, log_beta_prior=np.log(1./FLAGS.alpha0)) / num_nodes
        self.cost += self.kl_discrete + self.kl_v

        self.x_loss = tf.constant(0.)

        # Original lfrm does not use log-likelihood of side-information
        if (FLAGS.model != 'lfrm' and FLAGS.reconstruct_x):
            self.x_loss = tf.nn.weighted_cross_entropy_with_logits(logits=model.x_hat, targets=features, pos_weight=pos_weight_feats)
            self.x_loss = tf.reduce_mean(self.x_loss) #* norm_feats
            self.cost += self.x_loss
        
        self.regularization = model.get_regualizer_cost(lambda x: tf.reduce_sum(tf.abs(x)))
        self.grad_vars = self.optimizer.compute_gradients(self.cost)
        self.opt_op = self.optimizer.apply_gradients(self.grad_vars)
    
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                        tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))  

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
 
    return tf.reduce_mean(accuracy_all)
