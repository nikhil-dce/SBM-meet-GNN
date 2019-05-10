from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, WeightedInnerProductDecoder
import tensorflow as tf
from utils import *
from initializations import weight_variable_glorot

flags = tf.app.flags
FLAGS = flags.FLAGS
SMALL = 1e-16

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass
        
class DGLFRM(Model): # DGLFRM See paper
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, num_classes = 0, **kwargs):
        super(DGLFRM, self).__init__(**kwargs)

        self.num_classes = num_classes
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.training = placeholders['is_training']

        self.hidden = [int(x) for x in FLAGS.hidden.split('_')]
        self.num_hidden_layers = len(self.hidden)

        a_val = np.log(np.exp(FLAGS.alpha0) - 1) # inverse softplus
        b_val = np.log(np.exp(1.) - 1)
        initial = tf.zeros((self.hidden[self.num_hidden_layers-1]))
        with tf.variable_scope(self.name + "_globalvars"):
            self.a = tf.Variable(initial, name="alpha") + a_val
            self.b = tf.Variable(initial, name="beta") + b_val
            
            self.e = tf.constant(0, shape=[num_nodes, 1], name='e', dtype=tf.float32)
            self.c = tf.constant(0, shape=[1,1], name='c', dtype=tf.float32)

        with tf.variable_scope("weight_gen_1"):
            self.w_gen_1 = weight_variable_glorot(self.hidden[self.num_hidden_layers-1], FLAGS.g_hidden, name='weight_generator_1')
            self.b_gen_1 = tf.Variable(tf.zeros(FLAGS.g_hidden), name='bias_generator_1')

        with tf.variable_scope("weight_gen_2"):
            self.w_gen_2 = weight_variable_glorot(FLAGS.g_hidden, int(FLAGS.g_hidden/2), name='weight_generator_2')
            self.b_gen_2 = tf.Variable(tf.zeros(FLAGS.g_hidden/2), name='bias_generator_2')

        with tf.variable_scope("weight_gen_x"):
            self.w_gen_x = weight_variable_glorot(self.hidden[-1], self.input_dim, name='weight_generator_x_recon')
            self.b_gen_x = tf.Variable(tf.zeros(self.input_dim), name='bias_generator_x_recon')
        
        with tf.name_scope("build_model"):
            self.build()
           
    def get_regualizer_cost(self, regularizer):

        regularization = 0
        #regularization += self.weightedInnerProductDecoderLayer.apply_regularizer(regularizer)
        return regularization
        
    def _build(self):

        # total_bias not being used for now
        entity_bias = self.e   
        entity_bias_matrix = tf.tile(entity_bias, [1, self.n_samples])
        entity_bias_matrix += tf.transpose(entity_bias_matrix)
        self.total_bias = entity_bias_matrix + tf.tile(self.c, [self.n_samples, self.n_samples])
        
        print 'Build Dynamic Network....'
        for idx, hidden_layer in enumerate(self.hidden):

            with tf.name_scope("GCN_layer_" + str(idx)):

                if idx == 0:
                    h = GraphConvolutionSparse(input_dim=self.input_dim,
                                               output_dim=hidden_layer,
                                               adj=self.adj,
                                               features_nonzero=self.features_nonzero,
                                               act=lambda x: tf.nn.leaky_relu(x, alpha=0.2),
                                               dropout=self.dropout,
                                               logging=self.logging)(self.inputs)
                
                elif idx == self.num_hidden_layers-1:
                    
                    h1 = GraphConvolution(input_dim=self.hidden[idx-1],
                                         output_dim=hidden_layer,
                                         adj=self.adj,
                                         act=lambda x: x,
                                         dropout=self.dropout,
                                         logging=self.logging)(h)
                    h2 = GraphConvolution(input_dim=self.hidden[idx-1],
                                         output_dim=hidden_layer,
                                         adj=self.adj,
                                         act=lambda x: x,
                                         dropout=self.dropout,
                                         logging=self.logging)(h)
                    h3 = GraphConvolution(input_dim=self.hidden[idx-1],
                                         output_dim=hidden_layer,
                                         adj=self.adj,
                                         act=lambda x: x,
                                         dropout=self.dropout,
                                         logging=self.logging)(h)
                else:
                    h = GraphConvolution(input_dim=self.hidden[idx-1],
                                         output_dim=hidden_layer,
                                         adj=self.adj,
                                         act=lambda x: tf.nn.leaky_relu(x, alpha=0.2),
                                         dropout=self.dropout,
                                         logging=self.logging)(h)
        
        self.z_mean, self.z_log_std, self.pi_logit = h1, h2, h3 
        
        # See this 0.01
        beta_a = tf.nn.softplus(self.a) 
        beta_b = tf.nn.softplus(self.b) 

        beta_a = tf.expand_dims(beta_a, 0)
        beta_b = tf.expand_dims(beta_b, 0)
                
        self.beta_a = tf.tile(beta_a, [self.n_samples, 1])
        self.beta_b = tf.tile(beta_b, [self.n_samples, 1])

        self.v = kumaraswamy_sample(self.beta_a, self.beta_b)
        v_term = tf.log(self.v+SMALL)
        self.log_prior = tf.cumsum(v_term, axis=1)

        self.logit_post = self.pi_logit + logit(tf.exp(self.log_prior))
        
        # note: logsample is just logit(z_discrete), unless we've rounded
        self.z_discrete, self.z_real, _, self.y_sample = sample(self.z_mean, self.z_log_std, self.logit_post, None, None, FLAGS.temp_post, calc_v=False)
        self.z_discrete = tf.cond(tf.equal(self.training, tf.constant(False)), lambda: tf.round(self.z_discrete), lambda: self.z_discrete)

        z = tf.multiply(self.z_discrete, self.z_real)

        if FLAGS.deep_decoder:
            f = tf.nn.leaky_relu(tf.matmul(z, self.w_gen_1) + self.b_gen_1, alpha=0.2)        
            f = tf.matmul(f, self.w_gen_2) + self.b_gen_2
            self.reconstructions = InnerProductDecoder(act=lambda x: x,
                                      logging=self.logging)(f)
        else :
            self.reconstructions = InnerProductDecoder(act=lambda x: x,
                                                        logging=self.logging)(z)

        self.x_hat = tf.reshape(tf.matmul(z, self.w_gen_x) + self.b_gen_x, [-1])

        
    def monte_carlo_sample(self, pi_logit, z_mean, z_log_std, temp, S, sigmoid_fn, w_g1, b_g1, w_g2, b_g2):

        shape = list(np.shape(pi_logit))
        shape.insert(0, S)

        # mu + standard_samples * stand_deviation
        z_real = z_mean + np.multiply(np.random.normal(0, 1, shape), np.exp(z_log_std))
    
        # Concrete instead of Bernoulli => equivalent to reparametrize_discrete in tensorflow
        uniform = np.random.uniform(1e-4, 1. - 1e-4, shape)
        logistic = np.log(uniform) - np.log(1 - uniform)

        y_sample = (pi_logit + logistic) / temp

        z_discrete = sigmoid_fn(y_sample)
        z_discrete = np.round(z_discrete)        

        emb = np.multiply(z_real, z_discrete)

        if FLAGS.deep_decoder:
            f = np.matmul(emb, w_g1) + b_g1
            f = np.maximum(f, 0.2*f)
            f = np.matmul(f, w_g2) + b_g2
            emb_t = np.transpose(f, (0, 2, 1))
            adj_rec = np.matmul(f, emb_t)
        else :
            emb_t = np.transpose(emb, (0, 2, 1))
            adj_rec = np.matmul(emb, emb_t)
        
        adj_rec = np.mean(adj_rec, axis=0)
        z_activated = np.sum(z_discrete) / (shape[0] * shape[1])

        return adj_rec, z_activated

# DGLFRM End-------------------------------------------------------------

class DGLFRM_B(Model): # DVLFRM_B

    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, num_classes = 0, **kwargs):
        super(DGLFRM_B, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.training = placeholders['is_training']

        self.hidden = [int(x) for x in FLAGS.hidden.split('_')]
        self.num_hidden_layers = len(self.hidden)

        a_val = np.log(np.exp(FLAGS.alpha0) - 1) # inverse softplus
        b_val = np.log(np.exp(1.) - 1)
        initial = tf.zeros((self.hidden[self.num_hidden_layers-1]))
        with tf.variable_scope(self.name + "_globalvars"):
            self.a = tf.Variable(initial, name="alpha") + a_val
            self.b = tf.Variable(initial, name="beta") + b_val
            self.e = tf.Variable(tf.zeros((num_nodes, 1)), name='e')
            self.c = tf.Variable(tf.zeros((1,1)), name='c')
        
        with tf.variable_scope("weight_gen_1"):
            self.w_gen_1 = weight_variable_glorot(self.hidden[self.num_hidden_layers-1], FLAGS.g_hidden, name='weight_generator_1')
            self.b_gen_1 = tf.Variable(tf.zeros(FLAGS.g_hidden), name='bias_generator_1')

        with tf.variable_scope("weight_gen_2"):
            self.w_gen_2 = weight_variable_glorot(FLAGS.g_hidden, FLAGS.g_hidden/2, name='weight_generator_2')
            self.b_gen_2 = tf.Variable(tf.zeros(FLAGS.g_hidden/2), name='bias_generator_2')

        with tf.variable_scope("weight_gen_x"):
            self.w_gen_x = weight_variable_glorot(self.hidden[self.num_hidden_layers-1], self.input_dim, name='weight_generator_x_recon')
            self.b_gen_x = tf.Variable(tf.zeros(self.input_dim), name='bias_generator_x_recon')
        
        self.num_classes = num_classes

        with tf.name_scope("Model"):
            self.build()

    def get_regualizer_cost(self, regularizer):

        regularization = 0
        #regularization += self.weightedInnerProductDecoderLayer.apply_regularizer(regularizer)

        return regularization

    def _build(self):

        # total bias not being used for now
        entity_bias = self.e   
        entity_bias_matrix = tf.tile(entity_bias, [1, self.n_samples])
        entity_bias_matrix += tf.transpose(entity_bias_matrix)
        self.total_bias = entity_bias_matrix + tf.tile(self.c, [self.n_samples, self.n_samples])
        
        for idx, hidden_layer in enumerate(self.hidden):

            if idx == 0:

                if self.num_hidden_layers == 1:
                    activ = lambda x: x
                else:
                    activ = lambda x: tf.nn.leaky_relu(x, alpha=0.2)
                    
                h = GraphConvolutionSparse(input_dim=self.input_dim,
                                               output_dim=hidden_layer,
                                               adj=self.adj,
                                               features_nonzero=self.features_nonzero,
                                               act=activ,
                                               dropout=self.dropout,
                                               logging=self.logging)(self.inputs)
                
            elif idx == self.num_hidden_layers-1:
                h = GraphConvolution(input_dim=self.hidden[idx-1],
                                         output_dim=hidden_layer,
                                         adj=self.adj,
                                         act=lambda x: x,
                                         dropout=self.dropout,
                                         logging=self.logging)(h)
            else:
                h = GraphConvolution(input_dim=self.hidden[idx-1],
                                         output_dim=hidden_layer,
                                         adj=self.adj,
                                         act=lambda x: tf.nn.leaky_relu(x, alpha=0.2),
                                         dropout=self.dropout,
                                         logging=self.logging)(h)
        self.pi_logit = h

        # See this 0.01
        beta_a = tf.nn.softplus(self.a) + 0.01
        beta_b = tf.nn.softplus(self.b) + 0.01

        beta_a = tf.expand_dims(beta_a, 0)
        beta_b = tf.expand_dims(beta_b, 0)
                
        self.beta_a = tf.tile(beta_a, [self.n_samples, 1])
        self.beta_b = tf.tile(beta_b, [self.n_samples, 1])

        self.v = kumaraswamy_sample(self.beta_a, self.beta_b)
        v_term = tf.log(self.v+SMALL)
        self.log_prior = tf.cumsum(v_term, axis=1)

        self.logit_post = self.pi_logit + logit(tf.exp(self.log_prior))

        # note: logsample is just logit(z_discrete), unless we've rounded
        self.z, _ , _, self.y_sample = sample(None, None, self.logit_post, None, None, FLAGS.temp_post, calc_v=False, calc_real=False)
        self.z = tf.cond(tf.equal(self.training, tf.constant(False)), lambda: tf.round(self.z), lambda: self.z)

        if FLAGS.deep_decoder:
            f = tf.nn.leaky_relu(tf.matmul(self.z, self.w_gen_1) + self.b_gen_1, alpha=0.2)
            f = tf.matmul(f, self.w_gen_2) + self.b_gen_2

            self.reconstructions = InnerProductDecoder(act=lambda x: x,
                                      logging=self.logging)(f)
        else :
            f = tf.matmul(self.z, self.w_gen_1) + self.b_gen_1
        
            self.reconstructions = InnerProductDecoder(act=lambda x: x,
                                      logging=self.logging)(f) 

        self.x_hat = tf.reshape(tf.matmul(self.z, self.w_gen_x) + self.b_gen_x, [-1])

        
    def monte_carlo_sample (self, pi_logit, temp, S, sigmoid_fn, w_g1, b_g1, w_g2, b_g2):

        shape = list(np.shape(pi_logit))
        shape.insert(0, S)
    
        # Concrete instead of Bernoulli => equivalent to reparametrize_discrete in tensorflow
        uniform = np.random.uniform(1e-4, 1. - 1e-4, shape)
        logistic = np.log(uniform) - np.log(1 - uniform)
        y_sample = (pi_logit + logistic) / temp
        
        z_discrete = sigmoid_fn(y_sample)
        z_discrete = np.round(z_discrete)
    
        z_activated = np.sum(z_discrete) / (shape[0] * shape[1])

        if FLAGS.deep_decoder:
            f = np.matmul(z_discrete, w_g1) + b_g1
            f = np.maximum(f, 0.2*f)
            f = np.matmul(f, w_g2) + b_g2
        else:
            f = np.matmul(z_discrete, w_g1) + b_g1
        
        emb_t = np.transpose(f, (0, 2, 1))
        adj_rec = np.matmul(f, emb_t)
        adj_rec = np.mean(adj_rec, axis=0)
        
        return adj_rec, z_activated

        
#------------DGLFRM_B END----------------------
