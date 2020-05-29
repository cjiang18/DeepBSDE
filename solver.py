import logging
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tensorflow.keras.layers as layers
DELTA_CLIP = 50.0


class BSDESolver(object):
    """The fully connected neural network model."""
    def __init__(self, config, bsde,use_universal_model = False):
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde

        try:
            self.isXVA = bsde.isXVA
        except AttributeError:
            self.isXVA = False
        
        self.model = NonsharedModel(config, bsde,use_universal_model)
        #self.y_init = self.model.y_init

        try:
            lr_schedule = config.net_config.lr_schedule
        except AttributeError:
            lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                self.net_config.lr_boundaries, self.net_config.lr_values)     
            
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config.valid_size)

        # begin sgd iteration
        for step in tqdm(range(self.net_config.num_iterations+1)):
            if step % self.net_config.logging_frequency == 0:
                loss = self.loss_fn(valid_data, training=False).numpy()
                y_init = self.model.y_init.numpy()[0]
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, y_init, elapsed_time])
                if self.net_config.verbose:
                    #logging.info("step: %5u,    loss: %.4e, Y0: %.4e,   elapsed time: %3u" % (
                    #    step, loss, y_init, elapsed_time))
                    print("step: %5u,    loss: %.4e, Y0: %.4e,   elapsed time: %3u" % (
                        step, loss, y_init, elapsed_time))
            self.train_step(self.bsde.sample(self.net_config.batch_size))            
        return np.array(training_history)

    def loss_fn(self, inputs, training):
        if self.isXVA:
            dw, x, v_clean = inputs
            y_terminal = self.model(inputs, training)
            delta = y_terminal - self.bsde.g_tf(self.bsde.total_time, x[:, :, -1],v_clean[:,:,-1])
            # use linear approximation outside the clipped range
            loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                                        2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))
        else:
            dw, x = inputs
            y_terminal = self.model(inputs, training)
            delta = y_terminal - self.bsde.g_tf(self.bsde.total_time, x[:, :, -1])
            # use linear approximation outside the clipped range
            loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                                        2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))

        loss += 1000*(tf.maximum(self.model.y_init[0]-self.net_config.y_init_range[1],0)+tf.maximum(self.net_config.y_init_range[0]-self.model.y_init[0],0))
        return loss

    def grad(self, inputs, training):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_fn(inputs, training)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step(self, train_data):
        grad = self.grad(train_data, training=True)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))     


class NonsharedModel(tf.keras.Model):
    def __init__(self, config, bsde,use_universal_model):
        super(NonsharedModel, self).__init__()
        self.config = config
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde       
        self.dim = bsde.dim
        self.y_init = tf.Variable(np.random.uniform(low=self.net_config.y_init_range[0],
                                                    high=self.net_config.y_init_range[1],
                                                    size=[1]),dtype=self.net_config.dtype
                                  )
        self.z_init = tf.Variable(np.random.uniform(low=-.1, high=.1,
                                                    size=[1, self.eqn_config.dim]),dtype=self.net_config.dtype
                                  )        
        if use_universal_model: 
            self.subnet = get_universal_neural_network(self.dim+1)
        else:
            self.subnet = [FeedForwardSubNet(config,bsde.dim) for _ in range(self.bsde.num_time_interval-1)]

        try:
            self.isXVA = bsde.isXVA
        except AttributeError:
            self.isXVA = False
       

    def call(self, inputs, training):
        if self.isXVA:
            dw, x, v_clean = inputs
            time_stamp = np.arange(0, self.eqn_config.num_time_interval) * self.bsde.delta_t
            all_one_vec = tf.ones(shape=tf.stack([tf.shape(dw)[0], 1]), dtype=self.net_config.dtype)
            y = all_one_vec * self.y_init
            z = tf.matmul(all_one_vec, self.z_init)
            
            for t in range(0, self.bsde.num_time_interval-1):
                y = y - self.bsde.delta_t * (
                    self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z,v_clean[:,:,t])
                ) + tf.reduce_sum(z * dw[:, :, t], 1, keepdims=True) 
                try:          
                    z = self.subnet[t](x[:, :, t + 1], training) / self.bsde.dim
                except TypeError:
                    z = self.subnet(tf.concat([time_stamp[t+1]*all_one_vec,x[:, :, t + 1]],axis=1), training=training) / self.bsde.dim
            # terminal time
            y = y - self.bsde.delta_t * self.bsde.f_tf(time_stamp[-1], x[:, :, -2], y, z,v_clean[:,:,-2]) + \
                tf.reduce_sum(z * dw[:, :, -1], 1, keepdims=True)
        else:
            dw, x = inputs
            time_stamp = np.arange(0, self.eqn_config.num_time_interval) * self.bsde.delta_t
            all_one_vec = tf.ones(shape=tf.stack([tf.shape(dw)[0], 1]), dtype=self.net_config.dtype)
            y = all_one_vec * self.y_init
            z = tf.matmul(all_one_vec, self.z_init)
            
            for t in range(0, self.bsde.num_time_interval-1):
                y = y - self.bsde.delta_t * (
                    self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z)
                ) + tf.reduce_sum(z * dw[:, :, t], 1, keepdims=True)           
                try:          
                    z = self.subnet[t](x[:, :, t + 1], training) / self.bsde.dim
                except TypeError:
                    z = self.subnet(tf.concat([time_stamp[t+1]*all_one_vec,x[:, :, t + 1]],axis=1), training=training) / self.bsde.dim
            # terminal time
            y = y - self.bsde.delta_t * self.bsde.f_tf(time_stamp[-1], x[:, :, -2], y, z) + \
                tf.reduce_sum(z * dw[:, :, -1], 1, keepdims=True)
        return y         

    def predict_step(self, data):
        # this function returns value of y at each future time
        if self.isXVA: 
            dw, x, v_clean = data[0]
            time_stamp = np.arange(0, self.eqn_config.num_time_interval) * self.bsde.delta_t
            all_one_vec = tf.ones(shape=tf.stack([tf.shape(dw)[0], 1]), dtype=self.net_config.dtype)
            y = all_one_vec * self.y_init
            z = tf.matmul(all_one_vec, self.z_init)        
            
            history = tf.TensorArray(self.net_config.dtype,size=self.bsde.num_time_interval+1)     
            history = history.write(0,y)
            
            for t in range(0, self.bsde.num_time_interval-1):
                y = y - self.bsde.delta_t * (
                    self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z,v_clean[:,:,t])
                ) + tf.reduce_sum(z * dw[:, :, t], 1, keepdims=True)
                
                history = history.write(t+1,y)
                try:          
                    z = self.subnet[t](x[:, :, t + 1], training=False) / self.bsde.dim
                except TypeError:
                    z = self.subnet(tf.concat([time_stamp[t+1]*all_one_vec,x[:, :, t + 1]],axis=1), training=False) / self.bsde.dim
            # terminal time
            y = y - self.bsde.delta_t * self.bsde.f_tf(time_stamp[-1], x[:, :, -2], y, z,v_clean[:,:,-2]) + \
                tf.reduce_sum(z * dw[:, :, -1], 1, keepdims=True)
        
            history = history.write(self.bsde.num_time_interval,y)
            history = tf.transpose(history.stack(),perm=[1,2,0])
            return dw,x,v_clean,history      
        else:
            dw, x = data[0]
            time_stamp = np.arange(0, self.eqn_config.num_time_interval) * self.bsde.delta_t
            all_one_vec = tf.ones(shape=tf.stack([tf.shape(dw)[0], 1]), dtype=self.net_config.dtype)
            y = all_one_vec * self.y_init
            z = tf.matmul(all_one_vec, self.z_init)        
            
            history = tf.TensorArray(self.net_config.dtype,size=self.bsde.num_time_interval+1)     
            history = history.write(0,y)
            
            for t in range(0, self.bsde.num_time_interval-1):
                y = y - self.bsde.delta_t * (
                    self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z)
                ) + tf.reduce_sum(z * dw[:, :, t], 1, keepdims=True)
                
                history = history.write(t+1,y)
                try:          
                    z = self.subnet[t](x[:, :, t + 1], training=False) / self.bsde.dim
                except TypeError:
                    z = self.subnet(tf.concat([time_stamp[t+1]*all_one_vec,x[:, :, t + 1]],axis=1), training=False) / self.bsde.dim
            # terminal time
            y = y - self.bsde.delta_t * self.bsde.f_tf(time_stamp[-1], x[:, :, -2], y, z) + \
                tf.reduce_sum(z * dw[:, :, -1], 1, keepdims=True)
        
            history = history.write(self.bsde.num_time_interval,y)
            history = tf.transpose(history.stack(),perm=[1,2,0])
            return dw,x,history      

    def simulate_path(self,num_sample):
        if self.isXVA:
            return self.predict(num_sample)[3]
        else:
            return self.predict(num_sample)[2]           


class FeedForwardSubNet(tf.keras.Model):
    def __init__(self, config,dim):
        super(FeedForwardSubNet, self).__init__()        
        num_hiddens = config.net_config.num_hiddens
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(len(num_hiddens) + 2)]
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=False,
                                                   activation=None,)
                             for i in range(len(num_hiddens))]
        # final output should be gradient of size dim
        self.dense_layers.append(tf.keras.layers.Dense(dim, activation=None))

    def call(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense """
        x = self.bn_layers[0](x, training)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        return x

### univeral neural networks instead of one neural network at each time point
def get_universal_neural_network(input_dim):    
    input = layers.Input(shape=(input_dim,))
    x = layers.BatchNormalization()(input)    
    for i in range(5):
        x = layers.Dense(input_dim+10,'relu',False)(x)
        x = layers.BatchNormalization()(x)
    output = layers.Dense(input_dim-1,'relu')(x)
    #output = layers.Dense(2*dim,'relu')(x)
    return tf.keras.Model(input,output)
'''
def get_universal_neural_network(input_dim,num_neurons=20,num_hidden_blocks=4):
    
    input = tf.keras.Input(shape=(input_dim,))
    x = layers.BatchNormalization()(input)

    s = layers.Dense(num_neurons,activation='relu',use_bias=False)(x)
    s = layers.BatchNormalization()(s)
    for i in range(num_hidden_blocks-1):        

        z = layers.add([layers.Dense(num_neurons,None,False)(x),layers.Dense(num_neurons,None,False)(s)])
        z = Add_bias(num_neurons)(z)
        z = layers.Activation(tf.nn.sigmoid)(z)
       

        g = layers.add([layers.Dense(num_neurons,None,False)(x),layers.Dense(num_neurons,None,False)(s)])
        g = Add_bias(num_neurons)(g)
        g = layers.Activation(tf.nn.sigmoid)(g)

        r = layers.add([layers.Dense(num_neurons,None,False)(x),layers.Dense(num_neurons,None,False)(s)])
        r = Add_bias(num_neurons)(r)
        r = layers.Activation(tf.nn.sigmoid)(r)

        h = layers.add([layers.Dense(num_neurons,None,False)(x),layers.Dense(num_neurons,None,False)(layers.multiply([s,r]))])
        h = Add_bias(num_neurons)(h)
        h = layers.Activation(tf.nn.relu)(h)

        s = layers.add([layers.multiply([1-g,h]),layers.multiply([z,s])])
        s = layers.BatchNormalization()(s)
    
    output = layers.Dense(input_dim-1,None)(s)

    return tf.keras.Model(input,output)
'''
      
class Add_bias(tf.keras.layers.Layer):
    def __init__(self,units):        
        super(Add_bias, self).__init__()       
        self.units = units
    def build(self, input_shape):              
        self.b = self.add_weight(shape=(self.units,),
                                initializer='zeros',
                                trainable=True)
    def call(self, inputs):
        return inputs + self.b


    