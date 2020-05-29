import tensorflow as tf
from tqdm import tqdm
from solver import get_universal_neural_network
import numpy as np
from scipy.stats import multivariate_normal,norm
from tensorflow.keras import layers

class Env():
    def __init__(self,config,bsde): 
        self.net_config = config.net_config       
        self.bsde = bsde
        self.batch_size = config.net_config.batch_size
        self.current = 0 # an integer, the current time step
        self.y_init = tf.Variable(np.random.uniform(low=config.net_config.y_init_range[0],
                                                    high=config.net_config.y_init_range[1],
                                                    size=[1])) 
        self.timestamp = np.arange(0, self.bsde.num_time_interval+1) * self.bsde.delta_t
        self.dw, self.x = self.bsde.sample(self.batch_size)
#        self.t = tf.Variable(0.0,dtype=self.net_config.dtype)
        self.t = 0.0
        self.y = None
        lr_schedule = config.net_config.lr_schedule            
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)
        self.discount = np.exp(-self.bsde.rate*self.bsde.delta_t)

    def reset(self,training):
        if not training:
            self.batch_size=self.net_config.valid_size
        self.current = 0
        self.dw, self.x = self.bsde.sample(self.batch_size)
        self.y = None
#        self.t = tf.Variable(0.0,dtype=self.net_config.dtype)
        self.t = 0.0
        

    def set_y(self):
        self.y = self.y_init
    
    def penalty(self,x):
        if self.current == self.bsde.num_time_interval:
            return (self.y - self.bsde.g_tf(self.timestamp[self.current],x))**2
        else:
            return 0.0
    
    def get_state(self):
        return self.current, self.t, self.x[:,:,self.current]
    
    #@tf.function
    def act(self,z):
        self.y = self.y - self.bsde.delta_t*(self.bsde.f_tf(self.t,self.x[:,:,self.current],self.y,z)) + tf.reduce_sum(z*self.dw[:,:,self.current],1,keepdims=True)
        self.current =self.current+1
        self.t = self.timestamp[self.current]
        
    def act_and_update(self,z):
        self.act(z)
        return self.get_state()    


    @tf.function
    def train(self,grad):
        if self.current != self.bsde.num_time_interval:
            raise PermissionError             
        self.optimizer.apply_gradients(zip([grad], [self.y_init]))
    
    def get_discount(self):
        return self.discount
  

class Agent():
    def __init__(self,config,discount):
        self.net_config = config.net_config
        self.model = None
        self.z = None
        self.trace_discount = 0.5       
        lr_schedule = config.net_config.lr_schedule   
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)
        self.discount = discount

    def reset(self,training):
        self.z = [tf.Variable(0.0,dtype = self.net_config.dtype) for i in self.model.trainable_variables]
        if not training:
            self.batch_size = self.net_config.valid_size
    def loss_fn(self,t,x,action=None):
        raise NotImplementedError

    '''def grad(self, t,x,delta,action=None):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_fn(t,x,action)
        self.z = [z*self.trace_discount + grad for z,grad in zip(self.z,tape.gradient(loss, self.model.trainable_variables))]
        del tape
        return [(-delta) * z for z in self.z]'''   
   
    
    
    def train(self, t,x,delta,action=None):
        with tf.GradientTape(persistent=True) as tape:
            loss = -delta*self.loss_fn(t,x,action)
        grad=tape.gradient(loss, self.model.trainable_variables)
        del tape
        grad = self.grad(t,x,delta,action)        
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

    @tf.function
    def apply(self,t,x):
        if self.model:
            return self.model(tf.concat([tf.ones((x.shape[0],1),self.net_config.dtype)*t,x],axis=1))
        else:
            raise ValueError
    def call(self,t,x):
        raise  NotImplementedError



class Critic(Agent):
    def __init__(self, config,dim,discount):
        super().__init__(config,discount)
        self.model = get_critic_model(dim+1)
    
    def loss_fn(self, t, x,action):       
        return tf.reduce_mean(self.call(t,x))

    def call(self,t,x):
        return self.apply(t,x)
    
    @tf.function
    def train(self, t, x, delta, action=None):
        return super().train(t, x, delta, action=action)


class Actor(Agent):
    def __init__(self, config,dim,discount):
        super().__init__(config,discount)
        self.use_multivariate = False
        self.dim = dim
        self.model = get_actor_model(dim+1,self.use_multivariate)
        self.batch_size = config.net_config.batch_size
        self.I = 1.0
    def reset(self, training):
        super().reset(training)
        self.I = 1.0
        
    def loss_fn(self, t, x,action):
        output = self.apply(t,x)
        mu = output[:,:self.dim]
        if self.use_multivariate:
            sigma = tf.reshape(output[:,self.dim:],(-1,self.dim,self.dim))
            loss = -tf.reduce_mean(tf.linalg.logdet(sigma)
                - tf.tensordot(action-mu,tf.linalg.solve(sigma+1e-8*tf.eye(self.dim,dtype=self.net_config.dtype),tf.expand_dims(action-mu,-1)),[[1],[1]]))
        else:
            sigma = output[:,self.dim:]
            loss = -tf.reduce_mean(-tf.reduce_sum(sigma)-tf.square((action-mu)/sigma)/2,axis=1)
        self.I = self.I*self.discount
        return loss*self.discount
    
    
    def train(self, t, x, delta, action):        
        
        self.train_step(t,x,delta,action,tf.constant(self.I,self.net_config.dtype))
        self.I *= self.discount
    
    @tf.function
    def train_step(self,t, x, delta, action,I):        
        with tf.GradientTape(persistent=True) as tape:
            loss = -I*delta*self.loss_fn(t,x,action)
        grad=tape.gradient(loss, self.model.trainable_variables)
        del tape
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
    
    #@tf.function
    def call(self,t,x):
        output = self.apply(t,x)
        mu = output[:,:self.dim]
        if self.use_multivariate:
            sigma = tf.reshape(output[:,self.dim:],(-1,self.dim,self.dim))
            if self.dim==1:
                mu = tf.expand_dims(mu,axis=1)
                action = tf.stack([tf.concat([tf.random.normal([1],mu[i,j],sigma[i,j],dtype=self.net_config.dtype) 
                                                            for j in range(self.dim)],axis=0)
                                                                for i in range(self.batch_size)])
            return action
        else:
            sigma = tf.exp(output[:,self.dim:])
            if self.dim==1:
                mu = tf.expand_dims(mu,axis=1)
                sigma = tf.expand_dims(sigma,axis=1)
            action = np.array([multivariate_normal(mu[i],sigma[i],allow_singular=True).rvs() for i in range(self.batch_size)])
            #action = np.array([multivariate_normal(mu[i].numpy(),sigma[i].numpy(),allow_singular=True).rvs() for i in range(self.batch_size)])
            if len(action.shape)==1:
                action = np.expand_dims(action,axis=1)
            return tf.convert_to_tensor(action,dtype=self.net_config.dtype)

def get_critic_model(input_dim):
    input = layers.Input(shape=(input_dim,))
    x = layers.BatchNormalization()(input)    
    for i in range(4):
        x = layers.Dense(input_dim+10,'relu',False)(x)
        x = layers.BatchNormalization()(x)
    output = layers.Dense(1,'relu')(x)
    return tf.keras.Model(input,output)


def get_actor_model(input_dim,use_multivariate):
    dim = input_dim-1
    input = layers.Input(shape=(input_dim,))
    x = layers.BatchNormalization()(input)    
    for i in range(5):
        x = layers.Dense(input_dim+10,'relu',False)(x)
        x = layers.BatchNormalization()(x)
    if use_multivariate:
        output = layers.Dense(dim+dim*dim,'relu')(x)
    else:
        output = layers.Dense(2*dim,'relu')(x)
    return tf.keras.Model(input,output)


def episode(game,actor,critic,training=True):          
    game.reset(training)
    actor.reset(training)
    critic.reset(training)
    game.set_y()
    current,t,x = game.get_state()
    with tf.GradientTape(watch_accessed_variables=False)as tape:
        tape.watch(game.y_init)
        #for current in tqdm(range(game.bsde.num_time_interval-1)):
        while current < game.bsde.num_time_interval-1:                 
            action = actor.call(t,x)
            current,t1,x1 = game.act_and_update(action)            
            if training:
                delta = tf.reduce_mean(game.discount*critic.call(t1,x1)- critic.call(t,x) + game.penalty(x1))
                actor.train(tf.constant(t,dtype=game.net_config.dtype),tf.convert_to_tensor(x,dtype=game.net_config.dtype),delta,action)
                critic.train(tf.constant(t,dtype=game.net_config.dtype),tf.convert_to_tensor(x,dtype=game.net_config.dtype),delta)
            t,x = t1,x1     
             
        action = actor.call(t,x)
        current,t1,x1 = game.act_and_update(action)    

        loss = tf.reduce_mean(game.penalty(game.x[:,:,-1])) 
        loss = loss+ 1000*(tf.maximum(game.y_init[0]-game.net_config.y_init_range[1],0)+tf.maximum(game.net_config.y_init_range[0]-game.y_init[0],0))
       
    if training:
        delta = tf.reduce_mean(game.penalty(x1) - critic.call(t,x))
        actor.train(tf.constant(t,dtype=game.net_config.dtype),tf.convert_to_tensor(x,dtype=game.net_config.dtype),delta,action)
        critic.train(tf.constant(t,dtype=game.net_config.dtype),tf.convert_to_tensor(x,dtype=game.net_config.dtype),delta)                    
        game.train(tape.gradient(loss,game.y_init))        
    else:            
        return loss 