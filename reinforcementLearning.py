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
        self.t = 0.0
        self.y = None
        lr_schedule = config.net_config.lr_schedule            
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

    def reset(self):
        self.current = 0
        self.dw, self.x = self.bsde.sample(self.batch_size)
        self.y = None
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
    
    def act(self,z):

        self.y = self.y - self.bsde.delta_t*(self.bsde.f_tf(self.t,self.x[:,:,self.current],self.y,z)) + tf.reduce_sum(z*self.dw[:,:,self.current],1,keepdims=True)
        self.current +=1
        self.t = self.timestamp[self.current]
        
    def act_and_update(self,z):
        self.act(z)
        return self.get_state()    


    #@tf.function
    def train(self,grad):
        if self.current != self.bsde.num_time_interval:
            raise PermissionError             
        self.optimizer.apply_gradients(zip([grad], [self.y_init]))
  

class Agent():
    def __init__(self,config):
        self.net_config = config.net_config
        self.model = None
        self.z = 0
        self.gamma = 0.1       
        lr_schedule = config.net_config.lr_schedule          
            
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

    def reset(self):
        self.z = [0 for i in self.model.trainable_variables]
    def loss_fn(self,t,x,action=None):
        raise NotImplementedError

    def grad(self, t,x,delta,action=None):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_fn(t,x,action)
        self.z = [z*self.gamma + grad for z,grad in zip(self.z,tape.gradient(loss, self.model.trainable_variables))]
        del tape
        return [(-delta) * z for z in self.z]
    #@tf.function
    def train(self, t,x,delta,action=None):
        grad = self.grad(t,x,delta,action)        
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

    def apply(self,t,x):
        return self.model(tf.concat([tf.ones((x.shape[0],1),self.net_config.dtype)*t,x],axis=1))
    def call(self,t,x):
        raise  NotImplementedError



class Critic(Agent):
    def __init__(self, config,dim):
        super().__init__(config)
        self.model = get_critic_model(dim+1)
    
    def loss_fn(self, t, x,action):       
        return tf.reduce_mean(self.call(t,x))

    def call(self,t,x):
        return self.apply(t,x)


class Actor(Agent):
    def __init__(self, config,dim):
        super().__init__(config)
        self.dim = dim
        self.model = get_actor_model(dim+1)

    def loss_fn(self, t, x,action):
        output = self.apply(t,x)
        mu = output[:,:self.dim]
        sigma = output[:,self.dim:]
        loss = -tf.reduce_mean(tf.reduce_sum(tf.math.log(sigma)-tf.square((action-mu)/sigma)/2,axis=1))
        #sigma = tf.reshape(output[:,self.dim:],(-1,self.dim,self.dim))
        #loss = -tf.reduce_mean(tf.math.log(tf.linalg.trace(sigma)) 
        #        + tf.tensordot(action-mu,tf.linalg.solve(sigma+1e-8*tf.eye(self.dim,dtype=self.net_config.dtype),tf.expand_dims(action-mu,-1)),[[1],[1]]))
        
        return loss
    
    def call(self,t,x):
        output = self.apply(t,x)
        mu = output[:,self.dim]
        #sigma = tf.reshape(output[:,self.dim:],(-1,self.dim,self.dim))
        sigma = output[:,self.dim]
        #action = np.array([multivariate_normal(mu[i],sigma[i]+1e-8*tf.eye(self.dim,dtype=self.net_config.dtype)).rvs() for i,_ in enumerate(mu)])
        action = np.array([multivariate_normal(mu[i],sigma[i],allow_singular=True).rvs() for i,_ in enumerate(mu)])
        if self.dim==1:
            action = np.expand_dims(action,axis=1)
        return action

def get_critic_model(input_dim):
    input = layers.Input(shape=(input_dim,))
    x = layers.BatchNormalization()(input)    
    for i in range(3):
        x = layers.Dense(input_dim+10,'relu',False)(x)
        x = layers.BatchNormalization()(x)
    output = layers.Dense(1,'relu',False)(x)
    return tf.keras.Model(input,output)


def get_actor_model(input_dim):
    dim = input_dim-1
    input = layers.Input(shape=(input_dim,))
    x = layers.BatchNormalization()(input)    
    for i in range(3):
        x = layers.Dense(input_dim+10,'relu',False)(x)
        x = layers.BatchNormalization()(x)
    #output = layers.Dense(dim+dim*dim,'relu')(x)
    output = layers.Dense(2*dim,'relu')(x)
    return tf.keras.Model(input,output)

 
def episode(game,actor,critic,training=True):         
        game.reset()
        actor.reset()
        critic.reset()
        game.set_y()
        current,t,x = game.get_state()
        with tf.GradientTape(watch_accessed_variables=False)as tape:
            tape.watch(game.y_init)
            for current in range(game.bsde.num_time_interval-1):
            #while current < game.bsde.num_time_interval-1:            
                action = actor.call(t,x)
                current,t1,x1 = game.act_and_update(action)            
                if training:
                    delta = tf.reduce_mean(critic.call(t1,x1)- critic.call(t,x) + game.penalty(x1))
                    actor.train(t,x,delta,action)
                    critic.train(t,x,delta)
                t,x = t1,x1        
            
            action = actor.call(t,x)
            current,t1,x1 = game.act_and_update(action)     

            loss = tf.reduce_mean(game.penalty(game.x[:,:,-1])) 
            loss += 1000*(tf.maximum(game.y_init[0]-game.net_config.y_init_range[1],0)+np.maximum(game.net_config.y_init_range[0]-game.y_init[0],0))
            if training:
                delta = tf.reduce_mean(game.penalty(x1) - critic.call(t,x))
                actor.train(t,x,delta,action)
                critic.train(t,x,delta)                 
                game.train(tape.gradient(loss,game.y_init))
            else:            
                return tf.reduce_mean(game.penalty(game.x[:,:,-1]))
        
if __name__ == "__main__":
    no