#!/usr/bin/env python
# coding: utf-8

# In[239]:


import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


# In[228]:



# Percepton class
class Percepton:
    # constructor
    #   size - number of features
    #   lr - learning rate
    #   n-epochs - number epochs 
    def __init__(self, size, lr = 0.01, n_epochs = 1):
        self.size= size
        self.lr = lr
        self.n_epochs = n_epochs
        #self.weights = np.zeros((size,))  
        # Start with the inital weights 
        self.weights = np.array([-1.5,1,1]) 
    
    # forward()
    def forward(self, x):
        # Matrix multiplication: input is multiplied by weights to get the dot product
        z = x @ self.weights
        print("sum for",x, z)
        y = np.sign(z)
        #print("y:",y)
        return y
    
    # update
    # update weights
    #.  x - input
    #.  y is network prediction
    #   d is target 
    def update(self, x, y, d):
        delta =  (d - y)
        #print("delta: ", delta)
        new_weights = self.weights + self.lr * delta * x
    
        self.weights = new_weights
        print("Weights: ", self.weights)        
    
    # main training loop
    def train(self, X,Y):
        
        for t in range(self.n_epochs):
            print("epoch ", t)
            
            correct = 0
            n_samples = X.shape[0]
            for i in range(n_samples):
                x = X[i,:]
                d = Y[i]
                y = self.forward(x)
                
                self.update(x,y,d)
                
                if y == d:
                    correct +=1
            
            print("Accuracy: " , correct / n_samples)
            
            if correct == n_samples:
                print("All correct, done")
                break
                
        print("Complete")    
            
            
        


# In[229]:


X = np.array([[ -1,0, 0 ], 
              [ -1,0, 1 ],
              [ -1,1, 0 ],
              [ -1,1, 1]])
Yand = np.array([-1,-1,-1,1])
Yor = np.array([-1,1,1,1])

print(X,Yand)


# In[230]:


p =Percepton(3, lr = 0.3, n_epochs = 800)
# p.train(X,Yand)
p.train(X,Yor)


# In[231]:


a = np.array([1,1])
b = np.array([2,2])
print(a.shape)
a @ b


# # Part 2

# In[265]:


# Mutli layer percepton with 2 hidden layers, 2 neurons each


class MLP:
    np.random.seed(48)
    
    # At construction provide learning rate, number of epochs and activation function
    def __init__(self, sizes, weights1= None, weights2 = None, weights3 = None,
                 lr = 0.01, n_epochs = 1, activation_fun = "relu", verbose = True ):
        print (f"LR = {lr}, n_epochs = {n_epochs}")
        self.n_epochs = n_epochs
        self.lr = lr
        
        self.n_inputs = sizes[0]
        print("INPUTS", self.n_inputs)
        self.n_hidden1 = sizes[1]
        print("No. of hidden 1 =", self.n_hidden1)
        self.n_hidden2 = sizes[2]
        print("No. of hidden 2 =", self.n_hidden2)
        self.n_out = sizes[3]
        print("No. of output =", self.n_out)
        # initialize random weights 
        # we do + 1 to account for b
        self.weights1 = weights1
        if weights1 is None:
            self.weights1 = np.random.rand(self.n_inputs  + 1, self.n_hidden1) 
        
        self.weights2 = weights2
        if weights2 is None:
            self.weights2 = np.random.rand(self.n_hidden1 + 1, self.n_hidden2)
        
        self.weights3 = weights3
        if weights3 is None:
            self.weights3 = np.random.rand(self.n_hidden2 + 1, self.n_out)    
         
        if verbose:
            print("weights1 ", self.weights1)
            print("weights2 ", self.weights2)
            print("weights3 ", self.weights3)
        
        # set the activation function
        self.activation_fun = activation_fun
        self.verbose = verbose
        
    #supported activation functions:
    #  relu
    #  sigmoid
    #  binary step
    def activation(self, z):
        if self.activation_fun == "relu":
            return np.maximum(0, z)
            
        if self.activation_fun == "sigmoid":
            return 1.0/ ( 1.0 + np.exp(-z))
        
        if self.activation_fun == "binary_step":
            if np.any(z < 0):
                return np.zeros(z.shape)
            else:
                return np.ones(z.shape)
        
    # forward pass with sample set X
    def forward(self, x):
        # Hidden layer 1
        z1 = np.dot(x, self.weights1)
        if self.verbose:
            print("z1 : ", z1)
        self.hidden1 = self.activation(z1)
        if self.verbose:
            print("Hidden1: ",self.hidden1)
        self.hidden1 = np.concatenate((self.hidden1, self.b2), axis = 1 )
        if self.verbose:
            print("Hidden1: ",self.hidden1)
        
        # Hidden layer 2
        z2 = np.dot(self.hidden1, self.weights2)
        if self.verbose:
            print("Z2: ", z2)
        self.hidden2 = self.activation(z2)
        self.hidden2 = np.concatenate((self.hidden2, self.b3), axis = 1 )
        if self.verbose:
            print("Hidden2: ",self.hidden2)
        
        #Output layer
        z3 = np.dot(self.hidden2, self.weights3)
        if self.verbose:
            print("z3:", z3)  
        y = self.activation(z3)
        if self.verbose:
            print("y:",y)
        
        return y#.reshape((4,))
      
    # Back propogation and gradient decent
    def update(self, X, y, targets):
        error = 0.5 * np.sum(y - targets) ** 2
        if self.verbose:
            print("y ", y)
            print("targets ", targets)
        
        delta3 = (y - targets) * y
        if self.verbose:
            print("Delta 3 shape ", delta3.shape)
            print("Hidden 2 shape: ",self.hidden2.shape)
            print("Weights 3 shape: ",self.weights3.shape)
        
        h = self.hidden2 * (1.0 - self.hidden2)
        dw = np.dot( delta3, np.transpose ( self.weights3 ))
        delta2 = h * dw 
        
        if self.verbose:
            print("Delta 2 " , delta2.shape)
            print("Hidden 1 ", self.hidden1.shape)
        
        delta1 = self.hidden1 * (1.0 - self.hidden1) * np.dot(delta2[:,:-1],np.transpose(self.weights2))
        
        if self.verbose:
            print ("Delta 1 ", delta1)
        
        # updates to be applied to the weights
        update_weights1 = self.lr * np.dot(np.transpose(delta1),X)
        update_weights2 = self.lr * np.dot(np.transpose(self.hidden1) , delta2)
        update_weights3 = self.lr * np.dot(np.transpose(self.hidden2),  delta3)
        
        if self.verbose:
            print("Update weights1 ", update_weights1)
            print("Update weights2 ", update_weights2)
            print("Update weights3 ", update_weights3)
            #print("New weights1 ", update_weights1.shape)
            #print("New weights2 ", update_weights2.shape)
        
        # update the weights
        # Applying the update to the weights 
        self.weights1 -= update_weights1[:,:]
        self.weights2 -= update_weights2[:,:-1]
        self.weights3 -= update_weights3
        
        if self.verbose:
            print("Adjusted weights1", self.weights1)
            print("Adjusted weights2", self.weights2)
            print("Adjusted weights3", self.weights3)
      
        #print("Weights 1 ", self.weights1)
        
    # Evaluate the netowrk prediction accuracy
    def evaluate(self, y, targets):
        y = y.reshape(targets.shape)
        
        error = 0.5 * np.sum(y - targets) ** 2
        print("ERROR: ", error)
        equal = np.sum( y == targets)
        #print(y, targets)
        #print("equal ", equal)
        #print("Accuracy: ", equal / y.shape[0])
        
        return error
        
    def pre_train(self, X):
         
        n_samples = X.shape[0]
         # biases
        self.b1 = -1 * np.ones((n_samples, 1))
        if self.verbose:
            print("bias 1:",self.b1)
        self.b2 = -1 * np.ones((n_samples, 1))
        self.b3 = -1 * np.ones((n_samples, 1))
            
        #print("B shape: ", self.b1.shape)
        
        # Add bias to the input
        inputs = np.concatenate((X, self.b1), axis = 1)
        if self.verbose:
            print("inputs", inputs)
        
        return inputs
    
    # Main training loop, run forward/backward passes for N times
    def train(self, X, targets):
        #print("X: ",X.shape)
        #print("targets: ", targets.shape)
       
        inputs = self.pre_train(X)
        
        for i in range(self.n_epochs):
            print("Epoch ", i)
            
            #1) forward
            y = self.forward(inputs)
            
            #2) evaluate on training set
            error = self.evaluate(y, targets)
            
            #3) backward propagation
            self.update(X,y, targets)
            
        return error    


# In[266]:


mlp = MLP(sizes = (2,2,2,1), lr = 0.05, n_epochs =100 , activation_fun = "relu")


# In[267]:


X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
Yxor = np.transpose(np.array([0,1,1,0])).reshape((4,1))

mlp.train(X, Yxor)


# This network solvers the regression problem , to convert the ouput to classification solution 
#   we need to use softmax activation
# Regression : Fitting Error
# Classification: Accuracy

# # PArt 3

# In[83]:


from geneticalgorithm import geneticalgorithm as ga 


# In[108]:


get_ipython().run_cell_magic('time', '', '\ndef fun(lr):\n    mlp = MLP(sizes = (2,2,2,1), lr = lr, n_epochs =5 , activation_fun = "sigmoid", verbose= False)\n    error = mlp.train(X, Yxor)\n    return error\n\nga_params = {\n    "max_num_iteration": 1,\n    "population_size":5,\n    "mutation_probability":0.1,\n    "elit_ratio":0.01,\n    "crossover_probability":0.5,\n    "parents_portion":0.5,\n    "crossover_type":"uniform", # "one_point"\n    "max_iteration_without_improv": 3\n   \n}\n# Minimize the fitting error\nvarbound = np.array([[0.0001, 0.1]])\nmodel = ga(function = fun, \n           dimension = 1, \n           variable_type = "real", \n           variable_boundaries = varbound,\n           algorithm_parameters = ga_params)\n\nres = model.run()\n#print("REs:",res)')


# In[294]:


get_ipython().run_cell_magic('time', '', '\n# weights 1: 3 x 2 \n# weights 2: 3 x 2\n# weights 3: 3 x 1\n# total = 6 + 6 + 3 = 15\nDIM = 15\ndef fun(all_weights):\n    print(all_weights.shape)\n    \n    weights1 = all_weights[:6,].reshape((3,2))\n    weights2 = all_weights[6:12,].reshape((3,2))\n    weights3 = all_weights[12:15,].reshape((3,1))\n    \n    mlp = MLP(sizes = (2,2,2,1),  \n              weights1 = weights1, weights2 = weights2, weights3 = weights3,\n              activation_fun = "sigmoid", \n              verbose= False)\n    \n    inputs = mlp.pre_train(X)\n    y = mlp.forward(inputs)\n    error = mlp.evaluate(y, Yxor)\n    return error\n\nga_params = {\n    "max_num_iteration": 20,\n    "population_size":200,\n    "mutation_probability":0.1,\n    "elit_ratio":0.01,\n    "crossover_probability":0.5,\n    "parents_portion":0.5,\n    "crossover_type":"uniform", # "one_point"\n    "max_iteration_without_improv": 10\n   \n}\n# Minimize the fitting error\nvarbound = np.array([[-2, 2] for i in range(DIM) ])\nprint(varbound)\nmodel = ga(function = fun, \n           dimension = DIM, \n           variable_type = "real", \n           variable_boundaries = varbound,\n           algorithm_parameters = ga_params)\n\nres = model.run()\n#print("REs:",res)')


# # Partcile Swarm Optimization (PSO)

# In[273]:


get_ipython().run_cell_magic('time', '', 'import pyswarms as ps\nfrom pyswarms.utils.functions import single_obj as fx\n\n\ndef fun_for_pso(lrs):\n    errors = []\n    for lr in lrs:\n        mlp = MLP(sizes = (2,2,2,1), lr = lr, n_epochs =2 , activation_fun = "sigmoid", verbose= False)\n        error = mlp.train(X, Yxor)\n        errors.append(error)\n        \n    return errors\n\nn_dims = 1\n# c1 and c2 are accelerate constants\noptions = {"c1":0.5, # Cognitive parameter , how much confidence the particle has in itself\n           "c2":0.3, # social parameter  , how much confidence the particle has in it neigbours\n           "w":0.9   # inertia parameter, describes how the previous velocity influences the current velocity\n          }\n\nmin_bounds = np.array([0.0001])\nmax_bounds = np.array([0.1])\nbounds = (min_bounds, max_bounds)\n\noptimizer = ps.single.GlobalBestPSO(n_particles = 3, dimensions=1, options = options , bounds = bounds )\ncost, pos  = optimizer.optimize(fun_for_pso ,1)\nprint("Best Error ", cost,"achieved with LR = ", pos)')


# In[ ]:





# # Optimize the weights of MLP 

# In[302]:


get_ipython().run_cell_magic('time', '', 'import pyswarms as ps\nfrom pyswarms.utils.functions import single_obj as fx\n\nDIMS = 15\ndef one_run(all_weights):\n    print(all_weights.shape)\n    \n    weights1 = all_weights[:6,].reshape((3,2))\n    weights2 = all_weights[6:12,].reshape((3,2))\n    weights3 = all_weights[12:15,].reshape((3,1))\n    \n    mlp = MLP(sizes = (2,2,2,1),  \n              weights1 = weights1, weights2 = weights2, weights3 = weights3,\n              activation_fun = "sigmoid", \n              verbose= False)\n    \n    inputs = mlp.pre_train(X)\n    y = mlp.forward(inputs)\n    error = mlp.evaluate(y, Yxor)\n    return error\n\naverage_errors = []\n#Objective function for PSO\ndef fun_for_pso(many_params):\n    errors = []\n    for weights in many_params:\n       \n        error = one_run(weights)\n        \n        errors.append(error)\n    \n    average_errors.append(np.mean(errors))\n    \n    return errors\n\n\n# c1 and c2 are accelerate constants\noptions = {"c1":0.5, # Cognitive parameter , how much confidence the particle has in itself\n           "c2":0.1, # Social parameter  , how much confidence the particle has in it neigbours\n           "w":0.9   # Inertia parameter, describes how the previous velocity influences the current velocity\n          }\n\nmin_bounds = np.array([-2] * DIMS)\nmax_bounds = np.array([2] * DIMS)\nbounds = (min_bounds, max_bounds)\n\noptimizer = ps.single.GlobalBestPSO(n_particles = 250, dimensions=DIMS, options = options , bounds = bounds )\ncost, pos  = optimizer.optimize(fun_for_pso ,50)\nprint("Best Error ", cost,"achieved with WEIGHTS = ", pos)\n\nx_axis = np.arange(len(average_errors))\n\nplt.plot(x_axis, average_errors)')


# In[ ]:




