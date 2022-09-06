"""
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
Simple Feedforward Neural Network with Learned Orthogonality on MNIST

(C) 2019 Nikolay Manchev

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 
International License.

This code implements a simple feedforward ANN with the orthogonalisation 
techniques described in Manchev, N. and Spratling, M., "Solving gradient 
instability in deep neural networks with learned orthogonality"

-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

This network uses the mnist: Python utilities to download and parse the MNIST 
dataset. These are licensed under BSD 3-Clause "New" or "Revised" License
Copyright 2016 Marc Garcia <garcia.marc@gmail.com> and available at

https://github.com/datapythonista/mnist

-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

The update functions from Lasagne are distributed under the MIT Licensee
Copyright (c) 2014-2015 Lasagne contributors

Lasagne uses a shared copyright model: each contributor holds copyright over
their contributions to Lasagne. The project versioning records all such
contribution and copyright details.
By contributing to the Lasagne repository through pull-request, comment,
or otherwise, the contributor releases their content to the license and
copyright terms herein.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
"""

import mnist
import theano
import argparse

import theano.tensor as T
import numpy as np

from sklearn import preprocessing
from theano import function

#theano.config.optimizer="fast_compile"

def normal(shape, rng, variable_name, stdev=0.001):
    """
    Samples the standard normal distribution
    
    Parameters
    ----------
    shape         : dimensions of the returned array
    rng           : numpy.random.RandomState instance
    variable_name : name for the shared variable    
    stdev         : standard deviation of the distribution.
        
    Returns
    -------
    A shared variable with amples from the parameterized normal distribution.
    """
    return theano.shared(rng.normal(loc=0, scale=stdev, size=shape), 
                         variable_name)

def uniform(shape, rng, variable_name, stdev = 0.001):
    """
    Samples a uniform distribution
    
    Parameters
    ----------
    shape        : dimensions of the returned array
    rng          : numpy.random.RandomState instance
    stdev        : standard deviation of the distribution.
        
    Returns
    -------
    A shared variable with amples from the parameterized normal distribution.
    """    
    return theano.shared(rng.uniform(low=-1.0*stdev, high=stdev, size=shape),
                         variable_name)

def rand_ortho(shape, rng, variable_name, stdev = 0.001, alpha_p = 0.1, 
               Ep = 0.000001, max_it = 100) : 
    """
    Uses pre-training to orthogonalise a matrix. The matrix is initially 
    populated using samples from N(0, stdev=0.001).
        
    Parameters
    ----------
    shape         : shape of the matrix
    rng           : numpy.random.RandomState instance
    variable_name : name for the shared variable
    stdev         : standard deviation of the distribution.
    alpha_p       : pre-training learning rate
    Ep            : convergence criterion for the pre-training (Err < Ep)
    max_it        : maximum number of iterations in the pre-training
        
    Returns
    -------
    An orthognal/semi-orthogonal matrix learned using pre-training
    """        
    rows,cols = shape

    W = normal((rows,cols), rng, variable_name, stdev)
    
    if rows < cols:
      # rows must be orthonormal vectors
      identity = np.identity(rows)
      cost = T.sqrt(((T.dot(W, W.T) - identity)**2).sum())**2
    else:
      # columns must be orthonormal vectors
      identity = np.identity(cols)
      cost = T.sqrt(((T.dot(W.T, W) - identity)**2).sum())**2
                
    err_fn = function([],
                      cost,
                      updates = {(W, W - alpha_p * T.grad(cost, W))})

    converged = False
    i = 1

    while ((not converged) and (i <= max_it)):
        err = err_fn()
        if err < Ep:
            print("Pre-traiing for %s completed in %i steps." % 
                  (variable_name,i))
            converged = True
        i+=1
        
    assert converged, "Something went wrong. Pre-training didn't converged."

    return W  

def synaptic_init(shape, init_type, stdev, rng, variable_name):
    """
    Populates a matrics with given shape and sampled using selected init scheme
        
    Parameters
    ----------
    shape         : shape of the matrix
    init_type     : initialisation scheme (normal, uniform, oinit)
    rng           : numpy.random.RandomState instance
    variable_name : name for the shared variable
        
    Returns
    -------
    A Theano shared variable containg the initialised matrix
    """     
    assert init_type in ["normal", "uniform", "oinit"], "Unknown init type."
    
    if (init_type == "normal"):
        synaptic_matrix = normal(shape, rng, variable_name, stdev)
    elif (init_type == "uniform"):
        synaptic_matrix = uniform(shape, rng, variable_name, stdev)
    else:
        synaptic_matrix = rand_ortho(shape, rng, variable_name)

    return synaptic_matrix

def loadMNIST(rescaling=False):
    """
    Loads the MNIST data set (training + test)
        
    Parameters
    ----------
    rescaling     : if set to True centers the data to the mean and 
                    applies component-wise scaling to unit variance.
        
    Returns
    -------
    The MNIST train and test sets as shared variables
    """    
    X_train = mnist.train_images().reshape(60000,784)
    y_train = mnist.train_labels()
       
    X_test = mnist.test_images().reshape(10000,784)
    y_test = mnist.test_labels()    

    if (rescaling):
        X_train = preprocessing.scale(X_train.astype("float64"))
        X_test  = preprocessing.scale(X_test.astype("float64"))

    X_train = theano.shared(X_train, borrow=True)
    y_train = theano.shared(y_train, borrow=True) 
    X_test  = theano.shared(X_test, borrow=True) 
    y_test  = theano.shared(y_test, borrow=True)

    return X_train, T.cast(y_train, "int32"), X_test, T.cast(y_test, "int32")

def fit(lr, X_train, y_train, X_test, y_test, batch_size, n_hid, layers, activ, 
        maxepochs, init, stdev, gd_opt, penalty, chk_interval, rng):    
    """
    Trains a simple feedforward network on the MNIST dataset
        
    Parameters
    ----------
    lr            : learning rate (alpha)
    X_train       : MNIST training set samples
    y_train       : MNIST training set class labels
    X_test        : MNIST test set samples
    y_test        : MNIST test set class labels
    batch_size    : number of samples in a mini-batch (training set)
    n_hid         : number of neurons in the hidden layer
    layers        : number of hidden layers
    activ         : activation function    
    maxepochs     : number of training epochs
    init          : weight initialisation (xavier/oinit)
    stdev         : standard deviation of the weights distribution
    gd_opt        : weight update method (vanilla, rmsprop, adadelta, nesterov)
    penalty       : orthogonal penalty strength (lambda)
    chk_interval  : number of iterations between netwrok evaluation
    rng           : numpy.random.RandomState instance
    
    Returns
    -------
    Best accuracy achieved by the network on the test set
    """           
    print("------------------------------------------------------")
    print("******************************************************")
    print("Parameters - Simple feedforward ANN on MNIST")
    print("******************************************************")
    print("optimization   : %s" % gd_opt)
    print("learning_rate  : %.10f" % lr)
    print("maxepoch       : %i" % maxepochs)
    print("batch_size     : %i" % batch_size)
    print("hidden layers  : %i" % layers)
    print("n_hid          : %i" % n_hid)
    print("init           : %s" % init)
    print("stdev          : %f" % stdev)
    print("ortho penalty  : %f" % penalty)    
    print("******************************************************")    
    
    
    n_inp = X_train.get_value(borrow=True).shape[1]
    n_out = len(np.unique(y_train.eval()))
        
    n_train_batches = X_train.get_value(borrow=True).shape[0] // batch_size

    X = T.matrix()
    y = T.ivector()
    index = T.lscalar()  # minibatch index

    W = []
    B = []
    
    # Initialise all synaptic weights
    W.append(synaptic_init((n_inp, n_hid),init,stdev,rng, "W_input"))
    B.append(theano.shared(np.zeros((n_hid,))))
    
    for i in range(layers):
        W.append(synaptic_init((n_hid, n_hid),init,stdev,rng, "W"+str(i+1)))
        B.append(theano.shared(np.zeros((n_hid,))))

    W.append(synaptic_init((n_hid, n_out),init,stdev,rng, "W_output"))
    B.append(theano.shared(np.zeros((n_out,))))

    print("******************************************************")    
        
    # Compute layer-wise outputs
    last_layer = len(W)-1
    
    H = []
    H.append(activ(T.dot(X, W[0]) + B[0]))
    
    for i in range(last_layer-1):
        H.append(activ(T.dot(H[i], W[i+1]) + B[i+1]))
  
    # Compute network output
    P  = T.nnet.softmax(T.dot(H[last_layer-1], W[last_layer]) + B[last_layer])
       
    # Global error
    cost = -T.mean(T.log(P)[T.arange(y.shape[0]), y])

    y_pred = T.argmax(P, axis=1)
    err = T.mean(T.neq(y_pred, y))
    
    # Compute gradients
    dW, dB = [],[]
    updates = []
    
    norm = 0
        
    for i in range(len(W)):
        d_W,d_B = T.grad(cost,[W[i],B[i]])

        # Apply orthogonal penalty (lambda)
        if (penalty != 0):
            if (i==0):
              identity = np.identity(n_inp)
            else:
              identity = np.identity(n_hid)  
            
            d_W = d_W + penalty * T.grad(((T.dot(W[i], W[i].T) - identity)**2).sum(), W[i])

        
        dW.append(d_W)
        dB.append(d_B)
        
        updates.append((W[i], W[i] - lr*d_W))
        updates.append((B[i], B[i] - lr*d_B))
        
        norm += (d_W**2).sum() #+ (d_B**2).sum()

    # Compute weights norm
    norm = T.sqrt(norm)

    # Define training and evaluation functions 
    train = theano.function(
             inputs = [index],
             outputs = (cost, err, norm),
             updates = updates,
             givens = { X: X_train[index * batch_size: (index + 1) * batch_size],
                        y: y_train[index * batch_size: (index + 1) * batch_size] }
            )
    

    eval_step = theano.function(
                  inputs = [],
                  outputs = [cost,err],
                  givens={
                    X: X_test,
                    y: y_test
                  }             
             )

    # Compute starting error
    valid_cost, error = eval_step()
    print("Starting accuracy : %.2f" % (1 - error))
    print("------------------------")
    
    epoch = 1
        
    best_acc = 0

    training = True
        
    acc = []

    cost_avg, err_avg, norm_avg = 0, 0, 0
    
    while(training and epoch <= maxepochs):
       
        for minibatch_index in range(n_train_batches):
             
            # Train and update weights            
            cost, err, norm = train(minibatch_index)
            
            cost_avg += cost
            err_avg  += err
            norm_avg += norm
           
            samples_seen = ((epoch - 1) * n_train_batches + (minibatch_index+1)) * batch_size   

            # Time to evaluate?
            if (samples_seen % chk_interval == 0):
                
                if minibatch_index != 0:
                  err_avg  = err_avg  / chk_interval
                  cost_avg = cost_avg / chk_interval
                  norm_avg = norm_avg / chk_interval
                                
                valid_cost, error = eval_step() 
                
                acc.append((1.0 - error)*100)

                if acc[-1] > best_acc:
                    best_acc = acc[-1]                
                
                #print(show())
                
                print("Epoch %d" % epoch, ":", \
                      "Samples %07d" % samples_seen, ":", \
                      "cost %05.3f, " % cost_avg, \
                      "|dW| %7.4f, " % norm_avg, \
                      "val err %05.2f%%" % (error*100), ":", \
                      "best accuracy %05.2f%%" % best_acc)
 
                cost_avg, err_avg, norm_avg = 0, 0, 0
                      
        epoch+=1

    return best_acc

def main():
    
    parser = argparse.ArgumentParser(description="Feedforward ANN trained on \
                                     MNIST\nThis work is licensed under the \
                                     Creative Commons Attribution 4.0 \
                                     International License.")

    parser.add_argument("--lr",help="Learning rate", default = 1e-6, 
                        required=False, type=float)

    parser.add_argument("--hidden",help="Number of units in each hidden layer", 
                        default = 100, required=False, type=int)

    parser.add_argument("--init", help="Weight initialization and activation \
                        function", choices=["normal", "identity", "oinit"],
                        default = "oinit", required=False)

    parser.add_argument("--batchsize",help="Size of the minibatch", 
                        default = 20, required=False, type=int)

    parser.add_argument("--maxepochs",help="Maximum number of training epochs", 
                        default = 200, required=False, type=int)

    parser.add_argument("--penalty",help="Orthogonal penalty", default = 0, 
                        required=False, type=float)

    parser.add_argument("--opt", help="Optimizer", choices=["nesterov", 
                                                            "vanilla", 
                                                            "adadelta", 
                                                            "rmsprop"],
                        default = "vanilla", required=False)

    parser.add_argument("--stdev",help="Standard deviation for the random \
                        matrices", default = 0.001, 
                        required=False, type=float)

    parser.add_argument("--layers",help="Number of hidden layers", 
                        default = 10, required=False, type=int)

    parser.add_argument("--chk",help="Check interval", 
                        default = 10000, required=False, type=int)


    args = parser.parse_args()

    learning_rate = args.lr
    
    n_hid = args.hidden
    
    init = args.init
    
    activ = T.tanh

    batch_size = args.batchsize
    
    maxepochs = args.maxepochs
       
    gd_opt = args.opt
       
    penalty = args.penalty
    
    stdev = args.stdev
    
    layers = args.layers
    
    check_interval = args.chk

    rng = np.random.RandomState(1234)  
    
    X_train, y_train, X_test, y_test = loadMNIST(rescaling=True)
        
    fit(learning_rate, X_train, y_train, X_test, y_test, batch_size, n_hid, 
        layers, activ, maxepochs, init, stdev, gd_opt, penalty, check_interval,
        rng)

if __name__ == "__main__":
    main()