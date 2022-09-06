"""
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
Simple Recurrent Neural Network with Learned Orthogonality on MNIST

(C) 2019 Nikolay Manchev

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 
International License.

This code implements an SRRN with the orthogonalisation techniques described in
Manchev, N. and Spratling, M., "Solving gradient instability in deep neural 
networks with learned orthogonality"

-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

The implementation is based on the recurrent neural network implementation from
"On the difficulty to training recurrent neural networks", Razvan Pascanu, 
Tomas Mikolov, Yoshua Bengio, available at 

https://github.com/pascanur/trainingRNNs

This network also uses the mnist: Python utilities to download and parse the 
MNIST dataset. These are licensed under BSD 3-Clause "New" or "Revised" License
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

import numpy         as np
import theano.tensor as T
import theano
import argparse
import sys
import mnist
import math

from collections   import OrderedDict
from theano        import function
from sklearn       import preprocessing

#theano.config.floatX="float32"
#theano.config.optimizer="fast_compile"

def vanilla_sgd(params, grads, learning_rate):
    """
    Update rules for vanilla SGD. Based on the update functions from
    Lasagne (https://github.com/Lasagne/Lasagne)
    
    The update is computed as
    
        param := param - learning_rate * gradient
        
    Parameters
    ----------
    params        : list of shared varaibles that will be updated
    grads         : list of symbolic expressions that produce the gradients
    learning_rate : step size
    
    Returns
    -------
    A dictionary mapping each parameter in params to their update expression    
    """
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad

    return updates

def adadelta(params, grads, learning_rate, rho=0.95, epsilon=1e-6):
    """
    Update rules for Adadelta. Based on the update functions from
    Lasagne (https://github.com/Lasagne/Lasagne)
                    
    Parameters
    ----------
    params        : list of shared varaibles that will be updated
    grads         : list of symbolic expressions that produce the gradients
    learning_rate : step size
    rho           : squared gradient moving average decay factor
    epsilon       : small value added for numerical stability
    
    Returns
    -------
    A dictionary mapping each parameter in params to their update expression    
    """    
    updates = OrderedDict()
    
    one = T.constant(1)
    
    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)      
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype), 
                             broadcastable=param.broadcastable)
        delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype), 
                                   broadcastable=param.broadcastable)
        
        # update accu (as in rmsprop)
        accu_new = rho * accu + (one - rho) * grad ** 2
        updates[accu] = accu_new

        # compute parameter update, using the 'old' delta_accu
        update = (grad * T.sqrt(delta_accu + epsilon) /
                  T.sqrt(accu_new + epsilon))
        updates[param] = param - learning_rate * update

        # update delta_accu (as accu, but accumulating updates)
        delta_accu_new = rho * delta_accu + (one - rho) * update ** 2
        updates[delta_accu] = delta_accu_new        
        
    return updates

def nesterov_momentum(params, grads, learning_rate, momentum=0.9):
    """
    Update rules for Nesterov's accelerated SGD. Based on the update functions 
    from Lasagne (https://github.com/Lasagne/Lasagne)

    Parameters
    ----------
    params        : list of shared varaibles that will be updated
    grads         : list of symbolic expressions that produce the gradients
    learning_rate : step size
    momentum      : amount of momentum to apply
        
    Returns
    -------
    A dictionary mapping each parameter in params to their update expression    
    """    
    updates = vanilla_sgd(params, grads, learning_rate)
    
    for param in params:
      value = param.get_value(borrow=True)
      velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
      x = momentum * velocity + updates[param] - param
      updates[velocity] = x
      updates[param] = momentum * x + updates[param]

    return updates

def rmsprop(params, grads, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    """
    Update rules for RMSProp. Based on the update functions from Lasagne 
    (https://github.com/Lasagne/Lasagne)

    Parameters
    ----------
    params        : list of shared varaibles that will be updated
    grads         : list of symbolic expressions that produce the gradients
    learning_rate : step size
    rho           : gradient moving average decay factor
    epsilon       : small value added for numerical stability   
        
    Returns
    -------
    A dictionary mapping each parameter in params to their update expression 
    """
    updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = rho * accu + (one - rho) * grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))

    return updates

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


def loadMNIST(sample_train=0, sample_test=0, rescaling=False):
    """
    Loads the MNIST data set (training + test)
        
    Parameters
    ----------
    sample_train  : how many samples to take from the training set (0 disables
                    sampling and uses the whole training set instead)
    sample_test   : how many samples to take from the test set (0 disables
                    sampling and uses the whole test set instead)
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

    if (sample_train != 0) and (sample_test != 0):
        
        print("Elements in train : %i" % sample_train)
        print("Elements in test  : %i" % sample_test)
        
        idx_train = np.random.choice(np.arange(len(X_train)), sample_train, 
                                     replace=False)
        idx_test = np.random.choice(np.arange(len(X_test)), sample_test, 
                                    replace=False)
        
        X_train = X_train[idx_train]
        y_train = y_train[idx_train]

        X_test = X_test[idx_test]
        y_test = y_test[idx_test]


    if (rescaling):
        X_train = preprocessing.scale(X_train.astype("float64"))
        X_test  = preprocessing.scale(X_test.astype("float64"))

    # Swap axes
    X_train = np.swapaxes(np.expand_dims(X_train,axis=0),0,2)
    X_test = np.swapaxes(np.expand_dims(X_test,axis=0),0,2)

    onehot_encoder = preprocessing.OneHotEncoder(sparse=False)        
    y_train = onehot_encoder.fit_transform(y_train.reshape(-1,1))
    y_test = onehot_encoder.fit_transform(y_test.reshape(-1,1))

    X_train = theano.shared(X_train, borrow=True)
    y_train = theano.shared(y_train, borrow=True) 
    X_test  = theano.shared(X_test, borrow=True) 
    y_test  = theano.shared(y_test, borrow=True)

    return X_train, T.cast(y_train, "int32"), X_test, T.cast(y_test, "int32")

def fit(X_train, y_train, X_test, y_test, learning_rate, n_hid, init, 
        batch_size, maxepoch, gd_opt, rng, penalty, stdev, check_interval):
    """
    Trains an SRNN on the MNIST dataset (sequence of pixels tasks)
        
    Parameters
    ----------
    X_train        : MNIST training set samples
    y_train        : MNIST training set class labels
    X_test         : MNIST test set samples
    y_test         : MNIST test set class labels
    learning_rate  : learning rate (alpha)
    n_hid          : number of neurons in the hidden layer
    init           : weight initialisation (normal/uniform/oinit)
    batch_size     : number of samples in a mini-batch (training set)
    maxepoch       : number of training epochs
    gd_opt         : weight update method (vanilla,rmsprop,adadelta,nesterov)
    rng            : numpy.random.RandomState instance
    penalty        : orthogonal penalty strength (lambda)
    stdev          : standard deviation of the weights distribution
    check_interval : number of iterations between netwrok evaluation
    """         
    print("------------------------------------------------------")
    print("******************************************************")
    print("Parameters - SRNN on MNIST (sequence of pixels)")
    print("******************************************************")
    print("optimization       : %s" % gd_opt)
    print("learning_rate      : %f" % learning_rate)
    print("maxepoch           : %i" % maxepoch)
    print("batch_size         : %i" % batch_size)
    print("check_interval     : %i" % check_interval)
    print("n_hid              : %i" % n_hid)
    print("init               : %s" % init)
    print("stdev              : %f" % stdev)
    print("orthogonal penalty : %f" % penalty)        
    print("******************************************************") 
        
    # Number of inputs and outputs for the SRNN
    n_inp = 1
    n_out = 10
    
    activ = T.tanh

    # Initialise the synaptic weights
    if init == "normal":
        
        Wxh = normal((n_inp, n_hid),rng,"Wxh",stdev)
        Whh = normal((n_hid, n_hid),rng,"Whh",stdev)
        Why = normal((n_hid, n_out),rng,"Why",stdev)

    elif init == "uniform":

        Wxh = uniform((n_inp, n_hid),rng,"Wxh",stdev)
        Whh = uniform((n_hid, n_hid),rng,"Whh",stdev)
        Why = uniform((n_hid, n_out),rng,"Why",stdev)

    elif init == "oinit":

        Wxh = rand_ortho((n_inp, n_hid),rng,"Wxh",stdev)
        Whh = rand_ortho((n_hid, n_hid),rng,"Whh",stdev)
        Why = rand_ortho((n_hid, n_out),rng,"Why",stdev)
        
    bh  = np.zeros((n_hid,), dtype=theano.config.floatX)
    by  = np.zeros((n_out,), dtype=theano.config.floatX)

    bh  = theano.shared(bh, 'bh')
    by  = theano.shared(by, 'by')

    ###########################################################################
    # TRAINING                                                                #
    ###########################################################################

    h0 = T.alloc(np.array(0, dtype=theano.config.floatX), batch_size, n_hid)

    x = T.tensor3()
    t = T.imatrix()
    
    minibatch_index = T.lscalar("minibatch_index")

    lr = T.scalar()
    
    # Compute outputs fo the unrolled hidden layers
    h, _ = theano.scan(fn = lambda x_t, h_prev, Whh, Wxh, Why, bh: 
                         activ(T.dot(h_prev, Whh) + T.dot(x_t, Wxh) + bh), 
                       sequences = x,
                       outputs_info = [h0], # initialisation
                       non_sequences = [Whh, Wxh, Why, bh],
                       name = 'rec_layer')
    
    # Compute global output
    y = T.nnet.softmax(T.dot(h[-1], Why) + by)
    cost = -(t * T.log(y)).mean(axis=0).sum()

    # Compute gradients
    gWhh, gWxh, gWhy, gbh, gby = T.grad(cost, [Whh, Wxh, Why, bh, by])

    # Add the orthogonal penalty
    identity = np.identity(n_hid)
    identity = theano.shared(identity, "identity")

    if (penalty != 0):
        
        gWhh = gWhh + penalty * T.grad(T.sqrt(((T.dot(Whh, Whh.T) - 
                                                identity)**2).sum()), Whh)


    pen_err = T.sqrt(((T.dot(Whh, Whh.T) - identity)**2).sum())  

    # Compute weight norm
    norm_theta = T.sqrt((gWhh**2).sum() +
                        (gWxh**2).sum() +
                        (gWhy**2).sum() +
                        (gbh**2).sum() +
                        (gby**2).sum() )

    dWhh_norm = T.sqrt((gWhh**2).sum())

    # Gradient updates
    if gd_opt == "vanilla":

        updates_f = vanilla_sgd([Wxh, Whh, bh, Why, by],[gWxh, gWhh, gbh, gWhy,
                                gby], lr )            

    elif gd_opt == "adadelta":
        
        updates_f = adadelta([Wxh, Whh, bh, Why, by],[gWxh, gWhh, gbh, gWhy,
                             gby], lr)

    elif gd_opt == "nesterov":
        
        updates_f = nesterov_momentum([Wxh, Whh, bh, Why, by],[gWxh, gWhh, gbh,
                                      gWhy, gby], lr)

    elif gd_opt == "rmsprop":
        
        updates_f = rmsprop([Wxh, Whh, bh, Why, by],[gWxh, gWhh, gbh, gWhy, 
                            gby], lr )


    givens_train = {
        x : X_train[:,minibatch_index * batch_size: 
            (minibatch_index + 1) * batch_size,:],
        t : y_train[minibatch_index * batch_size: 
            (minibatch_index + 1) * batch_size,:]
    }

    train_step = function([minibatch_index, lr], 
                          [cost, norm_theta, dWhh_norm, pen_err],
                          updates = updates_f,
                          givens  = givens_train,
                          on_unused_input='warn')

    ###########################################################################
    # VALIDATION                                                              #
    ###########################################################################
    
    h0 = T.alloc(np.array(0, dtype=theano.config.floatX), 
                 X_test.get_value(borrow=True).shape[1], n_hid)    

    x = T.tensor3()
    t = T.imatrix()

    h, _ = theano.scan(fn = lambda x_t, h_prev, Whh, Wxh, Why: 
        activ(T.dot(h_prev, Whh) + T.dot(x_t, Wxh) + bh), 
                       sequences = x,
                       outputs_info = [h0],
                       non_sequences = [Whh, Wxh, Why],
                       name = 'validation')
    
    y = T.nnet.softmax(T.dot(h[-1], Why) + by)
    cost = -(t * T.log(y)).mean(axis=0).sum()
    error = T.neq(T.argmax(y, axis=1), T.argmax(t, axis=1)).mean()
        

    eval_step = function([], 
                         [cost, error],
                         givens = {x: X_test, 
                                   t: y_test})    

    print("******************************************************")
    print("Training starts...")
    print("******************************************************")

    training = True
     
    # Epoch counter
    n = 0
    
    avg_cost = 0
    avg_norm = 0
    best_acc = 0
    
    # Accuracy accumulator
    acc = []

    n_minibatches = math.ceil(X_train.get_value().shape[1] / batch_size)

    avg_cost = 0
    avg_norm = 0
    avg_dWhh_norm = 0
        
    # Check the error before any training commences
    valid_cost, error = eval_step()    
    print("Starting error: %05.2f" % error)
    
    while (training) and (n <= maxepoch):
    
        n += 1        

        for minibatch_index in range(n_minibatches):            

            # Train and update weights            
            tr_cost, W_norm, dWhh_norm, pen = train_step(minibatch_index, learning_rate)

            avg_cost += tr_cost
            avg_norm += W_norm
            avg_dWhh_norm += dWhh_norm

            # Time to evaluate?
            samples_seen = ((n - 1) * n_minibatches + (minibatch_index+1)) * batch_size                    

            if (samples_seen % check_interval == 0):
                
                if minibatch_index != 0:
                    avg_cost = avg_cost / check_interval
                    avg_norm = avg_norm / check_interval
                    avg_dWhh_norm = avg_dWhh_norm / check_interval
            
                valid_cost, error = eval_step()                
    
                acc.append((1.0 - error)*100)
                           
                if acc[-1] > best_acc:
                    best_acc = acc[-1]

                if np.isfinite(Whh.get_value()).any():
                  rho_Whh =np.max(abs(np.linalg.eigvals(Whh.get_value())))
                else:
                  rho_Whh = float('nan')
            
                print("Epoch %d" % n, ":", \
                      "Samples %d" % samples_seen, ":", \
                      "cost %05.3f, " % avg_cost, \
                      "|W| %7.2f, " % avg_norm, \
                      "|Whh| %7.2f, " % avg_dWhh_norm, \
                      "rho %01.2f," % rho_Whh, \
                      "val err %05.2f%%" % (error*100), ":", \
                      "best accuracy %05.2f%%" % best_acc)

                avg_cost = 0
                avg_norm = 0
                avg_dWhh_norm = 0

    print("------------------------------------------------------")

def main(args):

    parser = argparse.ArgumentParser(description="Train an SRNN against the \
                                     MNIST dataset (sequence of pixels).\
                                     This work is licensed under the Creative \
                                     Commons Attribution 4.0 International \
                                     License.")    
    
    parser.add_argument("--lr",help="Learning rate", default = 1e-6, 
                        required=False, type=float)

    parser.add_argument("--hidden",help="Number of units in the hidden layer", 
                        default = 100, required=False, type=int)

    parser.add_argument("--init", help="Weight initialization and activation \
                        function", choices=["normal", "uniform", "oinit"],
                        default = "oinit", required=False)

    parser.add_argument("--batchsize",help="Size of the minibatch", 
                        default = 20, required=False, type=int)

    parser.add_argument("--maxepochs",help="Maximum number of training epochs", 
                        default = 200, required=False, type=int)

    parser.add_argument("--penalty",help="Orthogonal penalty (lambda)", 
                        default = 0, required=False, type=float)

    parser.add_argument("--opt", help="Optimizer", choices=["nesterov", 
                                                            "vanilla", 
                                                            "adadelta", 
                                                            "rmsprop"],
                        default = "rmsprop", required=False)

    parser.add_argument("--stdev",help="Standard deviation for the random \
                        matrices", default = 0.001, required=False, type=float)

    parser.add_argument("--chk",help="Check interval", default = 16000, 
                        required=False, type=int)


    args = parser.parse_args()

    learning_rate = args.lr

    rng = np.random.RandomState(1234)    
        
    X_train, y_train, X_test, y_test = loadMNIST(0,0,rescaling=True)
        
    learning_rate = args.lr
    
    n_hid = args.hidden
    
    init = args.init
       
    batch_size = args.batchsize
    
    maxepoch = args.maxepochs
       
    gd_opt = args.opt
       
    penalty = args.penalty
    
    stdev = args.stdev
    
    check_interval = args.chk
    
    fit(X_train, y_train, X_test, y_test, learning_rate, n_hid, init, 
        batch_size, maxepoch, gd_opt, rng, penalty, stdev, check_interval)

if __name__ == "__main__":
    main(sys.argv)
