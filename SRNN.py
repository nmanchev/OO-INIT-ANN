"""
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
Simple Recurrent Neural Network with Learned Orthogonality

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

This network also uses the original data generation classes (TempOrderTask,
AddTask, PermTask, TempOrder3bitTask) used in Pascanu et al.

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

from tempOrder     import TempOrderTask
from addition      import AddTask
from permutation   import PermTask
from tempOrder3bit import TempOrder3bitTask
from collections   import OrderedDict
from theano        import function

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

def xavier_uniform(shape, rng, variable_name):
    """
    Generates a random matrix of shape *shape using normalised initialisation
    as suggested in Glorot, X., & Bengio, Y. "Understanding the difficulty 
    of training deep feedforward neural networks." (2010, March)
    
    W ~ U [- sqrt(6/sum(shape)), + sqrt(6/sum(shape))]
    
    Parameters
    ----------
    shape         : shape of W (tuple)
    rng           : numpy.random.RandomState instance
    variable_name : name for the shared variable based on W
        
    Returns
    -------
    A matrix of size shape populated using normalised initialisation
    """    
    urange = np.sqrt(6./(sum(shape)))
    A = -urange + 2 * urange * rng.rand(*shape)
    return theano.shared(A, variable_name)



def rand_ortho(shape, rng, variable_name, alpha_p = 0.1, Ep = 0.000001, 
               max_it = 100) : 
    """
    Uses pre-training to orthogonalise a matrix. The matrix is initially 
    populated using normalised initialisation.
        
    Parameters
    ----------
    shape         : shape of the matrix
    rng           : numpy.random.RandomState instance
    variable_name : name for the shared variable
    alpha_p       : pre-training learning rate
    Ep            : convergence criterion for the pre-training (Err < Ep)
    max_it        : maximum number of iterations in the pre-training
        
    Returns
    -------
    An orthognal/semi-orthogonal matrix learned using pre-training
    """        
    rows,cols = shape

    W = xavier_uniform((rows,cols), rng, variable_name)
    
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


def fit(rng, learning_rate, n_hid, init, batch_size, max_length, min_length, 
        task, maxiter, chk_interval, val_size, val_batch, gd_opt, task_name, 
        ortho_reg):
    """
    Trains an SRNN on a selected synthetic problem.
        
    Parameters
    ----------
    rng           : numpy.random.RandomState instance
    learning_rate : learning rate (alpha)
    n_hid         : number of neurons in the hidden layer
    init          : weight initialisation (xavier/oinit)
    batch_size    : number of samples in a mini-batch (training set)
    max_length    : max sequence length
    min_length    : min_sequence length
    task          : synthetic problem class instance (e.g. TempOrderTask)
    maxiter       : maximum training iterations
    chk_interval  : number of iterations between netwrok evaluation
    val_size      : number of samples in the test size 
    val_batch     : number of samples in a mini-batch (test set)
    gd_opt        : weight update method (vanilla, rmsprop, adadelta, nesterov)
    task_name     : name for the synthetic problem (for logging purposes)
    ortho_reg     : orthogonal penalty strength (lambda)
    
    Returns
    -------
    Number of training iterations and best accuracy achieved by the network
    """           
    print("------------------------------------------------------")
    print("******************************************************")
    print("Parameters - SRNN")
    print("******************************************************")
    print("task              : %s" % task_name)
    print("optimization      : %s" % gd_opt)
    print("learning_rate     : %f" % learning_rate)
    print("maxiter           : %i" % maxiter)
    print("batch_size        : %i" % batch_size)
    print("min_length        : %i" % min_length)
    print("max_length        : %i" % max_length)
    print("chk_interval      : %i" % chk_interval)
    print("n_hid             : %i" % n_hid)
    print("init              : %s" % init)
    print("val_size          : %i" % val_size)
    print("val_batch         : %i" % val_batch)
    print("orthogonal penalty: %f" % ortho_reg)        
    print("******************************************************")
        
    n_inp = task.nin
    n_out = task.nout
    
    # Initialise the synaptic weights
    if init == "xavier":
        
        Wxh = xavier_uniform((n_inp, n_hid),rng, "Wxh")
        Whh = xavier_uniform((n_hid, n_hid),rng, "Whh")
        Why = xavier_uniform((n_hid, n_out),rng, "Why")

    elif init == "oinit":

        Wxh = rand_ortho((n_inp, n_hid),rng, "Wxh")
        Whh = rand_ortho((n_hid, n_hid),rng, "Whh")
        Why = rand_ortho((n_hid, n_out),rng, "Why")    

        print("******************************************************")
    
    bh  = np.zeros((n_hid,), dtype=theano.config.floatX)
    bh  = theano.shared(bh, 'bh')

    by  = np.zeros((n_out,), dtype=theano.config.floatX)       
    by  = theano.shared(by, 'by')

    activ = T.tanh
    
    ###########################################################################
    # TRAINING                                                                #
    ###########################################################################

    h0 = T.alloc(np.array(0, dtype=theano.config.floatX), batch_size, n_hid)

    x = T.tensor3()
    t = T.matrix()

    lr = T.scalar()

    # Compute outputs fo the unrolled hidden layers
    h, _ = theano.scan(fn = lambda x_t, h_prev, Whh, Wxh, Why: 
                         activ(T.dot(h_prev, Whh) + T.dot(x_t, Wxh) + bh), 
                       sequences = x,
                       outputs_info = [h0],
                       non_sequences = [Whh, Wxh, Why],
                       name = 'rec_layer')

    # Compute global output
    if task.classifType == 'lastSoftmax':
        y = T.nnet.softmax(T.dot(h[-1], Why) + by)
        cost = -(t * T.log(y)).mean(axis=0).sum()
    elif task.classifType == 'lastLinear':
        y = T.dot(h[-1], Why) + by
        cost = ((t - y)**2).mean(axis=0).sum()
   
    # Compute gradients
    gWhh, gWxh, gWhy, gbh, gby = T.grad(cost, [Whh, Wxh, Why, bh, by])

    # Add the orthogonal penalty
    if (ortho_reg != 0):
        
        identity = np.identity(n_hid)
        identity = theano.shared(identity, "identity")
        
        gWhh = gWhh + ortho_reg * T.grad(T.sqrt(((T.dot(Whh, Whh.T) - 
                                                  identity)**2).sum()), Whh)

    # Compute weight norm
    norm_theta = T.sqrt((gWhh**2).sum() +
                        (gWxh**2).sum() +
                        (gWhy**2).sum() +
                        (gbh**2).sum() +
                        (gby**2).sum() )

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

    train_step = function([x,t,lr],[cost, norm_theta],
                            on_unused_input='warn',
                            updates=updates_f)
    
    ###########################################################################
    # VALIDATION                                                              #
    ###########################################################################
    
    h0 = T.alloc(np.array(0, dtype=theano.config.floatX), val_batch, n_hid)

    x = T.tensor3()   
    t = T.matrix()

    h, _ = theano.scan(fn = lambda x_t, h_prev, Whh, Wxh, Why: 
                              activ(T.dot(h_prev, Whh) + T.dot(x_t, Wxh) + bh), 
                       sequences = x,
                       outputs_info = [h0],
                       non_sequences = [Whh, Wxh, Why],
                       name = 'validation')
    
    if task.classifType == 'lastSoftmax':
        y = T.nnet.softmax(T.dot(h[-1], Why) + by)
        cost = -(t * T.log(y)).mean(axis=0).sum()
        error = T.neq(T.argmax(y, axis=1), T.argmax(t, axis=1)).mean()
        
    elif task.classifType == 'lastLinear':
        y = T.dot(h[-1], Why) + by
        cost = ((t - y)**2).mean(axis=0).sum()
        error = (((t - y)**2).sum(axis=1) > .04).mean()

    eval_step = function([x,t], [cost, error])       

    ###########################################################################
    # TRAINING LOOP                                                           #
    ###########################################################################

    print("Training")
    
    lr = learning_rate
    
    training = True
    
    n = 1
    avg_cost = 0
    avg_norm = 0
    best_score = 100
    
    while (training) and (n <= maxiter):
        
        # Variable sequence length?
        if max_length > min_length:
            length = min_length + rng.randint(max_length - min_length)
        else:
            length = min_length
                
        # Train and update weights
        train_x, train_y = task.generate(batch_size, length)
                
        tr_cost, tr_norm = train_step(train_x, train_y, lr)
        
        avg_cost += tr_cost
        avg_norm += tr_norm
        
        # Time to evaluate?
        if (n % chk_interval == 0):
            avg_cost = avg_cost / float(chk_interval)
            avg_norm = avg_norm / float(chk_interval)                    

            valid_cost = 0
            error = 0
            
            for dx in range(val_size // val_batch):
                if max_length > min_length:
                    length = min_length + rng.randint(max_length - min_length)
                else:
                    length = min_length
                    
                valid_x, valid_y = task.generate(val_batch, length)
                _cost, _error = eval_step(valid_x, valid_y)                

                error = error + _error
                valid_cost = valid_cost + _cost
            
            error = error*100. / float(val_size // val_batch)
            valid_cost = valid_cost / float(val_size // val_batch)
            
            if np.isfinite(Whh.get_value()).any():
                rho =np.max(abs(np.linalg.eigvals(Whh.get_value())))
            else:
                rho = float('nan')
            
            if error < best_score:
                best_score = error
        
            print("Iter %07d"%n,":", \
                  "cost %05.3f, " % avg_cost, \
                  "average gradient norm %7.3f, " % avg_norm, \
                  "rho %01.3f," % rho, \
                  "valid error %07.3f%%, " % error, \
                  "best valid error %07.3f%%" % best_score)

            if error < .0001 and np.isfinite(valid_cost):
                training = False
                print("PROBLEM SOLVED!")


            avg_cost = 0
            avg_norm = 0
        
        n += 1

    print("------------------------------------------------------")

    return n, best_score

def main(args): 

    rng = np.random.RandomState(1234)

    parser = argparse.ArgumentParser(description="Train an SRNN against the \
                                     pathological tasks defined in Hochreiter,\
                                     S. and Schmidhuber, J. (1997). Long \
                                     short-term memory. Neural Computation, \
                                     9(8), 1735–1780. This work is licensed \
                                     under the Creative Commons Attribution \
                                     4.0 International License.")

    parser.add_argument("--task", help="Pathological task", 
                        choices=["temporal", "temporal3","addition", "perm"],
                        required=True)

    parser.add_argument("--maxiter",help="Maximum number of iterations", 
                        default = 1000000, required=False, type=int)
    
    parser.add_argument("--batchsize",help="Size of the minibatch", 
                        default = 20, required=False, type=int)

    parser.add_argument("--min",help="Minimal length of the task", 
                        default = 50, required=False, type=int)

    parser.add_argument("--max",help="Maximal length of the task", 
                        default = 50, required=False, type=int)

    parser.add_argument("--chk",help="Check interval", 
                        default = 100, required=False, type=int)

    parser.add_argument("--hidden",help="Number of units in the hidden layer", 
                        default = 100, required=False, type=int)

    parser.add_argument("--init", help="Weight initialization and activation \
                        function", choices=["xavier", "oinit"],
                        default = "oinit", required=False)

    parser.add_argument("--lr",help="Learning rate", default = 0.1, 
                        required=False, type=float)
    
    parser.add_argument("--opt", help="Optimizer", 
                        choices=["nesterov", "vanilla", "adadelta", "rmsprop"],
                        default = "rmsprop", required=False)

    parser.add_argument("--reg", help="Orthogonal regularisation", 
                        default = 0.1, required=False, type=float)

    args = parser.parse_args()

    learning_rate = args.lr

    min_length = args.min
    max_length = args.max

    if (min_length > max_length):
        print("Invalid sequence length. Setting min to %i." % max_length)
        min_length = max_length
           
    if args.task == "temporal":
        task = TempOrderTask(rng, theano.config.floatX)        
    elif args.task == "addition":
        task = AddTask(rng, theano.config.floatX)
    elif args.task == "perm":
        task = PermTask(rng, theano.config.floatX)
    if args.task == "temporal3":
        task = TempOrder3bitTask(rng, theano.config.floatX)        
    
    maxiter = args.maxiter

    batch_size = args.batchsize

    chk_interval = args.chk

    n_hid = args.hidden

    init   = args.init
    gd_opt = args.opt
    
    val_size  = 10000
    val_batch = 1000
    
    fit(rng, learning_rate, n_hid, init, batch_size, max_length, min_length, 
        task, maxiter, chk_interval, val_size, val_batch, gd_opt, args.task,
        args.reg)    

if __name__=='__main__':
    main(sys.argv)
    
