import os, sys

# TODO(dumitru): add license, copyright, authors, credit

import numpy, time, cPickle, gzip, os, sys

import theano
import theano.tensor as T

# Use the MRG random number generator when on the GPU
if 'gpu' in theano.config.device:
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
else:
    from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from rbm import RBM

from hf import truncated_newton
from deep_autoencoder import DeepAutoencoder

from ConfigParser import ConfigParser

def run_deep_autoencoder( finetune_lr = 0.1, pretraining_epochs = 50,
             global_pretraining_epochs = 200, global_pretrain_lr = 0.02,
             pretrain_lr = 0.1, top_layer_pretrain_lr = 0.001, k = 1,
             training_epochs = 1000, dataset='curves', batch_size = 20,
             global_pretraining_batch_size = 7500, weight_decay = 0.00002,
             global_pretraining_optimization = 'hf', preconditioner = None,
             supervised_training_type = 'russ', reconstruction_cost_type =\
             'cross_entropy', seed = 123):

    """ Demonstrates how to train and test a Deep Autoencoder, using either
    Hessian-free learning or stochastic gradient descent

    This is demonstrated on the MNIST or Curves dataset.

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining
    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training
    :type k: int
    :param k: number of Gibbs steps in CD/PCD
    :type global_pretraining_epochs: int
    :param global_pretraining_epochs: number of epoch to do global reconstruction pre-training
    :type global_pretrain_lr: float
    :param global_pretrain_lr: learning rate to be used during global reconstruction pre-training for SGD
    :type global_pretraining_batch_size: float
    :param global_pretraining_batch_size: batch size for global pre-training
    :type global_pretraining_optimization: float
    :param global_pretraining_optimization: type of optimization to apply
    during global pre-training (hf or sgd)
    :type training_epochs: int
    :param training_epochs: maximal number of iterations ot run the optimizer
    :type finetune_lr: float
    :param finetune_lr: learning rate used in the fine-tuning stage
    :type dataset: string
    :param dataset: name of dataset to use
    :type batch_size: int
    :param batch_size: the size of a minibatch for SGD
    """

    # TODO(dumitru): put this into a config file for each dataset
    if dataset == 'mnist':
        dataset = 'mnist.pkl.gz'
        pretrained_model_file = 'hinton_pretrained_mnist'
        n_ins = 28*28
        n_outs = 10
        hidden_layers_sizes = [1000, 500, 250, 30]
    elif dataset == 'curves':
        dataset = 'curves.gz'
        pretrained_model_file = 'hinton_pretrained_model'
        n_ins = 28*28
        n_outs = 6
        hidden_layers_sizes = [400, 200, 100, 50, 25, 6]
    else:
        raise NotImplementedError("Unknown dataset")

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x , test_set_y  = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # numpy random generator. seed = None means non-deterministic random
    # numbers
    numpy_rng = numpy.random.RandomState(seed)

    print '... building the model'
    # construct the Deep Belief Network
    dbn = DeepAutoencoder(numpy_rng = numpy_rng, n_ins = n_ins,
                          hidden_layers_sizes = hidden_layers_sizes,
                          n_outs = n_outs, reconstruction_cost =\
                          reconstruction_cost_type, supervised_training =\
                          supervised_training_type, tied_weights = False)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    if pretraining_epochs > 0:
      if os.path.isfile(pretrained_model_file):
        print '... loading the pretrained model'
        dbn.load_rbm_weights(pretrained_model_file)

      else:
        print '... getting the pretraining functions'
        pretraining_fns = dbn.build_pretraining_functions(
            train_set_x   = train_set_x,
            batch_size    = batch_size,
            k             = k)

        print '... pre-training the model'
        start_time = time.clock()
        ## Pre-train layer-wise
        for i in xrange(dbn.n_layers):
           # go through pretraining epochs
           for epoch in xrange(pretraining_epochs):
              # go through the training set
              c = []
              for batch_index in xrange(n_train_batches):
                 if i == dbn.n_layers - 1:
                   plr = top_layer_pretrain_lr
                 else:
                   plr = pretrain_lr
                 c.append(pretraining_fns[i](index = batch_index,
                                             lr = plr, weight_decay = weight_decay ) )
              print 'Pre-training layer %i, epoch %d, cost %f'%(i,epoch,numpy.mean(c))

        end_time = time.clock()
        print >> sys.stderr, ('The pretraining code for file '+os.path.split(__file__)[1]+' ran for %.2fm' % ((end_time-start_time)/60.))

        print '... saving the model'
        dbn.save_rbm_weights(pretrained_model_file)

    #########################
    # GLOBAL RECONSTRUCTION #
    #########################
    print '... getting the global reconstruction functions'
    if global_pretraining_optimization == 'hf':
        if global_pretraining_epochs > 0:
           ridge = 1
        else:
           # Worse curvature when no-pretraining has been performed, a higher
           # ridge is likely to be required
           ridge = 256
        train_fn, validate_model, test_model =\
        dbn.build_global_pretraining_functions_hf(datasets = datasets,
                                                     train_batch_size =\
                                                     global_pretraining_batch_size,
                                                     preconditioner =\
                                                     preconditioner,
                                                     ridge = ridge,
                                                     maxiterations =
                                                     global_pretraining_epochs\
                                                     * n_train_batches)
    elif global_pretraining_optimization == 'sgd':
        train_fn, validate_model, test_model =\
        dbn.build_global_pretraining_functions_sgd(datasets = datasets,\
                                                      train_batch_size = batch_size,
                                                      learning_rate = global_pretrain_lr)
    else:
        raise NotImplementedError("Invalid global_pretraining_optimization")

    # TODO(dumitru): verify that this is a good idea when weights are randomly
    # initialized (vs. pre-trained)
    print '... initializing the reconstruction weights to the encoder weights'
    dbn.initialize_reconstruction_weights()

    print '... global pre-training of the model'
    start_time = time.clock()

    if global_pretraining_optimization == 'hf':
        costs_log = train_fn()
        print 'Global pre-training, reconstruction cost %f, mse %f '%(costs_log[-1][0],
                                                                  costs_log[-1][1])
    elif global_pretraining_optimization == 'sgd':
       for epoch in xrange(global_pretraining_epochs):
          # go through the training set
          c = []
          for batch_index in xrange(n_train_batches):
             c.append(train_fn(batch_index))
          cost_values = numpy.mean(c,axis=0)
          print 'Iter',epoch, 'Objective function value =', cost_values[0], 'other costs = ', cost_values[1:]
    else:
        raise NotImplementedError("Invalid global_pretraining_optimization")

    end_time = time.clock()
    print >> sys.stderr, ('The global pretraining code for file '+os.path.split(__file__)[1]+' ran for %.2fm' % ((end_time-start_time)/60.))

    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = dbn.build_finetune_functions (
                datasets = datasets, batch_size = batch_size,
                learning_rate = finetune_lr)

    print '... finetunning the model'
    # early-stopping parameters
    patience              = 4*n_train_batches # look as this many examples regardless
    patience_increase     = 2.    # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995 # a relative improvement of this much is
                                  # considered significant
    validation_frequency  = min(n_train_batches, patience/2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch


    best_params          = None
    best_validation_loss = float('inf')
    test_score           = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
      epoch = epoch + 1
      for minibatch_index in xrange(n_train_batches):

        minibatch_avg_cost = train_fn(minibatch_index)
        iter    = epoch * n_train_batches + minibatch_index

        if (iter+1) % validation_frequency == 0:

            validation_losses = validate_model()
            this_validation_loss = numpy.mean(validation_losses)
            print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                   (epoch, minibatch_index+1, n_train_batches, \
                    this_validation_loss*100.))

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:

                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                       improvement_threshold :
                    patience = max(patience, iter * patience_increase)

                # save best validation score and iteration number
                best_validation_loss = this_validation_loss
                best_iter = iter

                # test it on the test set
                test_losses = test_model()
                test_score = numpy.mean(test_losses)
                print(('     epoch %i, minibatch %i/%i, test error of best '
                      'model %f %%') %
                             (epoch, minibatch_index+1, n_train_batches,
                              test_score*100.))


        if patience <= iter :
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score*100.))
    print >> sys.stderr, ('The fine tuning code for file '+os.path.split(__file__)[1]+' ran for %.2fm' % ((end_time-start_time)/60.))

if __name__ == '__main__':

    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-d", "--dataset", dest="dataset",
                  help="dataset to use (mnist or curves)", default = "mnist")
    parser.add_option("-p", "--preconditioner",
                  help="lcg preconditioner", dest="preconditioner", default=None)
    parser.add_option("-b", "--batch_size",
                  help="global pretraining batch size", dest="gp_batch_size", default=3000)
    parser.add_option("-o", "--optimizer",
                  help="global reconstruction optimization method (sgd, hf)",
                      dest="global_pretraining_optimization", default="hf")
    parser.add_option("-s", "--seed",
                  help="seed for the random number generator (None =\
                      non-deterministic)", dest="seed", default=123)
    parser.add_option("-e", "--pretraining_epochs",
                  help="number of pretraining epochs",
                      dest="pretraining_epochs", default=50)


    (options, args) = parser.parse_args()

    if options.seed:
        options.seed = int(options.seed)

    run_deep_autoencoder(dataset=options.dataset,
                         preconditioner = options.preconditioner,
                         training_epochs = 0,
                         global_pretraining_epochs = 5000,
                         global_pretraining_batch_size = int(options.gp_batch_size),
                         global_pretraining_optimization = options.global_pretraining_optimization,
                         pretraining_epochs = int(options.pretraining_epochs),
                         seed = options.seed)
