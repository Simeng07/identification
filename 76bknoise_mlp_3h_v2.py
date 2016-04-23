# -*- coding: utf-8 -*-
'''
Build a tweet sentiment analyzer
'''
from collections import OrderedDict
import cPickle as pkl
import sys
import os
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import cnn_utils

from bknoise_load_data import load_bknoise_data
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv



# Set the random number generators' seeds for consistency
SEED = 10000000
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    返回list of list.  每个Minibatch是Index的list，大小是minibatchSize。最后一个minibatch大小不定。
    """
    idx_list = numpy.arange(n, dtype="int32")
   
    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff. 向GPU传入参数
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff. 从GPU读入参数到Numpy。
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

# use_noise是一个Theano Shared.
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):    # 主要是给每一层一个名字，然后合成参数名，防止众多参数重名。
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    初始化所有参数（比较复杂的层就调单独函数，简单的就在这里初始化）.
    这样写需要对各层都比较了解，看起来封装不好，但是灵活性好。
    """
    params = OrderedDict()
    # embedding
    #randn = numpy.random.rand(options['n_words'],
    #                          options['dim_proj'])
    #params['Wemb'] = (0.01 * randn).astype(config.floatX)
    #params = get_layer(options['encoder'])[0](options,
    #                                          params,
    #                                          prefix=options['encoder'])
    #param_init_cnn(options, params, prefix='c1')
    #param_init_cnn(options, params, prefix='c2')
    #param_init_cnn(options, params, prefix='c3')
    #param_init_cnn(options, params, prefix='c4')
    # hidden layer
    params['h1_W'], params['h1_b'] = cnn_utils.ini_hidden_params(options['h1_option'])
    params['h2_W'], params['h2_b'] = cnn_utils.ini_hidden_params(options['h2_option']) 
    params['h3_W'], params['h3_b'] = cnn_utils.ini_hidden_params(options['h3_option'])                                        
    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['h3_option'][1],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]
    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_cnn(options, params, prefix='cnn'):
    filter_shape = options[_p(prefix,'filter')]
    poolsize = options[_p(prefix,'pool')]
    fan_in = numpy.prod(filter_shape[1:])
    fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
    # initialize weights with random weights
    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    W = numpy.asarray(numpy.random.uniform(low=-W_bound, high=W_bound, 
            size=filter_shape), dtype=theano.config.floatX)
        # the bias is a 1D tensor -- one bias per output feature map
    b = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = b
    return params


def cnn_layer(tparams, input, options, prefix='cnn'):
    filter_shape = options[_p(prefix,'filter')]
    poolsize = options[_p(prefix,'pool')]
    image_shape = options[_p(prefix,'image')]
    assert image_shape[1] == filter_shape[1]
    # convolve input feature maps with filters
    conv_out = conv.conv2d(input=input, filters=tparams[_p(prefix, 'W')],
            filter_shape=filter_shape, image_shape=image_shape)

    # downsample each feature map individually, using maxpooling
    pooled_out = downsample.max_pool_2d(input=conv_out, ds=poolsize, ignore_border=False)

    # add the bias term. Since the bias is a vector (1D array), we first
    # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
    # thus be broadcasted across mini-batches and feature map
    # width & height
    output = tensor.tanh(pooled_out + tparams[_p(prefix, 'b')].dimshuffle('x', 0, 'x', 'x'))
    return output


def sgd(lr, tparams, grads, x, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared',allow_input_downcast=True)

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared',allow_input_downcast=True)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update',allow_input_downcast=True)

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared',allow_input_downcast=True)

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update',allow_input_downcast=True)

    return f_grad_shared, f_update


def build_model(tparams, options):
    trng = RandomStreams(SEED)
    # define all symbolic variables.
    use_noise = theano.shared(numpy_floatX(0.))
    x = tensor.matrix('x', dtype= config.floatX)
    y = tensor.ivector('y')
    
    nsample = x.shape[0]
    #rx = x.reshape((nsample, 1, 2049, 1))
    #cnnout = cnn_layer(tparams,rx,options,'c1')
    #cnnout = cnn_layer(tparams,cnnout,options,'c2')
    #cnnout2 = cnn_layer(tparams,cnnout2,options,'c3')
    #cnnout2 = cnn_layer(tparams,cnnout2,options,'c4')
    #cnnout = cnnout.flatten(2)
    proj = tensor.dot(x, tparams['h1_W']) + tparams['h1_b']   
    proj = tensor.tanh(proj) 
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)
    proj = tensor.dot(proj, tparams['h2_W']) + tparams['h2_b']   
    proj = tensor.tanh(proj)  
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)
    proj = tensor.dot(proj, tparams['h3_W']) + tparams['h3_b']   
    proj = tensor.tanh(proj)  
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)    
    #proj = tensor.dot(proj, tparams['h3_W']) + tparams['h3_b']   
    #proj = tensor.tanh(proj)  
    #if options['use_dropout']:
    #    proj = dropout_layer(proj, use_noise, trng)
    # hopefully, this produce (max_len,n_samples,y_dim)
    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])
    f_pred_prob = theano.function([x], pred, name='f_pred_prob')
    f_pred = theano.function([x], pred.argmax(axis=1), name='f_pred')
    cost = -tensor.mean(tensor.log(pred)[tensor.arange(y.shape[0]), y])
    return use_noise, x, y, f_pred_prob, f_pred, cost


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  [data[1][t] for t in valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples classified' % (n_done, n_samples)

    return probs


def pred_error(f_pred, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    total_pred = 0
    predy = None
    for _, valid_index in iterator:
        x =numpy.asarray( [data[0][t] for t in valid_index],dtype=config.floatX)
        y = numpy.asarray([data[1][t] for t in valid_index],dtype = 'int64')
        preds = f_pred(x)
        targets = y
        valid_err += ((preds == targets)).sum()
        total_pred += x.shape[0]
        if predy is None:
            predy = preds
        else:
            predy = numpy.concatenate((predy,preds))
    valid_err = 1. - numpy_floatX(valid_err) / total_pred
    return valid_err,predy


def get_pred_prob(f_pred, f_pred_prob, data, iterator, verbose=False):
    #valid_err = 0
    #total_pred = 0
    predy = None
    for _, valid_index in iterator:
        x =numpy.asarray( [data[0][t] for t in valid_index],dtype=config.floatX)
        y = numpy.asarray([data[1][t] for t in valid_index],dtype = 'int64')
        preds = f_pred_prob(x)
        #targets = y
        #preds = y
        #valid_err += ((preds == targets)).sum()
        #total_pred += x.shape[0]
        if predy is None:
            predy = preds
        else:
            predy = numpy.concatenate((predy,preds))
    #valid_err = 1. - numpy_floatX(valid_err) / total_pred
    numpy.save('predy_3h_3_6', predy)

def sample_from_lstm(f_pred, prepare_data,vocab):
    ini_sample='First'
    sample_len = 100
    tline =vocab(ini_sample)
    datax = [tline]
    datay = [tline] # no use in sample. here for code conssitence
    for i in xrange(sample_len):
        x, mask,how_long, y = prepare_data(datax,datay,maxlen=None)
        preds = f_pred(x, mask)
        nexty = preds[-1,0]
        tx = numpy.concatenate((datax[0], numpy.array([nexty])))
        #print tx.shape,tx
        datax = [tx]
        datay = [tx]
    print 'sampled text:',datax[0].shape
    print vocab(datax[0])


def sample_from_lstm_scan(f_pred, prepare_data,vocab):
    ini_sample='First'
    sample_len = 100
    tline =vocab(ini_sample)
    datax = [tline]
    datay = [tline] # no use in sample. here for code conssitence
    for i in xrange(sample_len):
        x, mask, y = prepare_data(datax,datay,maxlen=None)
        preds = f_pred(x, mask)
        nexty = preds[-1,0]
        tx = numpy.concatenate((datax[0], numpy.array([nexty])))
        #print tx.shape,tx
        datax = [tx]
        datay = [tx]
    print 'sampled text:',datax[0].shape
    print vocab(datax[0])


def train_cnn(
    dim_proj=400,  # word embeding dimension and LSTM number of hidden units.
    patience=100,  # Number of epoch to wait before early stop if no progress
    max_epochs=300,  # The maximum number of epoch to run
    dispFreq=-1,  # Display to stdout the training progress every N updates
    decay_c=1e-4,  # Weight decay for the classifier applied to the U weights.
    lrate=0.01,  # Learning rate for sgd (not used for adadelta and rmsprop)
    optimizer = rmsprop,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    saveto='bknoise_model.npz',  # The best model will be saved there
    validFreq=-1,  # Compute the validation error after this number of update.
    saveFreq=-1,  # Save the parameters after every saveFreq updates
    batch_size=9000,  # The batch size during training.
    image_width = 32, image_height=32,
    h1_option = (2049,1000, 1),
    h2_option = (1000,1000, 1),
    h3_option = (1000,500, 1),
    c1_filter = (32, 1, 3, 1), c1_pool = (1,1), c1_image = (None,1,2049,1),
    c2_filter = (16, 32, 3, 1), c2_pool = (3,1), c2_image = (None,32,2047,1),
    #c3_filter = (32, 32, 3, 3), c3_pool = (2,2), c3_image = (None,32,26,26),
    #c4_filter = (32, 32, 3, 3), c4_pool = (1,1), c4_image = (None,32,12,12),
    # Parameter for extra option
    noise_std=0.5,
    use_dropout=False,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
    dataset = 'bkclass_9_4096.pkl.gz',
):

    #config.optimizer='fast_compile'
    #config.exception_verbosity='high'
    #config.on_unused_input='ignore'
    if os.name=='nt': # run on my laptop
        max_epochs=2
        dispFreq=1
        validFreq=2
        
    # Model options
    model_options = locals().copy()
    #print "model options", model_options

    print 'Loading data'
    train, valid, test = load_bknoise_data(dataset)
    ydim = numpy.max(map(numpy.max,train[1]))+1
    print 'ydim:',ydim
    model_options['ydim'] = ydim

    print 'Building model'
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params(saveto, params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    if decay_c > 0.:  # 这里只是对最后一层进行decay
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay += (tparams['h1_W'] ** 2).sum()
        weight_decay += (tparams['h2_W'] ** 2).sum()
        weight_decay += (tparams['h3_W'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=tparams.values())
    f_grad = theano.function([x, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads, x, y, cost)

    print 'Optimization'
    kf_valid = get_minibatches_idx(len(valid[0]), batch_size)
    kf_test = get_minibatches_idx(len(test[0]), batch_size)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size,shuffle=True) 

    print "%d train examples" % len(train[0])
    print "%d valid examples" % len(valid[0])
    print "%d test examples" % len(test[0])

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = 5*len(train[0]) / batch_size #validation every 5 epoch
    if saveFreq == -1:
        saveFreq = 5*len(train[0]) / batch_size
    if dispFreq == -1:
        dispFreq = len(train[0]) / batch_size # display every epoch.

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.clock()
    try:
        for eidx in xrange(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size)

            for _, train_index in kf:
                
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = numpy.asarray([train[1][t] for t in train_index],dtype='int64')
                x = numpy.asarray([train[0][t] for t in train_index],dtype=config.floatX)

                n_samples += x.shape[1]
                #print 'x.shape' ,x.shape,y.shape
                cost = f_grad_shared(x, y)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost
                    
                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print 'Saving...',

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print 'Done'

                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.) # 预测的时候不需要dropout, 所有输出一起Average.
                    train_err,_ = pred_error(f_pred,  train, kf)
                    test_err,_ = pred_error(f_pred, test, kf_test)
                    valid_err = test_err
                    history_errs.append([test_err, train_err])
                    if (uidx == 0 or
                        valid_err <= numpy.array(history_errs)[:,
                                                               0].min()):
                        best_p = unzip(tparams)
                        bad_counter = 0

                    print ('Train ', train_err,  'Test ', test_err)

                    if (len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

            #print 'Seen %d samples' % n_samples

            if estop:
                break

    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.clock()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    
    train_err,_ = pred_error(f_pred, train, kf_train_sorted)
    #valid_err = pred_error(f_pred,  valid, kf_valid)
    test_err,predy  = pred_error(f_pred,  test,  kf_test)

    get_pred_prob(f_pred, f_pred_prob, test, kf_test)

    print 'Train ', train_err,  'Test ', test_err
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' %
                          (end_time - start_time))
    
                          
    #print 'train error histrory:', history_errs
    #predy = f_pred(test[0])  # dangoursou to sumup memory
    print type(predy),predy.shape
    print classification_report(test[1], predy)
    print confusion_matrix(test[1], predy)
    return train_err, valid_err, test_err


if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    train_cnn( )
