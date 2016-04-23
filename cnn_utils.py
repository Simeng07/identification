# -*- coding: utf-8 -*-

""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""

import numpy
import PIL.Image as Image
from matplotlib import pyplot
import theano
import pywt

def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(theano.config.floatX)

def ini_hidden_params(ini_info):
    n_in,n_out,itype = ini_info
    W_values = numpy.asarray(numpy.random.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
    if itype == 2:
        W_values *= 4
    b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
    return  W_values , b_values


'''
初始化为fft变换的filter。
conv over this kind of filter equels fft with diffrent phase
'''
def ini_fft_like_filter(flt):
    import math
    assert(flt.shape[1]==1 and flt.shape[3]==1) # can only used in time series
    assert(flt.shape[2]>=5) # at least 5 point to represent a sine curve
    nf = flt.shape[0]
    flen = flt.shape[2]
    for i in xrange(nf):
        dt = 2*math.pi/(i+4)
        for j in xrange(flen):
            flt[i,0,j,0]=math.sin(dt*j)
    return 0

'''
初始化为小波的wavelet。
'''
def ini_wavlet_like_filter(flt):
    fns, flts = get_all_filters()
    assert(flt.shape[1]==1 and flt.shape[3]==1) # can only used in time series
    assert(flt.shape[0]<=len(flts)) # the max number of filters that I got.
    nf = flt.shape[0]
    flen = flt.shape[2]
    for i in xrange(nf):
        tf = flts[i]
        tfl = tf.shape[0]
        offset = (flen-tfl)//2
        flt[i,0,offset:tfl+offset,0]=tf
    return 0

'''
得到pywt中实现的所有小波函数，非正交的?。
if level =3 ,the max len of the filter is ???
'''
def get_all_filters(wavlevel=5,lenlimit=100):
    wavs = []
    filters = []
    filternames = []
    #fas = pywt.families()
    fas = ['db','sym', 'coif','dmey']
    for fa in fas:
        falist = pywt.wavelist(fa)
        wavs = wavs + falist
    for wav in wavs:
        wavelet = pywt.Wavelet(wav)
        for i in xrange(wavlevel):
            [phi, psi, x] = wavelet.wavefun(level=i+1)
            llen = len(psi)
            if llen<lenlimit:
                print wav,' len:', len(psi),min(x),max(x)
                filters.append(psi)
                filternames.append(wav+'-'+str(i+1))
    return filternames, filters

def plot_train_history(train_loss,valid_loss):
    pyplot.plot(train_loss, linewidth=2, label="train")
    pyplot.plot(valid_loss, linewidth=2, label="valid")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    #pyplot.ylim(1e-3, 1e-2)
    pyplot.yscale("log")
    pyplot.show()


def visual_numpy(x,img_shape, tile_shape, spacing=1, nchannel=3 ):
    """
    visual numpy array to make an image of (imgwid+spacing)*ncol, (imghei+spacing)*nrow
    :type X: a 2-D ndarray ;
    :param X: every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param spacing: spacing between each image

    :param nchannel: 3:color, 1:grayscale
    """
    outfile = 'output_visual.png'
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert x.shape[1] == img_shape[0]*img_shape[1]*nchannel
    
    # if we are dealing with only one channel
    H, W = img_shape
    rn, cn = tile_shape
    Hs = spacing
    Ws = spacing

    # generate a matrix to store the output
    dt = x.dtype
    out_shape = ((H+Hs)*rn,(W+Ws)*cn,nchannel)
    image_data = numpy.zeros(out_shape, dtype=dt)
    for tile_row in xrange(rn):
        for tile_col in xrange(cn):
            indx = tile_row*cn+tile_col
            this_x = x[indx]
            #this_img = this_x.reshape((H,W,nchannel))
            this_img = this_x.reshape((nchannel,H,W))
            this_img = numpy.swapaxes(this_img,0,2) 
            this_img = numpy.swapaxes(this_img,0,1) 
            image_data[tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W,:
                    ] = this_img 
    image = Image.fromarray(image_data)
    image.save(outfile)
    

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar



def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2


    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array


if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    #na, fs = get_all_filters( )
    #print 'get ', len(fs),'filters'
    te = numpy.zeros((5,1,20,1),dtype=float)
    i = ini_fft_like_filter(te)
    print te