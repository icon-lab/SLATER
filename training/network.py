import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.ops.upfirdn_2d import upsample_2d, downsample_2d, upsample_conv_2d, conv_downsample_2d
from dnnlib.tflib.ops.fused_bias_act import fused_bias_act
from dnnlib import EasyDict
from training import misc
import math

#************************************************************************************************************
# return the shape of a tensor as a list
def get_shape(x):
    shape, dyn_shape = x.shape.as_list().copy(), tf.shape(x)
    for index, dim in enumerate(shape):
        if dim is None:
            shape[index] = dyn_shape[index]
    return shape

#************************************************************************************************************
# return dimensions of elements in a tensor
def element_dim(x):
    return np.prod(get_shape(x)[1:])

#************************************************************************************************************
# convert tensors to 2d
def to_2d(x, mode):
    shape = get_shape(x)
    if len(shape) == 2:
        return x
    if mode == "last":
        return tf.reshape(x, [-1, shape[-1]])
    else:
        return tf.reshape(x, [shape[0], element_dim(x)])

#************************************************************************************************************
# get/create a weight tensor for a convolution or fully-connected layer
def get_weight(shape, gain = 1, use_wscale = True, lrmul = 1, weight_var = "weight"):
    fan_in = np.prod(shape[:-1])
    he_std = gain / np.sqrt(fan_in)

    # Equalized learning rate
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable(weight_var, shape = shape, initializer = init) * runtime_coef

#************************************************************************************************************
# fully-connected layer
def dense_layer(x, dim, gain = 1, use_wscale = True, lrmul = 1, weight_var = "weight", name = None):
    if name is not None:
        weight_var = "{}_{}".format(weight_var, name)

    if len(get_shape(x)) > 2:
        x = to_2d(x, "first")

    w = get_weight([get_shape(x)[1], dim], gain = gain, use_wscale = use_wscale,
        lrmul = lrmul, weight_var = weight_var)

    return tf.matmul(x, w)
#************************************************************************************************************

# apply bias and activation function
def apply_bias_act(x, act = "linear", alpha = None, gain = None, lrmul = 1, bias_var = "bias", name = None):
    if name is not None:
        bias_var = "{}_{}".format(bias_var, name)
    b = tf.get_variable(bias_var, shape = [get_shape(x)[1]], initializer = tf.initializers.zeros()) * lrmul
    return fused_bias_act(x, b = b, act = act, alpha = alpha, gain = gain)

#************************************************************************************************************
# normalization types instance, batch or layer-wise.
def norm(x, norm_type, parametric = True):
    if norm_type == "instance":
        x = tf.contrib.layers.instance_norm(x, data_format = "NCHW", center = parametric, scale = parametric)
    elif norm_type == "batch":
        x = tf.contrib.layers.batch_norm(x, data_format = "NCHW", center = parametric, scale = parametric)
    elif norm_type == "layer":
        x = tf.contrib.layers.layer_norm(inputs = x, begin_norm_axis = -1, begin_params_axis = -1)
    return x

#************************************************************************************************************
# normalize tensor according to the integration type
def attention_normalize(x, num, integration, norm):
    shape = get_shape(x)
    x = tf.reshape(x, [-1, num] + get_shape(x)[1:])
    x = tf.cast(x, tf.float32)

    norm_axis = 1 if norm == "instance" else 2

    if integration in ["add", "both"]:
        x -= tf.reduce_mean(x, axis = norm_axis, keepdims = True)
    if integration in ["mul", "both"]:
        x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis = norm_axis, keepdims = True) + 1e-8)

    # return x to its original shape
    x = tf.reshape(x, shape)
    return x

#************************************************************************************************************
# minibatch standard deviation layer see StyleGAN for details.
def minibatch_stddev_layer(x, group_size = 4, num_new_features = 1, dims = 2):
    shape = get_shape(x) 
    last_dims = [shape[3]] if dims == 2 else []
    group_size = tf.minimum(group_size, shape[0])
    y = tf.reshape(x, [group_size, -1, num_new_features, shape[1]//num_new_features, shape[2]] + last_dims) 
    y = tf.cast(y, tf.float32)
    y -= tf.reduce_mean(y, axis = 0, keepdims = True) 
    y = tf.reduce_mean(tf.square(y), axis = 0) 
    y = tf.sqrt(y + 1e-8) 
    y = tf.reduce_mean(y, axis = [2, 3] + ([4] if dims == 2 else []), keepdims = True) 
    y = tf.reduce_mean(y, axis = [2]) 
    y = tf.tile(y, [group_size, 1, shape[2]] + last_dims) 
    return tf.concat([x, y], axis = 1) 

#************************************************************************************************************
# create a random dropout mask
def random_dp_binary(shape, dropout):
    if dropout == 0:
        return tf.ones(shape)
    eps = tf.random.uniform(shape)
    keep_mask = (eps >= dropout)
    return keep_mask

#************************************************************************************************************
# perform dropout
def dropout(x, dp, noise_shape = None):
    if dp is None or dp == 0.0:
        return x
    return tf.nn.dropout(x, keep_prob = 1.0 - dp, noise_shape = noise_shape)

#************************************************************************************************************
# set a mask for logits
def logits_mask(x, mask):
    return x + tf.cast(1 - tf.cast(mask, tf.int32), tf.float32) * -10000.0


#************************************************************************************************************
# 2d linear embeddings
def get_linear_embeddings(size, dim, num, rng = 1.0):
    pi = tf.constant(math.pi)
    theta = tf.range(0, pi, pi / num)
    dirs = tf.stack([tf.cos(theta), tf.sin(theta)], axis = -1)
    embs = tf.get_variable(name = "emb", shape = [num, int(dim / num)],
        initializer = tf.random_uniform_initializer())

    c = tf.linspace(-rng, rng, size)
    x = tf.tile(tf.expand_dims(c, axis = 0), [size, 1])
    y = tf.tile(tf.expand_dims(c, axis = 1), [1, size])
    xy = tf.stack([x,y], axis = -1)

    lens = tf.reduce_sum(tf.expand_dims(xy, axis = 2) * dirs, axis = -1, keepdims = True)
    emb = tf.reshape(lens * embs, [size, size, dim])
    return emb

#************************************************************************************************************
# construct sinusoidal embeddings spanning the 2d space
def get_sinusoidal_embeddings(size, dim, num = 2):
    if num == 2:
        c = tf.expand_dims(tf.to_float(tf.linspace(-1.0, 1.0, size)), axis = -1)
        i = tf.to_float(tf.range(int(dim / 4)))

        peSin = tf.sin(c / (tf.pow(10000.0, 4 * i / dim)))
        peCos = tf.cos(c / (tf.pow(10000.0, 4 * i / dim)))

        peSinX = tf.tile(tf.expand_dims(peSin, axis = 0), [size, 1, 1])
        peCosX = tf.tile(tf.expand_dims(peCos, axis = 0), [size, 1, 1])
        peSinY = tf.tile(tf.expand_dims(peSin, axis = 1), [1, size, 1])
        peCosY = tf.tile(tf.expand_dims(peCos, axis = 1), [1, size, 1])

        emb = tf.concat([peSinX, peCosX, peSinY, peCosY], axis = -1)
    else:
        pi = tf.constant(math.pi)
        theta = tf.range(0, pi, pi / num)
        dirs = tf.stack([tf.cos(theta), tf.sin(theta)], axis = -1)

        c = tf.linspace(-1.0, 1.0, size)
        x = tf.tile(tf.expand_dims(c, axis = 0), [size, 1])
        y = tf.tile(tf.expand_dims(c, axis = 1), [1, size])
        xy = tf.stack([x,y], axis = -1)

        lens = tf.reduce_sum(tf.expand_dims(xy, axis = -2) * dirs, axis = -1, keepdims = True)

        i = tf.to_float(tf.range(int(dim / (2 * num))))
        sins = tf.sin(lens / (tf.pow(10000.0, 2 * num * i / dim)))
        coss = tf.cos(lens / (tf.pow(10000.0, 2 * num * i / dim)))
        emb = tf.reshape(tf.concat([sins, coss], axis = -1), [size, size, dim])

    return emb

#************************************************************************************************************
# construct positional embeddings with different types (sinusoidal, linear or trainable)    
def get_positional_embeddings(max_res, dim, pos_type = "sinus", dir_num = 2, init = "uniform", shared = False):
    embs = []
    initializer = tf.random_uniform_initializer() if init == "uniform" else tf.initializers.random_normal()
    for res in range(max_res + 1):
        with tf.variable_scope("pos_emb%d" % res):
            size = 2 ** res
            if pos_type == "sinus":
                emb = get_sinusoidal_embeddings(size, dim, num = dir_num)
            elif pos_type == "linear":
                emb = get_linear_embeddings(size, dim, num = dir_num)
            elif pos_type == "trainable2d":
                emb = tf.get_variable(name = "emb", shape = [size, size, dim], initializer = initializer)
            else: # pos_type == "trainable"
                xemb = tf.get_variable(name = "x_emb", shape = [size, int(dim / 2)], initializer = initializer)
                yemb = xemb if shared else tf.get_variable(name = "y_emb", shape = [size, int(dim / 2)],
                    initializer = initializer)
                xemb = tf.tile(tf.expand_dims(xemb, axis = 0), [size, 1, 1])
                yemb = tf.tile(tf.expand_dims(yemb, axis = 1), [1, size, 1])
                emb = tf.concat([xemb, yemb], axis = -1)
            embs.append(emb)
    return embs

#************************************************************************************************************
def get_embeddings(size, dim, init = "uniform", name = None):
    initializer = tf.random_uniform_initializer() if init == "uniform" else tf.initializers.random_normal()
    with tf.variable_scope(name):
        emb = tf.get_variable(name = "emb", shape = [size, dim], initializer = initializer)
    return emb

#************************************************************************************************************
def get_relative_embeddings(l, dim, embs):
    diffs = tf.expand_dims(tf.range(l), axis = -1) - tf.range(l)
    diffs -= tf.reduce_min(diffs)
    ret = tf.gather(embs, tf.reshape(diffs, [-1]))
    ret = tf.reshape(ret, [1, l, l, dim])
    return ret

#************************************************************************************************************
# non-linear layer with a resnet connection optionally perform attention    
def nnlayer(x, dim, act, lrmul = 1, y = None, ff = True, pool = False, name = "", **kwargs):
    shape = get_shape(x)
    _x = x
    # Split attention types only for convention
    if y is not None and y != x: # cross-attention
        x = cross_attention_transformer_block(from_tensor = x, to_tensor = y, dim = dim, name = name, **kwargs)[0]
    elif y is not None and y==x: # self-attention
        x = self_attention_transformer_block(from_tensor = x, to_tensor = y, dim = dim, name = name, **kwargs)[0]

    if ff: # feed-forward
        if pool:
            x = to_2d(x, "last")

        with tf.variable_scope("Dense%s_0" % name):
            x = apply_bias_act(dense_layer(x, dim, lrmul = lrmul), act = act, lrmul = lrmul)
        with tf.variable_scope("Dense%s_1" % name):
            x = apply_bias_act(dense_layer(x, dim, lrmul = lrmul), lrmul = lrmul)

        if pool:
            x = tf.reshape(x, shape)

        x = tf.nn.leaky_relu(x + _x) # resnet connection
    return x

#************************************************************************************************************
# multi-layer perceptron with a nonlinearity 'act'.
# optionally use resnet connections and self-attention.
def mlp(x, resnet, layers_num, dim, act, lrmul, pooling = "mean", transformer = False, norm_type = None, **kwargs):
    shape = get_shape(x)

    if len(get_shape(x)) > 2:
        if pooling == "cnct":
            with tf.variable_scope("Dense_pool"):
                x = apply_bias_act(dense_layer(x, dim), act = act)
        elif pooling == "batch":
            x = to_2d(x, "last")
        else:
            pool_shape = (get_shape(x)[-2], get_shape(x)[-1])
            x = tf.nn.avg_pool(x, pool_shape, pool_shape, padding = "SAME", data_format = "NCHW")
            x = to_2d(x, "first")

    if resnet:
        half_layers_num = int(layers_num / 2)
        for layer_idx in range(half_layers_num):
            y = x if transformer else None
            x = nnlayer(x, dim, act, lrmul, y = y, name = layer_idx, **kwargs)
            x = norm(x, norm_type)

        with tf.variable_scope("Dense%d" % layer_idx):
            x = apply_bias_act(dense_layer(x, dim, lrmul = lrmul), act = act, lrmul = lrmul)

    else:
        for layer_idx in range(layers_num):
            with tf.variable_scope("Dense%d" % layer_idx):
                x = apply_bias_act(dense_layer(x, dim, lrmul = lrmul), act = act, lrmul = lrmul)
                x = norm(x, norm_type)

    x = tf.reshape(x, [-1] + shape[1:-1] + [dim])
    return x


#************************************************************************************************************
# convolution layer with optional upsampling or downsampling
def conv2d_layer(x, dim, kernel, up = False, down = False, resample_kernel = None, gain = 1, use_wscale = True, lrmul = 1, weight_var = "weight"):
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, get_shape(x)[1], dim], gain = gain, use_wscale = use_wscale, lrmul = lrmul, weight_var = weight_var)
    if up:
        x = upsample_conv_2d(x, w, data_format = "NCHW", k = resample_kernel)
    elif down:
        x = conv_downsample_2d(x, w, data_format = "NCHW", k = resample_kernel)
    else:
        x = tf.nn.conv2d(x, w, data_format = "NCHW", strides = [1, 1, 1, 1], padding = "SAME")
    return x

#************************************************************************************************************
# modulated convolution layer (see StyleGAN for details)
def modulated_convolution_layer(x, y, dim, kernel,up = False, down = False,resample_kernel = None, modulate = True, demodulate = True, fused_modconv = True,  
        noconv = False, gain = 1, use_wscale = True, lrmul = 1, weight_var = "weight", mod_weight_var = "mod_weight", mod_bias_var = "mod_bias"):

    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1

    w = get_weight([kernel, kernel, get_shape(x)[1], dim], gain = gain, use_wscale = use_wscale,
        lrmul = lrmul, weight_var = weight_var)
    ww = w[np.newaxis]

    s = dense_layer(y, dim = get_shape(x)[1], weight_var = mod_weight_var) 
    s = apply_bias_act(s, bias_var = mod_bias_var) + 1 
    
    if modulate:
        ww *= s[:, np.newaxis, np.newaxis, :, np.newaxis] 
        if demodulate:
            d = tf.rsqrt(tf.reduce_sum(tf.square(ww), axis = [1, 2, 3]) + 1e-8) 
            ww *= d[:, np.newaxis, np.newaxis, np.newaxis, :] 
    else:
        ww += tf.zeros_like(s[:, np.newaxis, np.newaxis, :, np.newaxis])

    if fused_modconv:
        x = tf.reshape(x, [1, -1, get_shape(x)[-2], get_shape(x)[-1]])  
        w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]), get_shape(ww)[1:4] + [-1])
    else:
        if modulate:
            x *= s[:, :, np.newaxis, np.newaxis] 

    if noconv:
        if up:
            x = upsample_2d(x, k = resample_kernel)
        elif down:
            x = downsample_2d(x, k = resample_kernel)
    else:
        if up:
            x = upsample_conv_2d(x, w, data_format = "NCHW", k = resample_kernel)
        elif down:
            x = conv_downsample_2d(x, w, data_format = "NCHW", k = resample_kernel)
        else:
            x = tf.nn.conv2d(x, w, data_format = "NCHW", strides = [1,1,1,1], padding = "SAME")

    if fused_modconv:
        x = tf.reshape(x, [-1, dim] + get_shape(x)[-2:]) 
    elif modulate and demodulate:
        x *= d[:, :, np.newaxis, np.newaxis] 

    return x

#************************************************************************************************************
# validate transformer input shape (reshape to 2d)
def process_input(t, t_pos, t_len, name):
    shape = get_shape(t)

    if len(shape) > 3:
        misc.error("Transformer {}_tensor has {} shape. should be up to 3 dims.".format(name, shape))
    elif len(shape) == 3:
        batch_size, t_len, _ = shape
    else:
        if t_len is None:
            misc.error("If {}_tensor has two dimensions, must specify {}_len.".format(name, name))
        batch_size = tf.cast(shape[0] / t_len, tf.int32)

    # reshape tensors to 2d
    t = to_2d(t, "last")
    if t_pos is not None:
        t_pos = tf.tile(to_2d(t_pos, "last"), [batch_size, 1])

    return t, t_pos, shape, t_len, batch_size

#************************************************************************************************************
# transpose tensor to scores
def transpose_for_scores(x, batch_size, num_heads, elem_num, head_size):
    x = tf.reshape(x, [batch_size, elem_num, num_heads, head_size])
    x = tf.transpose(x, [0, 2, 1, 3]) 
    return x

#************************************************************************************************************
# calculate attention probabilities using tf.nn.softmax
def compute_probs(scores, dp):
    probs = tf.nn.softmax(scores)
    shape = get_shape(probs)
    shape[-2] = 1
    probs = dropout(probs, dp / 2)
    probs = dropout(probs, dp / 2, shape)
    return probs

#************************************************************************************************************
# scale and bias the given tensor in transformer
def integrate(tensor, tensor_len, control, integration, norm):
    dim = get_shape(tensor)[-1]

    # normalization
    if norm is not None:
        tensor = attention_normalize(tensor, tensor_len, integration, norm)

    # compute gain/bias
    control_dim = {"add": dim, "mul": dim, "both": 2 * dim}[integration]
    bias = gain = control = apply_bias_act(dense_layer(control, control_dim, name = "out"), name = "out")
    if integration == "both":
        gain, bias = tf.split(control, 2, axis = -1)

    # modulation
    if integration != "add":
        tensor *= (gain + 1)
    if integration != "mul":
        tensor += bias

    return tensor

#************************************************************************************************************
# computing centroid assignments (see k-means algorithm for details, Lloyd et al. 1982)
def compute_assignments(att_probs):
    centroid_assignments = (att_probs / (tf.reduce_sum(att_probs, axis = -2, keepdims = True) + 1e-8))
    centroid_assignments = tf.transpose(centroid_assignments, [0, 1, 3, 2]) # [B, N, T, F]
    return centroid_assignments

#************************************************************************************************************
# using centroids for attention calculations (see k-means algorithm for details, Lloyd et al. 1982)
def compute_centroids(_queries, queries, to_from, to_len, from_len, batch_size, num_heads, 
        size_head, parametric):
    
    dim = 2 * size_head
    from_elements = tf.concat([_queries, queries - _queries], axis = -1)
    from_elements = transpose_for_scores(from_elements, batch_size, num_heads, from_len, dim) 

    if to_from is not None:

        if get_shape(to_from)[-2] < to_len:
            s = int(math.sqrt(get_shape(to_from)[-2]))
            to_from = upsample_2d(tf.reshape(to_from, [batch_size * num_heads, s, s, from_len]), factor = 2, data_format = "NHWC")
            to_from = tf.reshape(to_from, [batch_size, num_heads, to_len, from_len])

        if get_shape(to_from)[-1] < from_len:
            s = int(math.sqrt(get_shape(to_from)[-1]))
            to_from = upsample_2d(tf.reshape(to_from, [batch_size * num_heads, to_len, s, s]), factor = 2, data_format = "NCHW")
            to_from = tf.reshape(to_from, [batch_size, num_heads, to_len, from_len])


        to_centroids = tf.matmul(to_from, from_elements)

    if to_from is None or parametric:
        if parametric:
            to_centroids = tf.tile(tf.get_variable("toasgn_init", shape = [1, num_heads, to_len, dim],
                initializer = tf.initializers.random_normal()), [batch_size, 1, 1, 1])
        else:
            to_centroids = apply_bias_act(dense_layer(queries, dim * num_heads, name = "key2"), name = "key2")
            to_centroids = transpose_for_scores(to_centroids, batch_size, num_heads, dim, dim)

    return from_elements, to_centroids

#************************************************************************************************************
# construct cross-attention-transformer used between latents and images
def cross_attention_transformer_block(
        dim,                                  # dimension of the layer
        from_tensor,        to_tensor,        
        from_len = None,    to_len = None,    
        from_pos = None,    to_pos = None,    # the positional encodings for the cross attention tensors
        num_heads = 1,                        # number of attention heads (default value is 1 for slater)
        att_dp = 0.12,                        # dropout rate of attention
        att_mask = None,                      # Attention mask to block from/to elements [batch_size, from_len, to_len]
        integration = "mul",                  # integration type (default value is 'mul' for slater)
        norm = "layer",                       # normalization type
        kmeans = False,                       # see k-means algorithm (Lloyd et al 1982).
        kmeans_iters = 1,                     # number of k-means iterations per layer
        att_vars = {},                        # variables used in k-means algorithm carried through layers
                                              # suffix
        name = ""): 

    assert from_tensor != to_tensor # be sure for cross-attention
    
    from_tensor, from_pos, from_shape, from_len, batch_size = process_input(from_tensor, from_pos, from_len, "from")
    to_tensor,   to_pos,   to_shape,   to_len,   _          = process_input(to_tensor, to_pos, to_len, "to")

    size_head = int(dim / num_heads)
    to_from = att_vars.get("centroid_assignments")

    with tf.variable_scope("AttLayer_{}".format(name)):
        queries = apply_bias_act(dense_layer(from_tensor, dim, name = "query"), name = "query") 
        keys    = apply_bias_act(dense_layer(to_tensor, dim, name = "key"), name = "key")      
        values  = apply_bias_act(dense_layer(to_tensor, dim, name = "value"), name = "value")  
        _queries = queries

        if from_pos is not None:
            queries += apply_bias_act(dense_layer(from_pos, dim, name = "from_pos"), name = "from_pos")
        if to_pos is not None:
            keys += apply_bias_act(dense_layer(to_pos, dim, name = "to_pos"), name = "to_pos")

        if kmeans:
            from_elements, to_centroids = compute_centroids(_queries, queries, to_from,
                to_len, from_len, batch_size, num_heads, size_head, parametric = True)

        values = transpose_for_scores(values, batch_size, num_heads, to_len, size_head)     
        queries = transpose_for_scores(queries, batch_size, num_heads, from_len, size_head) 
        keys = transpose_for_scores(keys, batch_size, num_heads, to_len, size_head)         
        att_scores = tf.matmul(queries, keys, transpose_b = True)                           
        att_probs = None

        for i in range(kmeans_iters):
            with tf.variable_scope("iter_{}".format(i)):
                if kmeans:
                    if i > 0:
                        to_from = compute_assignments(att_probs)
                        to_centroids = tf.matmul(to_from, from_elements)

                    w = tf.get_variable(name = "st_weights", shape = [num_heads, 1, get_shape(from_elements)[-1]],
                        initializer = tf.ones_initializer())
                    att_scores = tf.matmul(from_elements * w, to_centroids, transpose_b = True)

                att_scores = tf.multiply(att_scores, 1.0 / math.sqrt(float(size_head)))
                if att_mask is not None:
                    att_scores = logits_mask(att_scores, tf.expand_dims(att_mask, axis = 1))
                att_probs = compute_probs(att_scores, att_dp)



        if kmeans:
            to_from = compute_assignments(att_probs)

        control = tf.matmul(att_probs, values) 
        control = tf.transpose(control, [0, 2, 1, 3]) 
        control = tf.reshape(control, [batch_size * from_len, dim]) 
        from_tensor = integrate(from_tensor, from_len, control, integration, norm)

    if len(from_shape) > 2:
        from_tensor = tf.reshape(from_tensor, from_shape)

    return from_tensor, att_probs, {"centroid_assignments": to_from}

#************************************************************************************************************
# construct a self-attention-transformer block
def self_attention_transformer_block(
        dim,                                  # dimension of the layer
        from_tensor,        to_tensor,        
        from_len = None,    to_len = None,    
        from_pos = None,    to_pos = None,    # the positional encodings for the cross attention tensors
        num_heads = 1,                        # number of attention heads (default value is 1 for slater)
        att_dp = 0.12,                        # dropout rate of attention
        att_mask = None,                      # Attention mask to block from/to elements [batch_size, from_len, to_len]
        integration = "mul",                  # integration type (default value is 'mul' for slater)
        norm = "layer",                       # normalization type
        kmeans = False,                       # see k-means algorithm (Lloyd et al 1982).
        kmeans_iters = 1,                     # number of k-means iterations per layer
        att_vars = {},                        # variables used in k-means algorithm carried through layers
                                              # suffix
        name = ""): 
    
    assert from_tensor == to_tensor # be sure for self-attention
    
    from_tensor, from_pos, from_shape, from_len, batch_size = process_input(from_tensor, from_pos, from_len, "from")
    to_tensor,   to_pos,   to_shape,   to_len,   _          = process_input(to_tensor, to_pos, to_len, "to")

    size_head = int(dim / num_heads)
    to_from = att_vars.get("centroid_assignments")

    with tf.variable_scope("AttLayer_{}".format(name)):
        queries = apply_bias_act(dense_layer(from_tensor, dim, name = "query"), name = "query") 
        keys    = apply_bias_act(dense_layer(to_tensor, dim, name = "key"), name = "key")       
        values  = apply_bias_act(dense_layer(to_tensor, dim, name = "value"), name = "value")   
        _queries = queries

        if from_pos is not None:
            queries += apply_bias_act(dense_layer(from_pos, dim, name = "from_pos"), name = "from_pos")
        if to_pos is not None:
            keys += apply_bias_act(dense_layer(to_pos, dim, name = "to_pos"), name = "to_pos")

        if kmeans:
            from_elements, to_centroids = compute_centroids(_queries, queries, to_from,
                to_len, from_len, batch_size, num_heads, size_head, parametric = True)

        values = transpose_for_scores(values, batch_size, num_heads, to_len, size_head)    
        queries = transpose_for_scores(queries, batch_size, num_heads, from_len, size_head) 
        keys = transpose_for_scores(keys, batch_size, num_heads, to_len, size_head)       
        att_scores = tf.matmul(queries, keys, transpose_b = True)                       
        att_probs = None

        for i in range(kmeans_iters):
            with tf.variable_scope("iter_{}".format(i)):
                if kmeans:
                    if i > 0:
                        to_from = compute_assignments(att_probs)
                        to_centroids = tf.matmul(to_from, from_elements)

                    w = tf.get_variable(name = "st_weights", shape = [num_heads, 1, get_shape(from_elements)[-1]],
                        initializer = tf.ones_initializer())
                    att_scores = tf.matmul(from_elements * w, to_centroids, transpose_b = True)

                att_scores = tf.multiply(att_scores, 1.0 / math.sqrt(float(size_head)))
                if att_mask is not None:
                    att_scores = logits_mask(att_scores, tf.expand_dims(att_mask, axis = 1))
                att_probs = compute_probs(att_scores, att_dp)


        if kmeans:
            to_from = compute_assignments(att_probs)

        control = tf.matmul(att_probs, values) 
        control = tf.transpose(control, [0, 2, 1, 3])
        control = tf.reshape(control, [batch_size * from_len, dim]) 
        from_tensor = integrate(from_tensor, from_len, control, integration, norm)

    if len(from_shape) > 2:
        from_tensor = tf.reshape(from_tensor, from_shape)

    return from_tensor, att_probs, {"centroid_assignments": to_from}

#************************************************************************************************************
# generator
#************************************************************************************************************
def G_slater(
    latents_in,                               # input latent vectors (z)
    labels_in,                                # labels (not used in slater)
    is_training             = False,          # enables training options
    is_validation           = False,          # change truncation constant when mode is validation
    is_template_graph       = False,          # True = template graph constructed by the Network class, False = actual evaluation
    components              = EasyDict(),     # one large generator network consists of synthesizer and mapper
    mapping_func            = "G_mapper",     # function name of the mapper network
    synthesis_func          = "G_synthesizer",# function name of the synthesizer network
    truncation_psi          = 0.65,           # truncation trick hyperparameter
    truncation_cutoff       = None,           # cut-off layer for truncation (disabled by default)
    truncation_psi_val      = None,           # use truncation in validation (disabled by default)
    truncation_cutoff_val   = None,           # cut-off layer for truncation in validation (disabled by default)
    dlatent_avg_beta        = 0.995,          # moving average of latent vectors when taking mean
    style_mixing            = 0.9,            # style-mixing probablity during training (change latent vectors between different resolutions)
    component_mixing        = 0.0,            # component mixing (not used in slater)
    component_dropout       = 0.0,            # component dropout (not used in slater)
    **kwargs):

    # argument validation
    assert not is_training or not is_validation
    assert isinstance(components, EasyDict)
    latents_in = tf.cast(latents_in, tf.float32)
    labels_in = tf.cast(labels_in, tf.float32)

    # general settings
    if is_validation:
        truncation_psi = truncation_psi_val
        truncation_cutoff = truncation_cutoff_val
    if is_training or (truncation_psi is not None and not tflib.is_tf_expression(truncation_psi) and truncation_psi == 1):
        truncation_psi = None
    if is_training:
        truncation_cutoff = None
    if not is_training or (dlatent_avg_beta is not None and not tflib.is_tf_expression(dlatent_avg_beta) and dlatent_avg_beta == 1):
        dlatent_avg_beta = None
    if not is_training or (style_mixing is not None and not tflib.is_tf_expression(style_mixing) and style_mixing <= 0):
        style_mixing = None
    if not is_training or (component_mixing is not None and not tflib.is_tf_expression(component_mixing) and component_mixing <= 0):
        component_mixing = None
    if not is_training:
        kwargs["attention_dropout"] = 0.0

    # define variables
    k = kwargs["components_num"]
    latent_size = kwargs["latent_size"]
    dlatent_size = kwargs["dlatent_size"]

    latents_num = k + 1 # k local latents + 1 global latent
    
    # setup networks
    # set synthesis network
    if "synthesis" not in components:
        components.synthesis = tflib.Network("G_synthesizer", func_name = globals()[synthesis_func], **kwargs)
    num_layers = components.synthesis.input_shape[2]
    # set mapper network
    if "mapping" not in components:
        components.mapping = tflib.Network("G_mapper", func_name = globals()[mapping_func],
            dlatent_broadcast = num_layers, **kwargs)


    latents_in.set_shape([None, latents_num, latent_size])
    batch_size = get_shape(latents_in)[0]

    # initialization of trainable positional encodings for latents
    latent_pos = get_embeddings(k, dlatent_size, name = "ltnt_emb")
    component_mask = random_dp_binary([batch_size, k], component_dropout)
    component_mask = tf.expand_dims(component_mask, axis = 1)

    dlatent_avg = tf.get_variable("dlatent_avg", shape = [dlatent_size], 
        initializer = tf.initializers.zeros(), trainable = False)


    # evaluate mapping network
    dlatents = components.mapping.get_output_for(latents_in, labels_in, latent_pos, 
        component_mask, is_training = is_training, **kwargs)
    dlatents = tf.cast(dlatents, tf.float32)

    # find average intermediate latent vectors
    if dlatent_avg_beta is not None:
        with tf.variable_scope("DlatentAvg"):
            batch_avg = tf.reduce_mean(dlatents[:, :, 0], axis = [0, 1])
            update_op = tf.assign(dlatent_avg, tflib.lerp(batch_avg, dlatent_avg, dlatent_avg_beta))
            with tf.control_dependencies([update_op]):
                dlatents = tf.identity(dlatents)

    def mixing(latents_in, dlatents, prob, num, idx):
        if prob is None or prob == 0:
            return dlatents

        with tf.variable_scope("StyleMix"):
            latents2 = tf.random_normal(get_shape(latents_in))
            dlatents2 = components.mapping.get_output_for(latents2, labels_in, latent_pos, 
                component_mask, is_training = is_training, **kwargs)
            dlatents2 = tf.cast(dlatents2, tf.float32)
            mixing_cutoff = tf.cond(
                tf.random_uniform([], 0.0, 1.0) < prob,
                lambda: tf.random_uniform([], 1, num, dtype = tf.int32),
                lambda: num)
            dlatents = tf.where(tf.broadcast_to(idx < mixing_cutoff, get_shape(dlatents)), dlatents, dlatents2)
        return dlatents

    # style-mixing (see stylegan for details)
    layer_idx = np.arange(num_layers)[np.newaxis, np.newaxis, :, np.newaxis]
    dlatents = mixing(latents_in, dlatents, style_mixing, num_layers, layer_idx)

    # truncation trick (see stylegan for details)
    if truncation_psi is not None:
        with tf.variable_scope("Truncation"):
            layer_idx = np.arange(num_layers)[np.newaxis, np.newaxis, :, np.newaxis]
            layer_psi = np.ones(layer_idx.shape, dtype = np.float32)
            if truncation_cutoff is None:
                layer_psi *= truncation_psi
            else:
                layer_psi = tf.where(layer_idx < truncation_cutoff, layer_psi * truncation_psi, layer_psi)
            dlatents = tflib.lerp(dlatent_avg, dlatents, layer_psi)

    # network evaluation
    imgs_out, maps_out = components.synthesis.get_output_for(dlatents, latent_pos, component_mask,
        is_training = is_training, force_clean_graph = is_template_graph, **kwargs)

    imgs_out = tf.identity(imgs_out, name = "images_out") # [batch_size, number of channels, height, width]
    maps_out = tf.identity(maps_out, name = "maps_out") # [batch_size, number of local latent vectors, number of layers, heads_num, height, width]
    ret = (imgs_out, maps_out)

    return ret

#************************************************************************************************************
# mapper network (M)
#************************************************************************************************************    
def G_mapper(
    latents_in,                             # random latent vectors (z) 
    labels_in,                              # labels (not used in SLATER)
    latent_pos,                             # positional embeddings for latents (used in self-attention-transformer)
    component_mask,                         # drop out mask (not used in SLATER)
    components_num          = 16,           # number of local latent components z_1,...,z_k
    latent_size             = 512,          # latent dimensionality per component.
    dlatent_size            = 512,          # disentangled latent dimensionality
    label_size              = 0,            # label dimensionality, 0 if no labels (no labels in SLATER)
    dlatent_broadcast       = None,         # tile latent vectors to num_layer to control style in all layers (used in SLATER)
    normalize_latents       = True,         # normalize latent vectors (z)
    mapping_layersnum       = 8,            # number of mapping layers
    mapping_dim             = None,         # number of activations in the mapping layers
    mapping_lrmul           = 0.01,         # learning rate multiplier for the mapping layers
    mapping_nonlinearity    = "lrelu",      # activation function
    num_heads               = 1,            # number of attention heads
    attention_dropout       = 0.12,         # attention dropout rate
    **_kwargs):

    act = mapping_nonlinearity
    k = components_num
    latents_num = k + 1 # total number of latents = num(local_latents) + global_latent

    net_dim = mapping_dim
    layersnum = mapping_layersnum
    lrmul = mapping_lrmul
    ltnt2ltnt = True
    resnet = True

    # input tensors
    latents_in.set_shape([None, latents_num, latent_size])
    labels_in.set_shape([None, label_size])
    latent_pos.set_shape([k, dlatent_size])
    component_mask.set_shape([None, 1, k])

    batch_size = get_shape(latents_in)[0]

    x = latents_in

    if net_dim is None:
        net_dim = dlatent_size
    else:
        x = to_2d(x, "last")
        x = apply_bias_act(dense_layer(x, net_dim, name = "map_start"), name = "map_start")
        x = tf.reshape(x, [batch_size, latents_num, net_dim])
        if latent_pos is not None:
            latent_pos = apply_bias_act(dense_layer(latent_pos, net_dim, name = "map_pos"), name = "map_pos")

    if label_size:
        with tf.variable_scope("LabelConcat"):
            w = tf.get_variable("weight", shape = [label_size, latent_size], initializer = tf.initializers.random_normal())
            l = tf.tile(tf.expand_dims(tf.matmul(labels_in, w), axis = 1), (1, latents_num, 1))
            x = tf.concat([x, l], axis = 1)

    # splitting latent vectors to global and local
    x, g = tf.split(x, [k, 1], axis = 1)
    g = tf.squeeze(g, axis = 1)

    # normalize latent vectors
    if normalize_latents:
        with tf.variable_scope("Normalize"):
            x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis = -1, keepdims = True) + 1e-8)

    mlp_kwargs = {}
    if ltnt2ltnt:
        mlp_kwargs.update({         "transformer": ltnt2ltnt,
           "num_heads": 1,          "att_dp": attention_dropout,
           "from_pos": latent_pos,  "to_pos": latent_pos,
           "from_len": k,           "to_len": k,
        })

    # mapper layers
    if k == 0:
        x = tf.zeros([batch_size, 0, net_dim])
    else:
        x = mlp(x, resnet, layersnum, net_dim, act, lrmul, pooling = "batch",
            att_mask = component_mask, **mlp_kwargs)

    with tf.variable_scope("global"):
        # mapping global latent separately
        g = mlp(g, resnet, layersnum, net_dim, act, lrmul)
    # concatenate back global and local latent vectors
    x = tf.concat([x, tf.expand_dims(g, axis = 1)], axis = 1)

    # tile latent vectors to all resolution layers to control style and local features in each layer.
    if dlatent_broadcast is not None:
        with tf.variable_scope("Broadcast"):
            x = tf.tile(x[:, :, np.newaxis], [1, 1, dlatent_broadcast, 1])


    x = tf.identity(x, name = "dlatents_out")
    return x # [batch size, num_layers, number of latent vectors, latent dimension]

#************************************************************************************************************
# synthesizer network (G)
#************************************************************************************************************
def G_synthesizer(
    dlatents_in,                        # intermediate latent vectors (W) : k local + 1 global
    latent_pos,                         # positional embeddings for latents
    component_mask,                     # component dropout mask (not used in slater)
    dlatent_size        = 512,          # latent dimension
    pos_dim             = None,         # positional embeddings dimension
    num_channels        = 3,            # number of channels (rgb default)
    resolution          = 1024,         # resolution (overwritten by used dataset)
    fmap_base           = 16 << 10,     # overall multiplier for the network dimension
    fmap_decay          = 1.0,          # log2 network dimension reduction when doubling the resolution
    fmap_min            = 1,            # minimum network dimension in any layer
    fmap_max            = 512,          # maximum network dimension in any layer
    architecture        = "resnet",     # resnet connections used in slater
    nonlinearity        = "lrelu",      # activation function
    resample_kernel     = [1, 3, 3, 1], # low-pass filter to apply when resampling activations
    fused_modconv       = True,         # use modulated convolution layer
    style               = True,         # use global style modulation (see StyleGAN)
    local_noise         = True,         # add stochastic noise to activations
    randomize_noise     = True,         # change noise variables every time
    components_num      = 16,           # number of local latent vectors
    num_heads           = 1,            # number of attention heads
    attention_dropout   = 0.12,         # attention dropout rate
    integration         = "mul",        # feature integration type: additive, multiplicative or both
    norm                = None,         # feature normalization type (optional): instance, batch or layer
    kmeans              = False,        # track and update image-to-latents assignment centroids
    kmeans_iters        = 1,            # number of K-means iterations per layer (see k-means algorithm)
    start_res           = 0,            # transformer minimum resolution layer to be used at
    end_res             = 8,            # transformer maximum resolution layer to be used at
    use_pos             = True,         # use positional encoding for latents
    pos_type            = "sinus",      # positional encoding type: linear, sinus, trainable, trainable2d
    pos_init            = "uniform",    # positional encoding initialization distribution
    pos_directions_num  = 2,            # positional encoding number of spatial directions
    **_kwargs):                         

    # settings
    k = components_num
    act = nonlinearity
    latents_num = k + 1 # k local + 1 global latent
    resolution_log2 = int(np.log2(resolution))
    num_layers = resolution_log2 * 2 - 1

    if pos_dim is None:
        pos_dim = dlatent_size

    assert resolution == 2**resolution_log2 and resolution >= 4

    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)

    def get_global(dlatents, res):
        return dlatents[:, -1]

    # inputs
    dlatents_in.set_shape([None, latents_num, num_layers, dlatent_size])
    component_mask.set_shape([None, 1, k])
    latent_pos.set_shape([k, dlatent_size])

    if not use_pos:
        latent_pos = None

    batch_size = get_shape(dlatents_in)[0]

    # positional encodings for the images
    grid_poses = get_positional_embeddings(resolution_log2, pos_dim, pos_type, pos_directions_num, pos_init)

    # noise adding to features
    noise_layers = []
    for layer_idx in range(num_layers - 1):
        # Infer layer resolution from its index
        res = (layer_idx + 5) // 2
        batch_multiplier = 1
        noise_shape = [batch_multiplier, 1, 2**res, 2**res]

        # local noise variables
        noise_layers.append(tf.get_variable("noise%d" % layer_idx, shape = noise_shape,
            initializer = tf.initializers.random_normal(), trainable = False))

    def add_noise(x, layer_idx):
        if randomize_noise:
            shape = get_shape(x) 
            shape[1] = 1
            noise = tf.random_normal(shape)
        else:
            noise = noise_layers[layer_idx]
        strength = tf.get_variable("noise_strength", shape = [], initializer = tf.initializers.zeros())
        x += strength * noise
        return x

    def synthesizer_layer(x, dlatents, layer_idx, dim, kernel, att_vars, up = False):
        att_map = None
        res = (layer_idx + 5) // 2
        dlatent_global = get_global(dlatents_in, res)[:, layer_idx + 1]
        new_dlatents = None
        if dlatents is None:
            dlatents = dlatents_in[:, :-1, layer_idx + 1]

        _fused_modconv, noconv = fused_modconv, False

        # perform modulated_convolution
        x = modulated_convolution_layer(x, dlatent_global, dim, kernel, up = up,
            resample_kernel = resample_kernel, fused_modconv = _fused_modconv, modulate = style, noconv = noconv)
        shape = get_shape(x)

        if res >= start_res and res < end_res:
            x = tf.transpose(tf.reshape(x, [shape[0], shape[1], shape[2] * shape[3]]), [0, 2, 1])

            # arguments used in attention-blocks see run_network.py for explanation of each argument
            kwargs = {
                "num_heads": num_heads,
                "integration": integration,     
                "norm": norm,                   
                "att_mask": component_mask,     
                "att_dp": attention_dropout,    
                "from_pos": grid_poses[res],    
                "to_pos": latent_pos,           
                "kmeans": kmeans,               
                "kmeans_iters": kmeans_iters,  
                "att_vars": att_vars,                                                           
            }
            # cross Attention Transformer Layer information flow from local latent vectors to images
            x, att_map, att_vars = cross_attention_transformer_block(from_tensor = x, to_tensor = dlatents, dim = dim,
                name = "l2n", **kwargs)

            x = tf.reshape(tf.transpose(x, [0, 2, 1]), shape)

        # add local stochastic noise to image features
        if local_noise:
            x = add_noise(x, layer_idx)
        x = apply_bias_act(x, act = act)

        return x, new_dlatents, att_map, att_vars

    def block(x, res, dlatents, dim, att_vars, up = True): 
        t = x
        # upsampling + convolution + cross attention transformer
        with tf.variable_scope("Conv0_up"):
            x, dlatents, att_map1, att_vars = synthesizer_layer(x, dlatents, layer_idx = res*2-5,
                dim = dim, kernel = 3, up = up, att_vars = att_vars)
        
        # modulated convolution layer + cross attention transformer
        with tf.variable_scope("Conv1"):
            x, dlatents, att_map2, att_vars = synthesizer_layer(x, dlatents, layer_idx = res*2-4,
                dim = dim, kernel = 3, att_vars = att_vars)
        
        # resnet connection
        if architecture == "resnet":
            with tf.variable_scope("Skip"):
                t = conv2d_layer(t, dim = dim, kernel = 1, up = up, resample_kernel = resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))

        att_maps = [att_map1, att_map2]
        return x, dlatents, att_maps, att_vars

    def upsample(y):
        with tf.variable_scope("Upsample"):
            return upsample_2d(y, k = resample_kernel)


    def torgb(t, y, res, dlatents): 
        with tf.variable_scope("ToRGB"):
            if res == resolution_log2:
                if res <= end_res:
                    with tf.variable_scope("extraLayer"):
                        t = modulated_convolution_layer(t, dlatents[:, res*2-3], dim = nf(res-1),
                            kernel = 3, fused_modconv = fused_modconv, modulate = style)

            t = modulated_convolution_layer(t, dlatents[:, res*2-2], dim = num_channels,
                kernel = 1, demodulate = False, fused_modconv = fused_modconv, modulate = style)
            t = apply_bias_act(t)

            if y is not None:
                t += y

            return t

    imgs_out, dlatents, att_maps = None, None, []
    att_vars = {"centroid_assignments": None}

    with tf.variable_scope("4x4"):
        with tf.variable_scope("Const"):
            stem_size = 1
            x = tf.get_variable("const", shape = [stem_size, nf(1), 4, 4],
                initializer = tf.initializers.random_normal())
            x = tf.tile(x, [batch_size, 1, 1, 1])

        with tf.variable_scope("Conv"):
            x, dlatents, att_map, att_vars = synthesizer_layer(x, dlatents, layer_idx = 0, dim = nf(1),
                kernel = 3, att_vars = att_vars)
            att_maps.append(att_map)
        if architecture == "skip":
            imgs_out = torgb(x, imgs_out, 2, get_global(dlatents_in, res))

    # main layers
    for res in range(3, resolution_log2 + 1):
        with tf.variable_scope("%dx%d" % (2**res, 2**res)):
            # generator block: transformer, convolution and upsampling
            x, dlatents, _att_maps, att_vars = block(x, res, dlatents, dim = nf(res-1), att_vars = att_vars)
            att_maps += _att_maps

            if architecture == "skip" or res == resolution_log2:
                if architecture == "skip":
                    imgs_out = upsample(imgs_out)
                imgs_out = torgb(x, imgs_out, res, get_global(dlatents_in, res))

    def list2tensor(att_list):
        att_list = [att_map for att_map in att_list if att_map is not None]
        if len(att_list) == 0:
            return None

        maps_out = []
        for att_map in att_list:
            s = int(math.sqrt(get_shape(att_map)[2]))
            att_map = tf.transpose(tf.reshape(att_map, [-1, s, s, k]), [0, 3, 1, 2]) 
            if s < resolution:
                att_map = upsample_2d(att_map, factor = int(resolution / s))
            att_map = tf.reshape(att_map, [-1, num_heads, k, resolution, resolution]) 
            maps_out.append(att_map)

        maps_out = tf.transpose(tf.stack(maps_out, axis = 1), [0, 3, 1, 2, 4, 5]) 
        return maps_out

    maps_out = list2tensor(att_maps)

    return imgs_out, maps_out

#************************************************************************************************************
# discriminator network (D)
#************************************************************************************************************    
def D_slater(
    images_in,                          # input images
    labels_in,                          # input labels (not used in slater)
    label_size          = 0,            # label dimension
    num_channels        = 3,            # rgb
    resolution          = 1024,         # resolution (overwritten by dataset)
    fmap_base           = 16 << 10,     # overall multiplication constant
    fmap_decay          = 1.0,          # log2 network dimension reduction when doubling the resolution
    fmap_min            = 1,            # minimum network dimension in any layer
    fmap_max            = 512,          # maximum network dimension in any layer
    architecture        = "resnet",     # archirecture (resnet in slater)
    nonlinearity        = "lrelu",      # activation function
    mbstd_group_size    = 4,            # minibatch standard deviation size
    mbstd_num_features  = 1,            # num of features for minibatch standard deviation layer
    resample_kernel     = [1, 3, 3, 1], # low-pass filter used for activations
    **_kwargs):

    act = nonlinearity
    resolution_log2 = int(np.log2(resolution))


    assert architecture in ["orig", "skip", "resnet"]
    assert resolution == 2**resolution_log2 and resolution >= 4

    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)

    images_in.set_shape([None, num_channels, resolution, resolution])
    labels_in.set_shape([None, label_size])

    images_in = tf.cast(images_in, tf.float32)
    labels_in = tf.cast(labels_in, tf.float32)

    def fromrgb(x, y, res):
        with tf.variable_scope("FromRGB"):
            t = apply_bias_act(conv2d_layer(y, dim = nf(res-1), kernel = 1), act = act)
            if x is not None:
                t += x
            return t

    def block(x, res, aggregators): 
        t = x
        ksize = 3 
        with tf.variable_scope("Conv0"):
            x = apply_bias_act(conv2d_layer(x, dim = nf(res-1), kernel = ksize), act = act)

        with tf.variable_scope("Conv1_down"):
            x = apply_bias_act(conv2d_layer(x, dim = nf(res-2), kernel = ksize, down = True,
                resample_kernel = resample_kernel), act = act)

        if architecture == "resnet":
            with tf.variable_scope("Skip"):
                t = conv2d_layer(t, dim = nf(res-2), kernel = 1, down = True, resample_kernel = resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))

        return x, aggregators

    def downsample(y):
        with tf.variable_scope("Downsample"):
            return downsample_2d(y, k = resample_kernel)

    x = None
    aggregators = None

    for res in range(resolution_log2, 2, -1):
        with tf.variable_scope("%dx%d" % (2**res, 2**res)):
            if architecture == "skip" or res == resolution_log2:
                x = fromrgb(x, images_in, res)
            x, aggregators = block(x, res, aggregators)
            if architecture == "skip":
                images_in = downsample(images_in)

    with tf.variable_scope("4x4"):
        if architecture == "skip":
            x = fromrgb(x, images_in, 2)
        if mbstd_group_size > 1:
            with tf.variable_scope("MinibatchStddev"):
                x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
        with tf.variable_scope("Conv"):
            x = apply_bias_act(conv2d_layer(x, dim = nf(1), kernel = 3), act = act)
        with tf.variable_scope("Dense0"):
            x = apply_bias_act(dense_layer(x, dim = nf(0)), act = act) 


    with tf.variable_scope("Output"):
        x = apply_bias_act(dense_layer(x, dim = max(labels_in.shape[1], 1)))
        if labels_in.shape[1] > 0:
            x = tf.reduce_sum(x * labels_in, axis = 1, keepdims = True) 

    scores_out = tf.identity(x, name = "scores_out")
    return scores_out # shape: [batch_size, 1]
