# managing training process, evaluating network in certain periods, initialize networks and feed them with the dataset.
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

from training import dataset as data
#from training import dataset_float as data  # delete comment to use floating values in complex datasets

from training import misc
from training import visualize
import pretrained_networks
import glob

#************************************************************************************************************
# data process
# set shape of input images to RGB and apply mirror augment. If necessary, adjusting the range of images. (used in single-coil dataset, not for multi-coil) 
def process_reals(x, drange_data, drange_net, mirror_augment):
    with tf.name_scope("DynamicRange"):
        x = tf.cast(x, tf.float32)
        x.set_shape([None, 3, None, None])
        x = misc.adjust_dynamic_range(x, drange_data, drange_net) # comment out this line when using complex datasets
    if mirror_augment:
        with tf.name_scope("MirrorAugment"):
            x = tf.where(tf.random_uniform([tf.shape(x)[0]]) < 0.5, x, tf.reverse(x, [3]))
    return x

def read_data(data, name, shape, minibatch_gpu_in):
    var = tf.Variable(name = name, trainable = False,
        initial_value = tf.zeros(shape))
    data_write = tf.concat([data, var[minibatch_gpu_in:]], axis = 0)
    data_fetch_op = tf.assign(var, data_write)
    data_read = var[:minibatch_gpu_in]
    return data_read, data_fetch_op

#************************************************************************************************************
# setting up the training schedule
def training_schedule(
    sched_args,
    cur_nimg,                      # number of generated images (used to measure training length by converting to kimg)
    dataset,                       # dataset object
    lrate_rampup_kimg  = 0,        # learning rate rump up kimg
    tick_kimg          = 8):       # interval for producing snapshots

    # dictionary of schedule
    s = dnnlib.EasyDict()

    s.kimg = cur_nimg / 1000.0 # convert number of images to thousand of images for simplicity
    s.tick_kimg = tick_kimg
    s.resolution = 2 ** dataset.resolution_log2

    for arg in ["G_lrate", "D_lrate", "minibatch_size", "minibatch_gpu"]:
        s[arg] = sched_args[arg]

    # if rump-up is set
    if lrate_rampup_kimg > 0:
        rampup = min(s.kimg / lrate_rampup_kimg, 1.0)
        s.G_lrate *= rampup
        s.D_lrate *= rampup

    return s

# setting up the optimizer for training
def set_optimizer(cN, lrate_in, minibatch_multiplier, lazy_regularization = True, clip = None):
    args = dict(cN.opt_args)
    args["minibatch_multiplier"] = minibatch_multiplier
    args["learning_rate"] = lrate_in
    if lazy_regularization:
        mb_ratio = cN.reg_interval / (cN.reg_interval + 1)
        args["learning_rate"] *= mb_ratio
        if "beta1" in args: args["beta1"] **= mb_ratio
        if "beta2" in args: args["beta2"] **= mb_ratio
    cN.opt = tflib.Optimizer(name = "Loss{}".format(cN.name), clip = clip, **args)
    cN.reg_opt = tflib.Optimizer(name = "Reg{}".format(cN.name), share = cN.opt, clip = clip, **args)

def set_optimizer_ops(cN, lazy_regularization, no_op):
    cN.reg_norm = tf.constant(0.0)
    cN.trainables = cN.gpu.trainables

    if cN.reg is not None:
        if lazy_regularization:
            cN.reg_opt.register_gradients(tf.reduce_mean(cN.reg * cN.reg_interval), cN.trainables)
            cN.reg_norm = cN.reg_opt.norm
        else:
            cN.loss += cN.reg

    cN.opt.register_gradients(tf.reduce_mean(cN.loss), cN.trainables)
    cN.norm = cN.opt.norm

    cN.loss_op = tf.reduce_mean(cN.loss) if cN.loss is not None else no_op
    cN.regval_op = tf.reduce_mean(cN.reg) if cN.reg is not None else no_op
    cN.ops = {"loss": cN.loss_op, "reg": cN.regval_op, "norm": cN.norm}

def emaAvg(avg, value, alpha = 0.995):
    if value is None:
        return avg
    if avg is None:
        return value
    return avg * alpha + value * (1 - alpha)

# load the network from given snapshot
def load_nets(resume_pkl, lG, lD, lGs, recompile):
    misc.log("Loading networks from %s..." % resume_pkl, "white")
    rG, rD, rGs = pretrained_networks.load_networks(resume_pkl)
    
    if recompile:
        misc.log("Copying nets...")
        lG.copy_vars_from(rG); lD.copy_vars_from(rD); lGs.copy_vars_from(rGs)
    else:
        lG, lD, lGs = rG, rD, rGs
    return lG, lD, lGs

#************************************************************************************************************
# main loop used in training
def training_loop(
    cG = {}, cD = {},                   
    dataset_args            = {},       
    sched_args              = {},       
    vis_args                = {},       
    grid_args               = {},      
    tf_config               = {},      
    eval                    = False,    # evaluation mode
    train                   = False,    # training mode
    data_dir                = None,     # dataset directory
    total_kimg              = 25000,    # length of training (thousand of images - kimg)
    mirror_augment          = False,    # mirror augmentation
    drange_net              = [-1,1],   # dynamic range of images when feeding to the network
    ratio                   = 1.0,      # height / width ratio of images
    minibatch_repeats       = 4,        
    lazy_regularization     = True,     # use a seperate step for regularization
    smoothing_kimg          = 10.0,     # half-life of the moving average of generator weights
    clip                    = None,     # gradient clipping (not used in slater)
    resume_pkl              = None,     # optional network snapshot to resume training, if None start from scratch
    resume_kimg             = 0.0,      # optional resume kimg (affects schedule)
    resume_time             = 0.0,      # optional resume time (affects reporting)
    recompile               = False,    # true if recompile network from code, false when loading from snapshot
    summarize               = True,     # create summaries in Tensorboard
    save_tf_graph           = False,    # include full graph in saves
    save_weight_histograms  = False,    # include weight histograms in saves
    img_snapshot_ticks      = 10,       # saving frequency for image snapshots (measured in kimg)
    network_snapshot_ticks  = 10,       # saving frequency for network snapshots (measured in kimg)
    last_snapshots          = -1,       # -1 to save all snapshots, otherwise specify the number of latest snapshots to be saved  
    printname               = "",       # name of the experiment (overwritten in run_network.py)
    eval_images_num         = 5,
    # Architecture
    merge                   = False):   # merge generated images

    # initialize TensorFlow graph
    tflib.init_tf(tf_config)
    num_gpus = dnnlib.submit_config.num_gpus
    cG.name, cD.name = "g", "d"

    # loading dataset and configuring schedule
    dataset = data.load_dataset(data_dir = dnnlib.convert_path(data_dir), verbose = True, **dataset_args)
    sched = training_schedule(sched_args, cur_nimg = total_kimg * 1000, dataset = dataset)

    # construct networks
    with tf.device("/gpu:0"):
        no_op = tf.no_op()
        G, D, Gs = None, None, None
        if resume_pkl is None or recompile:
            misc.log("Constructing networks...", "white")
            print(dataset.label_size)
            G = tflib.Network("G", num_channels = dataset.shape[0], resolution = dataset.shape[1], 
                label_size = dataset.label_size, **cG.args)
            D = tflib.Network("D", num_channels = dataset.shape[0], resolution = dataset.shape[1], 
                label_size = dataset.label_size, **cD.args)
            Gs = G.clone("Gs")
        if resume_pkl is not None:
            G, D, Gs = load_nets(resume_pkl, G, D, Gs, recompile)

    G.print_layers()
    D.print_layers()

    # create real images grid
    grid_size, grid_reals, grid_labels = misc.setup_snapshot_img_grid(dataset, **grid_args)
    misc.save_image_grid(grid_reals, dnnlib.make_run_dir_path("reals.png"), drange = dataset.dynamic_range, grid_size = grid_size)
    grid_latents = np.random.randn(np.prod(grid_size), *G.input_shape[1:])

    # evaluation mode => save snapshot and visaulize.
    if eval:
        pkl = dnnlib.make_run_dir_path("network-eval-snapshot-%06d.pkl" % resume_kimg)
        misc.save_pkl((G, D, Gs), pkl, remove = False)

        visualize.eval(G, dataset, batch_size = sched.minibatch_gpu,
            drange_net = drange_net, ratio = ratio, **vis_args)

    # no need to use dataset in evaluation mode
    if not train:
        dataset.close()
        exit()

    misc.log("Building TensorFlow graph...", "white")
    with tf.name_scope("Inputs"), tf.device("/cpu:0"):
        lrate_in_g           = tf.placeholder(tf.float32, name = "lrate_in_g", shape = [])
        lrate_in_d           = tf.placeholder(tf.float32, name = "lrate_in_d", shape = [])
        step                 = tf.placeholder(tf.int32, name = "step", shape = [])
        minibatch_size_in    = tf.placeholder(tf.int32, name = "minibatch_size_in", shape=[])
        minibatch_gpu_in     = tf.placeholder(tf.int32, name = "minibatch_gpu_in", shape=[])
        minibatch_multiplier = minibatch_size_in // (minibatch_gpu_in * num_gpus)
        beta                 = 0.5 ** tf.div(tf.cast(minibatch_size_in, tf.float32), 
                                smoothing_kimg * 1000.0) if smoothing_kimg > 0.0 else 0.0

    # setting up the optimizer
    for cN, lr in [(cG, lrate_in_g), (cD, lrate_in_d)]:
        set_optimizer(cN, lr, minibatch_multiplier, lazy_regularization, clip)

    # building and dividing TensorFlow graph per gpu
    data_fetch_ops = []
    for gpu in range(num_gpus):
        with tf.name_scope("GPU%d" % gpu), tf.device("/gpu:%d" % gpu):

            for cN, N in [(cG, G), (cD, D)]:
                cN.gpu = N if gpu == 0 else N.clone(N.name + "_shadow")
            Gs_gpu = Gs if gpu == 0 else Gs.clone(Gs.name + "_shadow")

            with tf.name_scope("DataFetch"):
                reals, labels = dataset.get_minibatch_tf()
                reals = process_reals(reals, dataset.dynamic_range, drange_net, mirror_augment)
                reals, reals_fetch = read_data(reals, "reals",
                    [sched.minibatch_gpu] + dataset.shape, minibatch_gpu_in)
                labels, labels_fetch = read_data(labels, "labels",
                    [sched.minibatch_gpu, dataset.label_size], minibatch_gpu_in)
                data_fetch_ops += [reals_fetch, labels_fetch]

            with tf.name_scope("G_loss"):
                cG.loss, cG.reg = dnnlib.util.call_func_by_name(G = cG.gpu, D = cD.gpu, dataset = dataset,
                    reals = reals, minibatch_size = minibatch_gpu_in, **cG.loss_args)

            with tf.name_scope("D_loss"):
                cD.loss, cD.reg = dnnlib.util.call_func_by_name(G = cG.gpu, D = cD.gpu, dataset = dataset,
                    reals = reals, labels = labels, minibatch_size = minibatch_gpu_in, **cD.loss_args)

            for cN in [cG, cD]:
                set_optimizer_ops(cN, lazy_regularization, no_op)

    data_fetch_op = tf.group(*data_fetch_ops)
    for cN in [cG, cD]:
        cN.train_op = cN.opt.apply_updates()
        cN.reg_op = cN.reg_opt.apply_updates(allow_no_op = True)
    Gs_update_op = Gs.setup_as_moving_average_of(G, beta = beta)

    with tf.device("/gpu:0"):
        try:
            peak_gpu_mem_op = tf.contrib.memory_stats.MaxBytesInUse()
        except tf.errors.NotFoundError:
            peak_gpu_mem_op = tf.constant(0)
    tflib.init_uninitialized_vars()

    # create Tensorboard summaries
    if summarize:
        misc.log("Initializing logs...", "white")
        summary_log = tf.summary.FileWriter(dnnlib.make_run_dir_path())
        if save_tf_graph:
            summary_log.add_graph(tf.get_default_graph())
        if save_weight_histograms:
            G.setup_weight_histograms(); D.setup_weight_histograms()

    # start training
    misc.log("Training for %d kimg..." % total_kimg, "white")
    dnnlib.RunContext.get().update("", cur_epoch = resume_kimg, max_epoch = total_kimg)
    maintenance_time = dnnlib.RunContext.get().get_last_update_interval()

    cur_tick, running_mb_counter = -1, 0
    cur_nimg = int(resume_kimg * 1000)
    tick_start_nimg = cur_nimg
    for cN in [cG, cD]:
        cN.lossvals_agg = {k: None for k in ["loss", "reg", "norm", "reg_norm"]}
        cN.opt.reset_optimizer_state()

    # main training loop
    while cur_nimg < total_kimg * 1000:
        if dnnlib.RunContext.get().should_stop():
            break

        sched = training_schedule(sched_args, cur_nimg = cur_nimg, dataset = dataset)
        assert sched.minibatch_size % (sched.minibatch_gpu * num_gpus) == 0
        dataset.configure(sched.minibatch_gpu)

        feed_dict = {
            lrate_in_g: sched.G_lrate,
            lrate_in_d: sched.D_lrate,
            minibatch_size_in: sched.minibatch_size,
            minibatch_gpu_in: sched.minibatch_gpu,
            step: sched.kimg
        }

        for _repeat in range(minibatch_repeats):
            rounds = range(0, sched.minibatch_size, sched.minibatch_gpu * num_gpus)
            for cN in [cG, cD]:
                cN.run_reg = lazy_regularization and (running_mb_counter % cN.reg_interval == 0)
            cur_nimg += sched.minibatch_size
            running_mb_counter += 1

            for cN in [cG, cD]:
                cN.lossvals = {k: None for k in ["loss", "reg", "norm", "reg_norm"]}

            # perform gradient accumulation
            for _round in rounds:
                cG.lossvals.update(tflib.run([cG.train_op, cG.ops], feed_dict)[1])
                if cG.run_reg:
                    _, cG.lossvals["reg_norm"] = tflib.run([cG.reg_op, cG.reg_norm], feed_dict)

                tflib.run(data_fetch_op, feed_dict)

                cD.lossvals.update(tflib.run([cD.train_op, cD.ops], feed_dict)[1])
                if cD.run_reg:
                    _, cD.lossvals["reg_norm"] = tflib.run([cD.reg_op, cD.reg_norm], feed_dict)

            tflib.run([Gs_update_op], feed_dict)

            for cN in [cG, cD]:
                for k in cN.lossvals_agg:
                    cN.lossvals_agg[k] = emaAvg(cN.lossvals_agg[k], cN.lossvals[k])

        done = (cur_nimg >= total_kimg * 1000)
        if cur_tick < 0 or cur_nimg >= tick_start_nimg + sched.tick_kimg * 1000 or done:
            cur_tick += 1
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = dnnlib.RunContext.get().get_time_since_last_update()
            total_time = dnnlib.RunContext.get().get_time_since_start() + resume_time

            # progress reports
            print(("tick %s kimg %s   loss/reg: G (%s %s) D (%s %s)   grad norms: G (%s %s) D (%s %s)   " + 
                   "time %s sec/kimg %s maxGPU %sGB %s") % (
                misc.bold("%-5d" % autosummary("Progress/tick", cur_tick)),
                misc.bcolored("{:>8.1f}".format(autosummary("Progress/kimg", cur_nimg / 1000.0)), "red"),
                misc.bcolored("{:>6.3f}".format(cG.lossvals_agg["loss"] or 0), "blue"),
                misc.bold( "{:>6.3f}".format(cG.lossvals_agg["reg"] or 0)),
                misc.bcolored("{:>6.3f}".format(cD.lossvals_agg["loss"] or 0), "blue"),
                misc.bold("{:>6.3f}".format(cD.lossvals_agg["reg"] or 0)),
                misc.cond_bcolored(cG.lossvals_agg["norm"], 20.0, "red"),
                misc.cond_bcolored(cG.lossvals_agg["reg_norm"], 20.0, "red"),
                misc.cond_bcolored(cD.lossvals_agg["norm"], 20.0, "red"),
                misc.cond_bcolored(cD.lossvals_agg["reg_norm"], 20.0, "red"),
                misc.bold("%-10s" % dnnlib.util.format_time(autosummary("Timing/total_sec", total_time))),
                "{:>7.2f}".format(autosummary("Timing/sec_per_kimg", tick_time / tick_kimg)),
                "{:>4.1f}".format(autosummary("Resources/peak_gpu_mem_gb", peak_gpu_mem_op.eval() / 2**30)),
                printname))

            autosummary("Timing/total_hours", total_time / (60.0 * 60.0))
            autosummary("Timing/total_days", total_time / (24.0 * 60.0 * 60.0))

            # saving image and network snapshots
            if img_snapshot_ticks is not None and (cur_tick % img_snapshot_ticks == 0 or done):
                visualize.eval(G, dataset, batch_size = sched.minibatch_gpu, training = True,
                    step = cur_nimg // 1000, grid_size = grid_size, latents = grid_latents, 
                    labels = grid_labels, drange_net = drange_net, ratio = ratio, **vis_args)

            if network_snapshot_ticks is not None and (cur_tick % network_snapshot_ticks == 0 or done):
                pkl = dnnlib.make_run_dir_path("network-snapshot-%06d.pkl" % (cur_nimg // 1000))
                misc.save_pkl((G, D, Gs), pkl, remove = False)

                if last_snapshots > 0:
                    misc.rm(sorted(glob.glob(dnnlib.make_run_dir_path("network*.pkl")))[:-last_snapshots])

            if summarize:
                tflib.autosummary.save_summaries(summary_log, cur_nimg)

            dnnlib.RunContext.get().update(None, cur_epoch = cur_nimg // 1000, max_epoch = total_kimg)
            maintenance_time = dnnlib.RunContext.get().get_last_update_interval() - tick_time

    # save the latest snapshot
    misc.save_pkl((G, D, Gs), dnnlib.make_run_dir_path("network-final.pkl"), remove = False)


    if summarize:
        summary_log.close()
    dataset.close()
