from warnings import simplefilter # filter the warnings
simplefilter(action = "ignore", category = FutureWarning)
import argparse
import copy
import glob
import sys
import os
import dnnlib
from dnnlib import EasyDict
from training import misc
import pretrained_networks
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#************************************************************************************************************
# conditional set functions
def cset(dicts, name, prop):
    if not isinstance(dicts, list):
        dicts = [dicts]
    if prop is not None:
        for d in dicts:
            d[name] = prop

def nset(args, name, prop):
    if name not in sys.argv:
        args[name] = prop

def dset(d, name, prop, default):
    if d[name] == default:
        d[name] = prop
        
#************************************************************************************************************
# settings for network
def set_net(net, reg_interval):
    ret = EasyDict()
    ret.args  = EasyDict(func_name = "training.network.{}_slater".format(net[0])) 
    ret.loss_args = EasyDict(func_name = "training.loss.{}_loss".format(net[0]))      
    ret.opt_args  = EasyDict(beta1 = 0.0, beta2 = 0.99, epsilon = 1e-8)               
    ret.reg_interval = reg_interval
    return ret

#************************************************************************************************************
def run(**args): 
    args      = EasyDict(args)
    train     = EasyDict(run_func_name = "training.training_loop.training_loop") 
    sched     = EasyDict()                                                      
    vis       = EasyDict()                                                      
    grid      = EasyDict(size = "1080p", layout = "random")                     
    sc        = dnnlib.SubmitConfig()                                            

    # convert store_true elements to True
    for arg in ["summarize", "keep_samples", "style", "fused_modconv", "local_noise"]:
        if args[arg] is None:
            args[arg] = True

    if not args.train and not args.eval:
        misc.log("specify train or evaluation mode by using --train or --eval", "red")

    task = args.dataset
    pretrained = "gdrive:{}-snapshot.pkl".format(task)
    if pretrained not in pretrained_networks.gdrive_urls:
        pretrained = None

    nset(args, "recompile", pretrained is not None)
    nset(args, "pretrained_pkl", pretrained)
        
    # tensorflow / gpu settings
    tf_config = {
        "rnd.np_random_seed": 1000,
        "allow_soft_placement": True,
        "gpu_options.per_process_gpu_memory_fraction": 1.0
    }
    if args.gpus != "":
        num_gpus = len(args.gpus.split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    assert num_gpus in [1, 2, 4, 8]
    sc.num_gpus = num_gpus

    cG = set_net("G", reg_interval = 4)
    cD = set_net("D", reg_interval = 16)

    
    # dataset settings
    args.ratio = args["ratio"]
    dataset_args = EasyDict(tfrecord_dir = args.dataset, max_imgs = args.train_images_num, 
        num_threads = args.num_threads)
    for arg in ["data_dir", "mirror_augment", "total_kimg", "ratio"]:
        cset(train, arg, args[arg])

    # general settings
    for arg in ["eval", "train", "recompile", "last_snapshots"]:
        cset(train, arg, args[arg])

    # rounding minibatch size to an integer
    args.batch_size -= args.batch_size % args.minibatch_size
    args.minibatch_std_size -= args.minibatch_std_size % args.minibatch_size
    args.latent_size -= args.latent_size % args.components_num
    if args.latent_size == 0:
        misc.error("--latent-size is too small. Must best a multiply of components-num")

    sched_args = {
        "G_lrate": "g_lr",
        "D_lrate": "d_lr",
        "minibatch_size": "batch_size",
        "minibatch_gpu": "minibatch_size"
    }
    for arg, cmd_arg in sched_args.items():
        cset(sched, arg, args[cmd_arg])


    cset(cG.args, "truncation_psi", args.truncation_psi)
    for arg in ["keep_samples", "num_heads"]:
        cset(vis, arg, args[arg])
    for arg in ["summarize", "eval_images_num"]:
        cset(train, arg, args[arg])

    # visualization settings
    args.vis_imgs = args.vis_images
    vis_types = ["imgs"]
    vis.vis_types = {arg for arg in vis_types if args["vis_{}".format(arg)]}

    vis_args = {
        "grid": "vis_grid",
    }
    for arg, cmd_arg in vis_args.items():
        cset(vis, arg, args[cmd_arg])

    cset(cG.args, "architecture", args.g_arch)
    cset(cD.args, "architecture", args.d_arch)
         
    if args.components_num > 1:
        args.latent_size = int(args.latent_size / args.components_num)
    cD.args.latent_size = cG.args.latent_size = cG.args.dlatent_size = args.latent_size
    cset([cG.args, cD.args, vis], "components_num", args.components_num)

    # mapper settings
    for arg in ["layersnum", "lrmul", "dim", "resnet"]:
        field = "mapping_{}".format(arg)
        cset(cG.args, field, args[field])

    # style-based generator model settings
    for arg in ["style", "fused_modconv", "local_noise"]:
        cset(cG.args, arg, args[arg])
    cD.args.mbstd_group_size = args.minibatch_std_size

    cset(cG.args, "transformer", args.transformer)

    args.norm = args.normalize
    for arg in ["norm", "integration", "kmeans", "kmeans_iters", "mapping_ltnt2ltnt"]:
        cset(cG.args, arg, args[arg])

    for arg in ["use_pos", "num_heads"]:
        cset([cG.args, cD.args], arg, args[arg])

    # encoding settings
    for arg in ["dim", "init", "directions_num"]:
        field = "pos_{}".format(arg)
        cset([cG.args, cD.args], field, args[field])

    # attention settings
    for arg in ["start_res", "end_res"]: 
        cset(cG.args, arg, args["g_{}".format(arg)])

    for arg in ["style_mixing", "attention_dropout"]:
        cset(cG.args, arg, args[arg])

    # loss settings
    gloss_args = {
        "loss_type": "g_loss",
        "reg_weight": "g_reg_weight",
    }
    dloss_args = {
        "loss_type": "d_loss",
        "reg_type": "d_reg",
        "gamma": "gamma"
    }
    for arg, cmd_arg in gloss_args.items():
        cset(cG.loss_args, arg, args[cmd_arg])
    for arg, cmd_arg in dloss_args.items():
        cset(cD.loss_args, arg, args[cmd_arg])

    # find latest directory used for the same experiment
    exp_dir = sorted(glob.glob("{}/{}-*".format(args.result_dir, args.expname)))
    run_id = 0
    if len(exp_dir) > 0:
        run_id = int(exp_dir[-1].split("-")[-1])
    if args.restart:
        run_id += 1

    run_name = "{}-{:03d}".format(args.expname, run_id)
    train.printname = "{} ".format(misc.bold(args.expname))

    snapshot, kimg, resume = None, 0, False
    pkls = sorted(glob.glob("{}/{}/network*.pkl".format(args.result_dir, run_name)))
    if args.pretrained_pkl is not None and args.pretrained_pkl != "None":
        if args.pretrained_pkl.startswith("gdrive"):
            if args.pretrained_pkl not in pretrained_networks.gdrive_urls:
                misc.error("--pretrained_pkl {} not available in the catalog (see pretrained_networks.py)")

            snapshot = args.pretrained_pkl
        else: 
            snapshot = glob.glob(args.pretrained_pkl)[0]
            if os.path.islink(snapshot):
                snapshot = os.readlink(snapshot)

        try:
            kimg = int(snapshot.split("-")[-1].split(".")[0])
        except:
            pass

    elif len(pkls) > 0:
        snapshot = pkls[-1]
        kimg = int(snapshot.split("-")[-1].split(".")[0])
        resume = True

    if snapshot:
        misc.log("Resuming {}, from {}, kimg {}".format(run_name, snapshot, kimg), "white")
        train.resume_pkl = snapshot
        train.resume_kimg = kimg
    else:
        misc.log("Start model training from scratch", "white")
    
    # submit settings
    sc.run_dir_root = args.result_dir
    sc.run_desc = args.expname
    sc.run_id = run_id
    sc.run_name = run_name
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True

    kwargs = EasyDict(train)
    kwargs.update(cG = cG, cD = cD)
    kwargs.update(dataset_args = dataset_args, vis_args = vis, sched_args = sched, 
        grid_args = grid, tf_config = tf_config)
    kwargs.submit_config = copy.deepcopy(sc)
    kwargs.resume = resume
    kwargs.load_config = args.reload

    dnnlib.submit_run(**kwargs)

#************************************************************************************************************
# helper functions
def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1", ""):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Error: Boolean value expected")

def _parse_comma_sep(s):
    if s is None or s.lower() == "none" or s == "":
        return []
    return s.split(",")


#************************************************************************************************************
def main():
    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawDescriptionHelpFormatter
    )

    # general settings
    parser.add_argument("--expname",            help = "name of the experiment", default = "slater_exp", type = str)
    parser.add_argument("--eval",               help = "evaulation mode", default = None, action = "store_true")
    parser.add_argument("--train",              help = "train mode", default = None, action = "store_true")
    parser.add_argument("--gpus",               help = "gpus will be used in the experiment seperated with commas", default = "0", type = str)
    parser.add_argument("--pretrained-pkl",     help = "to resume from a snapshot give its filename", default = None, type = str)
    parser.add_argument("--restart",            help = "restart training from scratch", default = False, action = "store_true")
    parser.add_argument("--reload",             help = "reload training options from the original configuration file if true", default = False, action = "store_true")
    parser.add_argument("--recompile",          help = "recompile model from source code when resuming training", default = None, action = "store_true")
    parser.add_argument("--last-snapshots",     help = "number of latest snapshots saved (default: all)", default = -1, type = int)

    # dataset settings
    parser.add_argument("--data-dir",           help = "root directory for datasets (default: datasets)", default = "datasets", metavar = "DIR")
    parser.add_argument("--dataset",            help = "name of the dataset will be used in training (sub-folder name in datasets folder)", required = True)
    parser.add_argument("--ratio",              help = "height/width ratio of images in the dataset", default = 1.0, type = float)
    parser.add_argument("--num-threads",        help = "number of processing threads (default: 4)", default = 4, type = int)
    parser.add_argument("--mirror-augment",     help = "apply mirror augment to the data (default: false)", default = False, action = "store_true")
    parser.add_argument("--train-images-num",   help = "number of images to be used in training none if use all", default = None, type = int)

    # training settings
    parser.add_argument("--batch-size",         help = "batch size to be used in the optimizer", default = 32, type = int)
    parser.add_argument("--minibatch-size",     help = "batch size per gpu", default = 4, type = int)
    parser.add_argument("--total-kimg",         help = "training duration in terms of number of thousand images (can be converted to epoch number by dividing kimg to dataset size)", metavar = "KIMG", default = 25000, type = int)
    parser.add_argument("--gamma",              help = "r1 regularization hyperparameter (default: 10)", default = 10, type = float)
    parser.add_argument("--g-lr",               help = "learning rate for generator", default = 0.002, type = float)
    parser.add_argument("--d-lr",               help = "learning rate for discriminator", default = 0.002, type = float)

    # evaluation settings
    parser.add_argument("--result-dir",         help = "directory to be used in saving results", default = "results", metavar = "DIR")
    parser.add_argument("--summarize",          help = "creating tensorboard summaries", default = True, metavar = "BOOL", type = _str_to_bool, nargs = "?")
    parser.add_argument("--truncation-psi",     help = "truncation psi to be used in producing sample images (just for visualization purposes see Stylegan for truncation details)", default = 0.65, type = float)
    parser.add_argument("--keep-samples",       help = "keep all image samples during training", default = True, metavar = "BOOL", type = _str_to_bool, nargs = "?")
    parser.add_argument("--eval-images-num",    help = "num of images to evaluate", default = 1000, type = int)

    # visualization settings
    parser.add_argument("--vis-images",         help = "save image samples", default = None, action = "store_true")
    parser.add_argument("--vis-grid",           help = "save images in a large grid", default = True, action = "store_true")

    # generator and discriminator archirecture settings
    parser.add_argument("--g-arch",             help = "architecture used in generator (resnet in slater)", default = "resnet", choices = ["orig", "skip", "resnet"], type = str)
    parser.add_argument("--d-arch",             help = "archirecture used in discriminator (resnet in slater)", default = "resnet", choices = ["orig", "skip", "resnet"], type = str)

    # mapper settings
    parser.add_argument("--mapping-layersnum",  help = "number of layers in mapping network", default = 8, type = int)
    parser.add_argument("--mapping-lrmul",      help = "mapping network learning rate multiplier", default = 0.01, type = float)
    parser.add_argument("--mapping-dim",        help = "mapping layer dimension = latent_size as default", default = None, type = int)
    parser.add_argument("--mapping-resnet",     help = "include resnet connections in the mapping as well (True in slater)", default = True, action = "store_true")

    # loss settings
    parser.add_argument("--g-loss",             help = "loss function used in the generator (nan-saturating logistic in slater)", default = "logistic_ns", choices = ["logistic", "logistic_ns", "hinge", "wgan"], type = str)
    parser.add_argument("--g-reg-weight",       help = "regularization hyperparameter in generator (1.0 in slater)", default = 1.0, type = float)
    parser.add_argument("--d-loss",             help = "loss function used in the discriminator (logistic with r1 regularization in slater)", default = "logistic", choices = ["wgan", "logistic", "hinge"], type = str)
    parser.add_argument("--d-reg",              help = "regularization type used in discriminator (r1 in slater)", default = "r1", choices = ["non", "gp", "r1", "r2"], type = str)

    # style-generative model settings
    parser.add_argument("--style",              help = "use global latent for high-level style-modulation (used in slater)", default = True, metavar = "BOOL", type = _str_to_bool, nargs = "?")
    parser.add_argument("--fused-modconv",      help = "using fused modulation and convolution (used in slater)", default = True, metavar = "BOOL", type = _str_to_bool, nargs = "?")
    parser.add_argument("--local-noise",        help = "noise adding for stochastic details (used in slater)", default = True, metavar = "BOOL", type = _str_to_bool, nargs = "?")
    parser.add_argument("--minibatch-std-size", help = "minibatch standard deviation layer size in  discriminator (4 in slater, see Stylegan for details)", default = 4, type = int)
    parser.add_argument("--style-mixing",       help = "style-mixing probability 0.9 by default like in Stylegan", default = 0.9, type = float)
    parser.add_argument("--attention-dropout",  help = "dropout rate for attention (0.12 in slater)", default = 0.12, type = float)

    # cross & self-attention block settings
    parser.add_argument("--transformer",        help = "include cross-attention-transformer blocks (used in slater)", default = True, action = "store_true")
    parser.add_argument("--latent-size",        help = "sum of local latent sizes (16 local latent vectors in slater each has dimension 32)", default = 512, type = int)
    parser.add_argument("--components-num",     help = "number of local latent vectors (16 in slater)", default = 16, type = int)
    parser.add_argument("--num-heads",          help = "number of heads used in attention blocks (1 in slater)", default = 1, type = int)
    parser.add_argument("--normalize",          help = "normalization type for features (layer-wise in slater)", default = "layer", choices = ["batch", "instance", "layer"], type = str)
    parser.add_argument("--integration",        help = "integration type of attention (multiplication in slater)", default = "mul", choices = ["add", "mul", "both"], type = str)
    parser.add_argument("--g-start-res",        help = "first layer where cross-attention-transformer block will be included (4x4 for slater)", default = 0, type = int)
    parser.add_argument("--g-end-res",          help = "last layer where cross-attention-transformer block will be included (128x128 for slater)", default = 8, type = int)
    parser.add_argument("--kmeans",             help = "use centroids when using attention-blocks (used in slater, see k-means algorithm for details Lloyd et al. 1982)", default = True, action = "store_true")
    parser.add_argument("--kmeans-iters",       help = "number of iterations used in k-means algorithm ", default = 1, type = int) # -per-layer
    parser.add_argument("--mapping-ltnt2ltnt",  help = "use self-attention-blocks in the mapper network", default = True, action = "store_true")
    
    # encoding for attention settings
    parser.add_argument("--use-pos",            help = "positional encoding for images (used in slater)", default = True, action = "store_true")
    parser.add_argument("--pos-dim",            help = "dimension for positional encodings (equal to latent_size in slater)", default = None, type = int)
    parser.add_argument("--pos-type",           help = "type of encoding used in images (sinusoidal in slater)", default = "sinus", choices = ["linear", "sinus", "trainable", "trainable2d"], type = str)
    parser.add_argument("--pos-init",           help = "type of initialization used in encoding (uniform in slater)", default = "uniform", choices = ["uniform", "normal"], type = str)
    parser.add_argument("--pos-directions-num", help = "dimension for encoding directions (x and y in slater, 2d)", default = 2, type = int)

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        misc.error("Dataset root directory does not exist")

    if not os.path.exists("{}/{}".format(args.data_dir, args.dataset)):
        misc.error("The dataset {}/{} directory does not exist".format(args.data_dir, args.dataset))


    run(**vars(args))
    
#************************************************************************************************************
if __name__ == "__main__":
    main()
