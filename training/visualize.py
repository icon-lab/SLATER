# save image grids
import shutil
import numpy as np
import PIL.Image

import dnnlib
import dnnlib.tflib as tflib
from tqdm import tqdm, trange
from training import misc

#************************************************************************************************************
def curr_batch_size(total_num, idx, batch_size):
    start = idx * batch_size
    end = min((idx + 1) * batch_size, total_num)
    return end - start

#************************************************************************************************************
def eval(
    G,
    dataset,                          # dataset object
    batch_size,                       # batch size for visualization
    training            = True,       # training mode
    latents             = None,       # source latents 
    labels              = None,       # source labels 
    ratio               = 1.0,        # height / width ratio of dataset
    drange_net          = [-1,1],     # generated image dynamic range
    vis_types           = None,       # type of visualizations
    num                 = 100,        # number of samples to be produced
    grid                = None,       # whether to save the samples in one large grid file
    grid_size           = None,       # grid size
    step                = None,       # step number
    keep_samples        = True,       # keep samples during training
    num_heads           = 1,          # num of attention heads
    components_num      = 16,         # num of latent components
    section_size        = 100):       # section size
    
    def prefix(step): return "" if step is None else "{:06d}_".format(step)
    def pattern_of(dir, step, suffix): return "eval/{}/{}%06d.{}".format(dir, prefix(step), suffix)

    vis = vis_types
    if training:
        vis = {"imgs"}
        section_size = num = len(latents)
    else:
        if vis is None:
            vis = {"imgs"}

    if grid is None:
        grid = training

    # build image functions
    save_images = misc.save_images_builder(drange_net, ratio, grid_size, grid, verbose=False)

    dirs = []
    if "imgs" in vis:            dirs += ["images"]

    if not keep_samples:
        shutil.rmtree("eval")
    for dir in dirs:
        misc.mkdir(dnnlib.make_run_dir_path("eval/{}".format(dir)))        

    for idx in range(0, num, section_size):
        curr_size = curr_batch_size(num, idx, section_size)
        # create random latents to generate images
        if latents is None:
            latents = np.random.randn(curr_size, *G.input_shape[1:])
        if labels is None:
            labels = dataset.get_minibatch_np(curr_size)[1]

        ret = G.run(latents, labels, randomize_noise = False, minibatch_size = batch_size, 
            return_dlatents = True) 
        images = ret[0]

    # save images
    if "imgs" in vis:
        save_images(images, pattern_of("images", step, "png"), idx)