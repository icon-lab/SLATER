# helper functions including dynamic scaling, printing etc. mainly built on Stylegan-2.
# Miscellaneous utility functions
import tensorflow as tf
import numpy as np
import PIL.Image
import pickle
import dnnlib
import math
import glob
import os

import seaborn as sns
from termcolor import colored
from tqdm import tqdm

#************************************************************************************************************
# printing operations

def bold(txt, **kwargs):
    return colored(str(txt),attrs = ["bold"])

def bcolored(txt, color):
    return colored(str(txt), color, attrs = ["bold"])

def cond_bcolored(num, maxval, color):
    num = num or 0
    txt = "{:>6.3f}".format(num)
    if maxval > 0 and num > maxval:
        return bcolored(txt, color)
    return txt

def error(txt):
    print(bcolored("Error: {}".format(txt), "red"))
    exit()

def log(txt, color = None):
    print(bcolored(txt, color) if color is not None else bold(txt))


#************************************************************************************************************
# file operations
    
def open_file_or_url(file_or_url):
    if dnnlib.util.is_url(file_or_url):
        return dnnlib.util.open_url(file_or_url, cache_dir = ".slater-cache")
    return open(file_or_url, "rb")

def load_pkl(file_or_url):
    with open_file_or_url(file_or_url) as file:
        return pickle.load(file, encoding = "latin1")

def save_pkl(obj, filename, remove = True):
    if remove:
        pattern = "{}-*.pkl".format("-".join(filename.split("-")[:-1]))
        toRemove = sorted(glob.glob(pattern))[:-5]

        for file in toRemove:
            os.remove(file)

    with open(filename, "wb") as file:
        pickle.dump(obj, file, protocol = pickle.HIGHEST_PROTOCOL)

def save_npy(mat, filename):
    with open(filename, 'wb') as f:
        np.save(f, mat)

def save_npys(npys, path, verbose = False, offset = 0):
    npys = enumerate(npys)
    if verbose:
        npys = tqdm(list(npys))
    for i, npy in npys:
        save_npy(npy, dnnlib.make_run_dir_path(path % (offset + i)))

def rm(files):
    for f in files:
        os.remove(f)

def mkdir(d):
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except:
            pass

#************************************************************************************************************
# image operations


# crop the center of a tensor
def crop_center(img, cw, ch):
    w, h = img.size
    return img.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2))

# crop the maximum sized center rectangle in a tensor
def crop_max_rectangle(img, ratio = 1.0):
    if ratio is None:
        return img
    s = min(img.size)
    return crop_center(img, s, ratio * s)

# pad to minimum square
def pad_min_square(img, pad_color = (0, 0, 0)):
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    result = PIL.Image.new(img.mode, (s, s), pad_color)
    offset_x = max(0, (h - w) // 2)
    offset_y = max(0, (w - h) // 2)
    result.paste(img, (offset_x, offset_y))
    return result

# adjust dynamic range of the image preserving floats
def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / \
            (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

# convert an image to a pillow image
def to_pil(img, drange = [0,1]):
    assert img.ndim == 2 or img.ndim == 3
    if img.ndim == 3:
        if img.shape[0] == 1:
            img = img[0] 
        else:
            img = img.transpose(1, 2, 0) 

    img = adjust_dynamic_range(img, drange, [0,255])
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    fmt = "L"
    if img.ndim == 3:
        fmt = {1: "L", 3: "RGB", 4: "RGBA"}.get(img.shape[-1], "L")
    img = PIL.Image.fromarray(img, fmt)

    return img

# apply mirror augment (not used in slater)
def apply_mirror_augment(minibatch):
    mask = np.random.rand(minibatch.shape[0]) < 0.5
    minibatch = np.array(minibatch)
    minibatch[mask] = minibatch[mask, :, :, ::-1]
    return minibatch


# create a pillow image grid to save multiple images in a single large grid
def create_image_grid(imgs, grid_size = None):
    assert imgs.ndim == 3 or imgs.ndim == 4
    num, img_w, img_h = imgs.shape[0], imgs.shape[-1], imgs.shape[-2]

    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros(list(imgs.shape[1:-2]) + [grid_h * img_h, grid_w * img_w], dtype = imgs.dtype)
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[..., y : y + img_h, x : x + img_w] = imgs[idx]
    return grid

def save_image_grid(imgs, filename, drange = [0,1], grid_size = None):
    to_pil(create_image_grid(imgs, grid_size), drange).save(filename)

# save image grid in a snapshot during training
def setup_snapshot_img_grid(dataset, size = "1080p", layout = "random"):
    gw = 1; gh = 1
    if size == "1080p":
        gw = np.clip(1920 // 256, 3, 32)
        gh = np.clip(1080 // 256, 2, 32)
    elif size == "4k":
        gw = np.clip(3840 // dataset.shape[2], 7, 32)
        gh = np.clip(2160 // dataset.shape[1], 4, 32)
    elif size == "8k":
        gw = np.clip(7680 // dataset.shape[2], 7, 32)
        gh = np.clip(4320 // dataset.shape[1], 4, 32)
    else: 
        gw = size

    reals = np.zeros([gw * gh] + dataset.shape, dtype = dataset.dtype)
    labels = np.zeros([gw * gh, dataset.label_size], dtype = dataset.label_dtype)

    if layout == "random":
        reals[:], labels[:] = dataset.get_minibatch_np(gw * gh)

    class_layouts = dict(row_per_class = [gw, 1], col_per_class = [1, gh], class4x4 = [4, 4])
    if layout in class_layouts:
        bw, bh = class_layouts[layout]
        nw = (gw - 1) // bw + 1
        nh = (gh - 1) // bh + 1
        blocks = [[] for _i in range(nw * nh)]
        for _iter in range(1000000):
            (real, seg), label = dataset.get_minibatch_np(1)
            idx = np.argmax(label[0])
            while idx < len(blocks) and len(blocks[idx]) >= bw * bh:
                idx += dataset.label_size
            if idx < len(blocks):
                blocks[idx].append((real, seg, label))
                if all(len(block) >= bw * bh for block in blocks):
                    break
        for i, block in enumerate(blocks):
            for j, (real, seg, label) in enumerate(block):
                x = (i %  nw) * bw + j %  bw
                y = (i // nw) * bh + j // bw
                if x < gw and y < gh:
                    reals[x + y * gw] = real[0]
                    labels[x + y * gw] = label[0]

    return (gw, gh), reals, labels

#************************************************************************************************************
# visualization operations
    
def get_colors(num):
    colors = sns.color_palette("hls", num)
    colors = [[(2 * p - 1) for p in c] for c in colors]
    return colors


def clean_filename(filename):
    return filename.replace("_00000", "").replace("00000_", "")

def save_images_builder(drange, ratio, grid_size, grid = False, verbose = False):
    def save_images(imgs, path, offset = 0):
        if grid:
            save_image_grid(imgs, dnnlib.make_run_dir_path(clean_filename(path % offset)), drange, grid_size)
        else:
            imgs = enumerate(imgs)
            if verbose:
                imgs = tqdm(list(imgs))
            for i, img in imgs:
                img = to_pil(img, drange = drange)
                img = crop_max_rectangle(img, ratio)
                img.save(dnnlib.make_run_dir_path(path % (offset + i)))
    return save_images

def save_blends_builder(drange, ratio, grid_size, grid = False, verbose = False, alpha = 0.3):
    def save_blends(imgs_a, imgs_b, path, offset = 0):
        if grid:
            img_a = to_pil(create_image_grid(imgs_a, grid_size), drange)
            img_b = to_pil(create_image_grid(imgs_b, grid_size), drange)
            blend = PIL.Image.blend(img_a, img_b, alpha = alpha)
            blend.save(dnnlib.make_run_dir_path(clean_filename(path % offset)))
        else:
            img_pairs = zip(imgs_a, imgs_b)
            img_pairs = enumerate(img_pairs)
            if verbose:
                img_pairs = tqdm(list(img_pairs))            
            for i, (img_a, img_b) in img_pairs:
                img_a = to_pil(img_a, drange = drange)
                img_b = to_pil(img_b, drange = drange)
                blend = PIL.Image.blend(img_a, img_b, alpha = alpha)
                blend = crop_max_rectangle(blend, ratio)
                blend.save(dnnlib.make_run_dir_path(path % (offset + i)))
    return save_blends
