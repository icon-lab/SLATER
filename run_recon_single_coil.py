# Main function for single-coil reconstruction. Designed for IXI dataset, can be modified accordingly.
import argparse
import numpy as np
import dnnlib
import re
import sys
import reconstruction_single_coil
import pretrained_networks
from training import dataset
from training import misc
import h5py
import os
from psnr import compute_psnr, compute_ssim

#************************************************************************************************************
# Fourier Operations

# 2d centered fft
def fft2c(im):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(im))) 

# 2d centered ifft
def ifft2c(d):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(d)))

#************************************************************************************************************
# reconstruct given  single-coil magnitude image
    
def recon_image(recon, targets, png_prefix, num_snapshots, mask):
    # configure snapshot steps
    snapshot_steps = set(recon.num_steps - np.linspace(0, recon.num_steps, num_snapshots, endpoint=False, dtype=int))
    # save target image
    misc.save_image_grid(targets.copy(), png_prefix + 'target.png', drange=[np.min(targets),np.max(targets)])
    targets_ = targets.copy()
    ssim = 0
    psnr = 0
    # start reconstruction
    recon.start(targets, mask)
    while recon.get_cur_step() < recon.num_steps:
        print('\r%d / %d ... ' % (recon.get_cur_step(), recon.num_steps), end='', flush=True)
        recon.step()
        if recon.get_cur_step() in snapshot_steps:
            # collect results and save them in the desired format
            reconstructed_image = recon.get_images()
            np.save(png_prefix +  'numpy_image.npy', reconstructed_image[0,0,:,:])
            untouched_images = recon.untouched_images()
            np.save(png_prefix + 'untouched_images.npy', untouched_images)
            misc.save_image_grid(reconstructed_image, png_prefix + 'step%04d.png' % recon.get_cur_step(), drange=[np.min(reconstructed_image), np.max(reconstructed_image)])
            np.save(png_prefix + 'mask.npy',recon.get_mask())
            target_images = targets_
            # adjust range of magnitude images to [0, 1] before computing psnr and ssim (ssim may not be same as MATLAB due to implementation)
            target_images = misc.adjust_dynamic_range(target_images,[np.min(target_images),np.max(target_images)], [0,1])
            psnr = compute_psnr(reconstructed_image[0,0,:,:],target_images[0,0,:,:])
            ssim = compute_ssim(reconstructed_image[0,0,:,:],target_images[0,0,:,:])
                
                
    print('\r%-30s\r' % '', end='', flush=True)
    
    return psnr, ssim

#----------------------------------------------------------------------------



#----------------------------------------------------------------------------

def reconstruct_magnitude_images(network_pkl,         # pretrained or initial network pkl to be used in reconstruction
                               dataset_name,        # give the name of dataset (sub-folder in data_dir)
                               data_dir,            # datasets directory 
                               num_images,          # number of images to reconstruct
                               num_snapshots,       # number of snapshots to produce intermediate results
                               contrast,            # target contrast (to be used in padding and cuting coil maps and u.s. masks)
                               acc_rate):           # acceleration rate to be used 

    f = h5py.File("datasets/single-coil-datasets/test/" + contrast.upper() + "_" + str(acc_rate) + "_multi_synth_recon_test.mat", 'r')
    us_masks = f['us_masks']


    # transpose masks to match with the MATLAB matrix ordering
    us_masks = np.transpose(us_masks)
    
    # load networks and initialize TensorFlow graph
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    recon = reconstruction_single_coil.Reconstructor()
    recon.set_network(Gs)
    psnr_array= []
    ssim_array = []

    for i in range(1):
        dataset_obj = dataset.load_dataset(data_dir=data_dir, tfrecord_dir=dataset_name, max_label_size=0, repeat=False, shuffle_mb=0)

        for image_idx in range(num_images):
            print('Reconstructing image %d/%d ...' % (image_idx, num_images))
            mask = us_masks[:,:,image_idx]
            recon.image_idx = image_idx
            images, _labels = dataset_obj.get_minibatch_np(1)
            images = misc.adjust_dynamic_range(images, [np.min(images), np.max(images)], [-1, 1])
            
            psnr,ssim = recon_image(recon, targets=images, png_prefix=dnnlib.make_run_dir_path(str(i) + '_image%04d-' % image_idx ), num_snapshots=num_snapshots,mask=mask)
    
    
    prefix = "IXI_reconstruction_" + contrast + "_" + acc_rate + "x"

    np.save('metric_results/ssim_' + prefix + '.npy',ssim_array)
    np.save('metric_results/psnr_' + prefix + '.npy', psnr_array)

#----------------------------------------------------------------------------

def _parse_num_range(s):
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return range(int(m.group(1)), int(m.group(2))+1)
    vals = s.split(',')
    return [int(x) for x in vals]


def main():
    parser = argparse.ArgumentParser(
        description='''Single-coil Reconstruction SLATER.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    reconstruct_magnitude_images_parser = subparsers.add_parser('reconstruct-magnitude-images', help='reconstruct-magnitude-images')
    reconstruct_magnitude_images_parser.add_argument('--network', help='filename for network snapshot pkl', dest='network_pkl', required=True)
    reconstruct_magnitude_images_parser.add_argument('--data-dir', help='dataset root directory', required=True)
    reconstruct_magnitude_images_parser.add_argument('--dataset', help='dataset name', dest='dataset_name', required=True)
    reconstruct_magnitude_images_parser.add_argument('--num-snapshots', type=int, help='number of intermediate steps to produce results', default=5)
    reconstruct_magnitude_images_parser.add_argument('--num-images', type=int, help='number of images to be reconstructed', default=1080)
    reconstruct_magnitude_images_parser.add_argument('--result-dir', help='directory to save results', default='results', metavar='DIR')
    reconstruct_magnitude_images_parser.add_argument('--acc-rate', dest='acc_rate', help="acceleration rate (4 and 8 used in slater)")
    reconstruct_magnitude_images_parser.add_argument('--contrast', dest='contrast', help="target contrast (t1 or t2 for IXI)")


    args = parser.parse_args()
    subcmd = args.command
    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    kwargs = vars(args)
    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 2
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_name =  'single-coil_reconstruction' + '_' +  str(args.dataset_name) + '_' + str(args.acc_rate) + 'x'
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = kwargs.pop('command') + '-' +  str(args.dataset_name) + '-' + str(args.acc_rate) + 'x'

    func_name_map = {
            'reconstruct-magnitude-images': 'run_recon_single_coil.reconstruct_magnitude_images'
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
