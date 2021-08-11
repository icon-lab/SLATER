# Main function for multi-coil reconstruction. Designed for fastMRI dataset, can be modified accordingly.
import argparse
import numpy as np
import dnnlib
import re
import sys
import reconstruction_multi_coil
import pretrained_networks
from training import dataset_float
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

# 2d centered fft of multiple images
def fft2c_multi_np(im):
    array = []
    for i in range(im.shape[2]):
        image = im[:,:,i]
        array.append(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image))))
    return np.stack(array[:],axis=2)

# 2d centered ifft of multiple k-spaces
def ifft2c_multi_np(d):
    array = []
    for i in range(d.shape[2]):
        data = d[:,:,i]
        array.append(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(data))))
    return np.stack(array[:],axis=2)

#************************************************************************************************************
# reconstruct given complex image
def recon_image(recon, targets, png_prefix, num_snapshots, mask, coil_map, contrast):
    
    # configure snapshot steps
    snapshot_steps = set(recon.num_steps - np.linspace(0, recon.num_steps, num_snapshots, endpoint=False, dtype=int))
    
    # create coil-combined magnitude png images from complex tfrecords
    if contrast == 'T1' or contrast=='FLAIR':
        targets_255_real = targets[0,0,128:384,96:416] 
        targets_255_imag = targets[0,1,128:384,96:416]
    else:
        targets_255_real = targets[0,0,112:400,64:448] 
        targets_255_imag = targets[0,1,112:400,64:448]
    targets_255 = targets_255_real + 1j * targets_255_imag
    targets_abs = np.abs(targets_255)
    
    # save coil-combined magnitude target image
    misc.save_image_grid(targets_abs[np.newaxis][np.newaxis], png_prefix + 'target.png', drange=[np.min(targets_abs),np.max(targets_abs)])

    ssim = 0
    psnr = 0
    
    # start reconstruction
    recon.start(targets,mask, coil_map)
    while recon.get_cur_step() < recon.num_steps:
        print('\r%d / %d ... ' % (recon.get_cur_step(), recon.num_steps), end='', flush=True)
        recon.step()
        if recon.get_cur_step() in snapshot_steps:
            
            # get and save image in complex and float formats, save undersampling mask and magnitude png images
            reconstructed_image = recon.get_images()
            np.save(png_prefix +  'numpy_image' +  'step%04d' % recon.get_cur_step() + '.npy', reconstructed_image[0,0,:,:])
            untouched_images = recon.untouched_images()
            np.save(png_prefix + 'untouched_images.npy', untouched_images[0,:,:,:])
            misc.save_image_grid(reconstructed_image, png_prefix + 'step%04d.png' % recon.get_cur_step(), drange=[np.min(reconstructed_image), np.max(reconstructed_image)])
            np.save(png_prefix + 'mask.npy',recon.get_mask())
            target_images = targets_abs[np.newaxis][np.newaxis]
            
            # adjust range of magnitude images to [0, 1] before computing psnr and ssim (ssim may not be same as MATLAB due to implementation)
            target_images = misc.adjust_dynamic_range(target_images,[np.min(target_images),np.max(target_images)], [0,1])
            reconstructed_image = misc.adjust_dynamic_range(reconstructed_image,[np.min(reconstructed_image),np.max(reconstructed_image)], [0,1])
            
            # compute psnr and ssim
            psnr = compute_psnr(reconstructed_image[0,0,:,:],target_images[0,0,:,:])
            ssim = compute_ssim(reconstructed_image[0,0,:,:],target_images[0,0,:,:])
                
                
    print('\r%-30s\r' % '', end='', flush=True)
    return psnr, ssim




#************************************************************************************************************
# main function for reconstruction complex multi-coil images
def reconstruct_complex_images(network_pkl,         # pretrained or initial network pkl to be used in reconstruction
                               dataset_name,        # give the name of dataset (sub-folder in data_dir)
                               data_dir,            # datasets directory 
                               num_images,          # number of images to reconstruct
                               num_snapshots,       # number of snapshots to produce intermediate results
                               contrast,            # target contrast (to be used in padding and cuting coil maps and u.s. masks)
                               acc_rate):           # acceleration rate to be used 
   
    # filename for dataset directory including undersampling mask and coil maps 
    filename = "datasets/multi-coil-datasets/test/" + contrast.upper()  + "_under_sampled_" + acc_rate + "x_multicoil_test.mat"
    f = h5py.File(filename, 'r')
    us1_masks = f['map']
    coil1_maps = f['coil_maps']

    us_masks = us1_masks[:,:,:]
    coil_maps = coil1_maps[:,:,:,:]

    
    # transpose items to match with the MATLAB matrix ordering
    us_masks = np.transpose(us_masks)
    coil_maps = np.transpose(coil_maps)

    # convert coil maps to complex numpy arrays
    maps = coil_maps['real'] + 1j * coil_maps['imag']

    # load networks and initialize TensorFlow graph
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    recon = reconstruction_multi_coil.Reconstructor()
    recon.contrast = contrast.upper() 
    recon.set_network(Gs,_D)
    psnr_array = []
    ssim_array = []

    for i in range(1):
        dataset_obj = dataset_float.load_dataset(data_dir=data_dir, tfrecord_dir=dataset_name, max_label_size=0, repeat=False, shuffle_mb=0)

        for image_idx in range(us_masks.shape[2]):
            
            # choose mask and coil_map sequentially
            mask = us_masks[:,:,image_idx]
            coil_map = maps[:,:,image_idx,:]
            
            # image id used for reporting
            recon.image_idx = image_idx
            
            print('Reconstructing image %d/%d ...' % (image_idx, num_images))
            
            # load a single complex image from dataset object
            images, _labels = dataset_obj.get_minibatch_np(1)
            
            # reconstruct image
            psnr,ssim = recon_image(recon, targets=images, png_prefix=dnnlib.make_run_dir_path(contrast + "_" + acc_rate + "x" + '_image%04d-' % image_idx ), num_snapshots=num_snapshots,mask=mask, coil_map = coil_map, contrast=contrast.upper())
            psnr_array.append(psnr)
            ssim_array.append(ssim)

            
    prefix = "fastMRI_reconstruction_" + contrast + "_" + acc_rate + "x"

    np.save('metric_results/ssim_' + prefix + '.npy',ssim_array)
    np.save('metric_results/psnr_' + prefix + '.npy', psnr_array)

	


#************************************************************************************************************

def _parse_num_range(s):
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return range(int(m.group(1)), int(m.group(2))+1)
    vals = s.split(',')
    return [int(x) for x in vals]

def main():
    parser = argparse.ArgumentParser(
        description='''Multi-coil Reconstruction SLATER.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    reconstruct_complex_images_parser = subparsers.add_parser('reconstruct-complex-images')
    reconstruct_complex_images_parser.add_argument('--network', help='filename for network snapshot pkl', dest='network_pkl', required=True)
    reconstruct_complex_images_parser.add_argument('--data-dir', help='dataset root directory', required=True)
    reconstruct_complex_images_parser.add_argument('--dataset', help='dataset name', dest='dataset_name', required=True)
    reconstruct_complex_images_parser.add_argument('--num-snapshots', type=int, help='number of intermediate steps to produce results', default=5)
    reconstruct_complex_images_parser.add_argument('--num-images', type=int, help='number of images to be reconstructed (400 for fastMRI)', default=400)
    reconstruct_complex_images_parser.add_argument('--result-dir', help='directory to save results', default='results', metavar='DIR')
    reconstruct_complex_images_parser.add_argument('--contrast', dest='contrast', help="target contrast (t1, t2 or flair)")
    reconstruct_complex_images_parser.add_argument('--acc-rate', dest='acc_rate', help="acceleration rate (4 and 8 used in slater)")

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
    sc.run_name = 'multi_coil_reconstruction' + '_' +  str(args.dataset_name) + '_' + str(args.acc_rate) + 'x'
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = kwargs.pop('command') + '-' +  str(args.dataset_name) + '-' + str(args.acc_rate) + 'x'

    func_name_map = {
        'reconstruct-complex-images': 'run_recon_multi_coil.reconstruct_complex_images'
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#************************************************************************************************************

if __name__ == "__main__":
    main()

#************************************************************************************************************