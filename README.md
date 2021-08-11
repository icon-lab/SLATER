Official TensorFlow implementation of an unsupervised reconstruction model using zero-Shot Learned Adversarial TransformERs (SLATER). (https://arxiv.org/abs/2105.08059)

Korkmaz, Y., Dar, S. U., Yurt, M., Ozbey, M., & Cukur, T. (2021). Unsupervised MRI Reconstruction via Zero-Shot Learned Adversarial Transformers. arXiv preprint arXiv:2105.08059.

**************************************************************************************************************************************
# Demo
The following commands are used to train and test SLATER to reconstruct undersampled MR acquisitions from single- and multi-coil datasets. You can download pretrained network snaphots and sample datasets from the links given below. 

For training the MRI prior we use fully-sampled images, for testing undersampling is performed based on selected acceleration rate.
We have used AdamOptimizer in training, RMSPropOptimizer with momentum parameter 0.9 in testing/inference. In the current settings AdamOptimizer is used, you can change underlying optimizer class in dnnlib/tflib/optimizer.py file. You can insert additional paramaters like momentum to the line 87 in the optimizer.py file.

Sample training command for multi-coil (fastMRI) dataset:
python run_network.py --train --gpus=0 --expname=fastmri_t1_train --dataset=fastmri-t1 --data-dir=datasets/multi-coil-datasets/train

Sample reconstruction/test command for fastMRI dataset:
python run_recon_multi_coil.py reconstruct-complex-images --network=pretrained_snapshots/fastmri-t1/network-snapshot-001282.pkl --dataset=fastmri-t1 --acc-rate=4 --contrast=t1 --data-dir=datasets/multi-coil-datasets/test

Sample training command for single-coil (IXI) dataset:
python run_network.py --train --gpus=0 --expname=ixi_t1_train --dataset=ixi_t1 --data-dir=datasets/single-coil-datasets/train

Sample reconstruction/test command for IXI dataset:
python run_recon_single_coil.py reconstruct-magnitude-images --network=pretrained_snapshots/ixi-t1/network-snapshot-001282.pkl --dataset=ixi_t1_test --acc-rate=4 --contrast=t1 --data-dir=datasets/single-coil-datasets/test

**************************************************************************************************************************************
# Dataset Information and Data Preperation
IXI dataset: https://brain-development.org/ixi-dataset/ 
fastMRI Brain dataset: https://fastmri.med.nyu.edu/

For IXI dataset image dimensions are 256x256.
For fastMRI dataset image dimensions vary with contrasts. (T1: 256x320, T2: 288x384, FLAIR: 256x320).

SLATER requires datasets in the tfrecords format. To create tfrecords file containing new datasets you can use dataset_tool.py:

To create single-coil datasets you need to give magnitude images to dataset_tool.py with create_from_images function by just giving image directory containing images in .png format. We included undersampling masks under datasets/single-coil-datasets/test. 

To create multi-coil datasets you need to provide hdf5 files containing fully sampled coil-combined complex images in a variable named 'images_fs' with shape [num_of_images,x,y] (can be modified accordingly). To do this, you can use create_from_hdf5 function in dataset_tool.py. 

The MRI priors are trained on coil-combined datasets that are saved in tfrecords files with a 3-channel order of [real, imaginary, dummy]. For test purposes, we included sample coil-sensitivity maps (complex variable with 4-dimensions [x,y,num_of_image,num_of_coils] named 'coil_maps') and undersampling masks (3-dimensions [x,y, num_of_image] named 'map') in the datasets/multi-coil-datasets/test folder in hdf5 format. 

Coil-sensitivity-maps are estimated using ESPIRIT (http://people.eecs.berkeley.edu/~mlustig/Software.html). Network implementations use libraries from Gansformer (https://github.com/dorarad/gansformer) and Stylegan-2 (https://github.com/NVlabs/stylegan2) repositories.

**************************************************************************************************************************************
# Pretrained networks 
You can download pretrained network snapshots and datasets from these links. You need to place downloaded folders (datasets and pretrained_snapshots folders) under the main repo to run those sample test commands given above.

Pretrained network snapshots for IXI-T1 and fastMRI-T1 can be downloaded from Google Drive:
https://drive.google.com/drive/folders/1_69T1KUeSZCpKD3G37qgDyAilWynKhEc?usp=sharing

Sample training and test datasets for IXI-T1 and fastMRI-T1 can be downloaded from Google Drive:
https://drive.google.com/drive/folders/1hLC8Pv7EzAH03tpHquDUuP-lLBasQ23Z?usp=sharing

**************************************************************************************************************************************
## Notice for training with multi-coil datasets
To train multi-coil (complex) datasets you need to remove/add some lines in training_loop.py:

Comment out line 8.
Delete comment at line 9. 
Comment out line 23.

**************************************************************************************************************************************
# Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.

@article{korkmaz2021unsupervised,
  title={Unsupervised MRI Reconstruction via Zero-Shot Learned Adversarial Transformers},
  author={Korkmaz, Yilmaz and Dar, Salman UH and Yurt, Mahmut and {\"O}zbey, Muzaffer and {\c{C}}ukur, Tolga},
  journal={arXiv preprint arXiv:2105.08059},
  year={2021}

(c) ICON Lab 2021

**************************************************************************************************************************************
# Prerequisites
Python 3.6
NVIDIA GPU + CUDA CuDNN
TensorFlow 1.14 or 1.15

**************************************************************************************************************************************
# Acknowledgments

This code uses libraries from the StyleGAN-2 (https://github.com/NVlabs/stylegan2) and Gansformer (https://github.com/dorarad/gansformer) repositories.

For questions/comments please send me an email: korkmaz@ee.bilkent.edu.tr
}
**************************************************************************************************************************************
