# Multi-coil reconstruction class 
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib


#************************************************************************************************************
class Reconstructor:
    def __init__(self):
        
        # inference settings
        self.num_steps                  = 1000         # number of optimization steps
        self.dlatent_avg_samples        = 10000        # number of latent samples to be used to calculate average latent 
        self.initial_learning_rate      = 0.1          # initial learning rate to reach with ramp up
        self.lr_rampdown_length         = 0.25         # learning rate ramp down length ( 0.25 * 1000 = last 250 steps)
        self.lr_rampup_length           = 0.05         # learning rate ramp up length (0.1 * 1000 = first 100 steps)
        
        # main settings
        self.verbose                    = False        # enable prints & reports for user
        self.clone_net                  = True         # clone network (beneficial for weight optimization)
        self.image_idx                  = None         # current reconstruction image id
        
        # variables to be used
        self._D                         = None
        self._Gs                        = None
        self._minibatch_size            = None
        self._dlatent_avg               = None
        self._dlatent_std               = None
        self._noise_vars                = None
        self._noise_init_op             = None
        self._noise_normalize_op        = None
        self._dlatents_var              = None
        self._noise_in                  = None
        self._dlatents_expr             = None
        self._images_expr               = None
        self._target_images_var         = None
        self._lpips                     = None
        self._dist                      = None
        self._loss                      = None
        self._reg_sizes                 = None
        self._lrate_in                  = None
        self._opt                       = None
        self._opt_step                  = None
        self._cur_step                  = None
        self.contrast                   = None
        self.pad_x                      = None
        self.pad_y                      = None
        self.initial_weights            = None
        self.attention_maps             = None

    def _info(self, *args):
        if self.verbose:
            print('Reconstructor:', *args)

#************************************************************************************************************
# fourier operations defined for numpy arrays
    
    # multi-array centered fft 
    def fft2c_multi_np(self,im):
        array = []
        for i in range(im.shape[2]):
            image = im[:,:,i]
            array.append(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image))))
        return np.stack(array[:],axis=2)

    # multi-array centered ifft 
    def ifft2c_multi_np(self,d):
        array = []
        for i in range(d.shape[2]):
            data = d[:,:,i]
            array.append(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(data))))
        return np.stack(array[:],axis=2)

    # single-array centered fft
    def fft2c_np(self,im):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(im))) 
    
    # single-array centered ifft
    def ifft2c_np(self,d):
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(d)))

#************************************************************************************************************
# fourier operations defined for TensorFlow tensors
        
    # multi-array centered fft 
    def fft2c_multi(self,im):
        array = []
        for i in range(im.shape[2]):
            image = im[:,:,i]
            array.append(tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(image))))
        print(array)
        return tf.stack(array[:],axis=2)

    # multi-array centered ifft
    def ifft2c_multi(self,d):
        array = []
        for i in range(d.shape[2]):
            data = d[:,:,i]
            array.append(tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(data))))
        return tf.stack(array[:],axis=2)

    # single-array centered fft
    def fft2c(self, im):
        return tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(im))) 
    
    # single-array centered ifft
    def ifft2c(self, d):
        return tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(d)))

#************************************************************************************************************
# configure network and optimization environment including loss and variables
        
    def set_network(self, Gs,D, minibatch_size=1):
        assert minibatch_size == 1
        self._Gs = Gs
        self._D = D
        self.initial_Gs = Gs.clone()
        self._minibatch_size = minibatch_size
        if self._Gs is None:
            return
        if self.clone_net:
            self._Gs = self._Gs.clone()

        
        # find average latent vector to be starting point of the optimization
        self._info("Initializing average latent using %d samples..." % self.dlatent_avg_samples)
        latent_samples = np.random.RandomState(123).randn(self.dlatent_avg_samples, *self._Gs.input_shapes[0][1:])
        
        # latent positional encoding (not important at this step)
        latent_pos = np.ones([16,32])
        dlatent_samples = self._Gs.components.mapping.run(latent_samples, None, latent_pos, None, is_training=False)[:, :, :1, :] # [N, 1, 512]
        
        # average latent vectors
        self._dlatent_avg = np.mean(dlatent_samples, axis = 0, keepdims = True) # [1, 1, 512]

#************************************************************************************************************
# construct weight tensors and initializer ops.  
        self._weight_vars = []
        weight_init_ops = []
        self.weights_ops = []
        self.initial_weights = []
        for w in self._Gs.vars:
            # find convolutional layer weights from TensorFlow graph to optimize
            if 'Conv1/weight' in w:
                
                # print target weights to be used in inference
                print(w)
                m = self._Gs.vars[w]
                
                # save a copy of each weight to initialize at the next image
                m_copy = self.initial_Gs.vars[w] 
                
                self.initial_weights.append(m_copy)
                self.weights_ops.append(m)
                weight_init_ops.append(tf.assign(m, m_copy))
        self._weight_init_op = tf.group(*weight_init_ops)    

#************************************************************************************************************
# construct noise variables and initializer ops.
        self._noise_vars = []
        noise_init_ops = []
        noise_normalize_ops = []
        while True:
            n = 'G_synthesis/noise%d' % len(self._noise_vars)
            if not n in self._Gs.vars:
                break
            v = self._Gs.vars[n]
            self._noise_vars.append(v)
            noise_init_ops.append(tf.assign(v, tf.random_normal(tf.shape(v), dtype=tf.float32)))
            noise_mean = tf.reduce_mean(v)
            noise_std = tf.reduce_mean((v - noise_mean)**2)**0.5
            noise_normalize_ops.append(tf.assign(v, (v - noise_mean) / noise_std))
            self._info(n, v)
        self._noise_init_op = tf.group(*noise_init_ops)
        self._noise_normalize_op = tf.group(*noise_normalize_ops)

#************************************************************************************************************
# necessary settings and image output graph

        # check target contrast to create undersampling mask shape
        if self.contrast == 'T1' or self.contrast=='FLAIR':
            self.mask = tf.Variable(tf.zeros([256,320], dtype=tf.complex64), trainable=False, dtype=tf.complex64)
        else:
            self.mask = tf.Variable(tf.zeros([288,384], dtype=tf.complex64), trainable=False, dtype=tf.complex64)
            
        print("mask shape: " + str(self.mask.shape))
        
        # find padding dimensions to fit into 512x512 images
        self.pad_x = int((512 - self.mask.shape[0]) // 2)
        self.pad_y = int((512 - self.mask.shape[1]) // 2)
        self.coil_map = tf.Variable(tf.zeros([(512- 2 * self.pad_x) ,(512- 2 * self.pad_y) ,5], dtype=tf.complex64), trainable=False, dtype=tf.complex64)
        
        # hold intermediate latent vectors in a single TensorFlow variable ( 1 global + k local components)                             
        self._dlatents_var = tf.Variable(tf.zeros([1,17,17,32]), name = "dlatents_var")
        
        # initialize latent position and component masks (will be overwritten at start)
        self.latent_pos = tf.Variable(tf.ones([16,32]))
        self.component_mask = tf.Variable(tf.ones([1,1,16]))

        # get generated images from synthesizer to graph
        self._images_expr, self.attention_maps = self._Gs.components.synthesis.get_output_for(self._dlatents_var, self.latent_pos,self.component_mask,randomize_noise = False, use_pos=True)

#************************************************************************************************************
# build loss graph
        self._target_images_var = tf.Variable(tf.zeros(self._images_expr.shape), name='target_images_var')

        # first convert target images from multi-channel to complex tensors
        self.target_images_var_complex = tf.squeeze(tf.complex(self._target_images_var[:,0,:,:], self._target_images_var[:,1,:,:]))
        # stack them to project onto single coils
        self.target_images_var_complex = tf.stack([self.target_images_var_complex,self.target_images_var_complex,self.target_images_var_complex,self.target_images_var_complex,self.target_images_var_complex],axis=2)
        # clip target images to fit into original size
        self.target_images_var_complex = self.target_images_var_complex[self.pad_x:(512-self.pad_x), self.pad_y:(512-self.pad_y), :]
        # multiply target images with coil maps to project coil-combined targets to sepearate coils
        self.full_org_image_coil_separate = tf.multiply(self.target_images_var_complex, self.coil_map)
        # create a coil-seperated mask
        self.coil_seperate_mask = tf.stack([self.mask, self.mask, self.mask, self.mask, self.mask], axis=2)
        # take 2d centered fourier transform of the coil seperated target images
        self.full_kspace_org_image_coil_separate = self.fft2c_multi(self.full_org_image_coil_separate)
        # undersample coil-seperated target image
        self.undersampled_kspace_org_image_coil_separate = tf.multiply(self.full_kspace_org_image_coil_separate ,self.coil_seperate_mask) 

        # same operations for generated images
        self.proc_images_expr_complex = tf.squeeze(tf.complex(self._images_expr[:,0,:,:],self._images_expr[:,1,:,:]))
        self.proc_images_expr_complex = self.proc_images_expr_complex[self.pad_x:(512-self.pad_x), self.pad_y:(512-self.pad_y)]
        self.proc_images_expr_complex = tf.stack([self.proc_images_expr_complex,self.proc_images_expr_complex,self.proc_images_expr_complex,self.proc_images_expr_complex,self.proc_images_expr_complex],axis=2)
        self.proc_images_expr_complex_coil_separate = tf.multiply(self.proc_images_expr_complex, self.coil_map)
        self.full_kspace_gen_image = self.fft2c_multi(self.proc_images_expr_complex_coil_separate)
        self.undersampled_kspace_gen_image_coil_separate = tf.math.multiply(self.full_kspace_gen_image,self.coil_seperate_mask)

        # combine l2 and l1-like losses (can be used separately)
        self._loss = tf.abs(tf.reduce_mean(tf.math.squared_difference(self.undersampled_kspace_org_image_coil_separate , self.undersampled_kspace_gen_image_coil_separate))) 
        self._loss += tf.reduce_mean(tf.abs(self.undersampled_kspace_org_image_coil_separate - self.undersampled_kspace_gen_image_coil_separate))

#************************************************************************************************************
# set up the optimizer
        self._lrate_in = tf.placeholder(tf.float32, [], name='lrate_in')   # adjust learning rate variable to be able to change in every step
        self._opt = dnnlib.tflib.Optimizer(learning_rate=self._lrate_in)   # initalize optimizer
        self._opt.register_gradients(self._loss, [self._dlatents_var] 
        + self.weights_ops + self._noise_vars)                             # draw gradient descent way by registering gradients 
        self._opt_step = self._opt.apply_updates()                         # define a single optimization step

#************************************************************************************************************
    def run(self, target_images):
        # start inference
        self.start(target_images, self.mask, self.coil_map)
        while self._cur_step < self.num_steps:
            self.step()

        pres = dnnlib.EasyDict()
        pres.dlatents = self.get_dlatents()
        pres.noises = self.get_noises()
        return pres

#************************************************************************************************************
    def start(self, target_images, mask, coil_map):
        assert self._Gs is not None

        self.target_images_initial = target_images

        # prepare coil combined target images.
        target_images = np.asarray(target_images.copy(), dtype='float32')                    
        self.target_images = target_images
        
        # initialize optimization state.
        tflib.set_vars({self._target_images_var: target_images,self._dlatents_var: np.tile(self._dlatent_avg, (self._minibatch_size, 1, 17, 1)), self.mask:mask, self.coil_map :coil_map, self.latent_pos:np.random.normal(0,1,[16,32]), self.component_mask: np.ones([1,1,16])})
        tflib.run(self._noise_init_op)
        tflib.run(self._weight_init_op)
        self._opt.reset_optimizer_state()
        self._cur_step = 0

#************************************************************************************************************
    def step(self):
        assert self._cur_step is not None
        if self._cur_step >= self.num_steps:
            return
        if self._cur_step == 0:
            self._info('Reconstructing...')

        # learning schedule hyperparameters.
        t = self._cur_step / self.num_steps
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        learning_rate = self.initial_learning_rate * lr_ramp

        # run a single step (normalize noises back to normal)
        feed_dict = {self._lrate_in: learning_rate}
        _, _ = tflib.run([self._opt_step, self._loss], feed_dict)
        tflib.run(self._noise_normalize_op)

        self._cur_step += 1

#************************************************************************************************************
    def get_cur_step(self):
        return self._cur_step

    def get_dlatents(self):
        return tflib.run(self._dlatents_var)

    def get_noises(self):
        return tflib.run(self._noise_vars)

#************************************************************************************************************
    
    def untouched_images(self):
        return tflib.run(self._images_expr) # return original outputs without data-consistency

    def get_mask(self):
        return tflib.run(self.mask) # return undersampling mask
    
    def get_coil_maps(self):
        return tflib.run(self.coil_map) # return coil maps
    
    def get_attention_maps(self):
        return tflib.run(self.attention_maps) # return attention maps

#************************************************************************************************************
    # perform data-consistency and return final reconstructed images
    def get_images(self):
       gen_im = tflib.run(self._images_expr)
       
       # get original mask and coil maps
       mask = self.get_mask()
       coil_maps_ = self.get_coil_maps()
       
       # convert channel seperated target and generated images to complex numpy arrays
       images_real = gen_im[0,0,:,:]
       images_imag = gen_im[0,1,:,:]
       images = images_real + 1j * images_imag
       images = np.stack([images,images,images,images,images], axis=2)
       images = images[self.pad_x:(512-self.pad_x), self.pad_y:(512-self.pad_y), :]
       
       target_images_ = self.target_images_initial.copy()
       target_images_real = target_images_[0,0,:,:]
       target_images_imag = target_images_[0,1,:,:]
       target_images = target_images_real + 1j * target_images_imag
       target_images = np.stack([target_images, target_images, target_images, target_images,target_images], axis=2)
       target_images = target_images[self.pad_x:(512-self.pad_x), self.pad_y:(512-self.pad_y), :]
       
       # generate coil seperated images by multiplying with coil maps
       images_coil_separate = images * coil_maps_
       target_images_coil_separate = target_images * coil_maps_
       
       # perform data-consistency
       kspace_generated = self.fft2c_multi_np(images_coil_separate)
       full_kspace_org_image = self.fft2c_multi_np(target_images_coil_separate)
       mask = np.stack([mask,mask,mask,mask,mask], axis=2)
       kspace_generated[mask>0] = full_kspace_org_image[mask>0]
       
       # take inverse fourier and convert complex coil-seperated image to coil-combined magnitude image
       resulting_images_coil_separate = self.ifft2c_multi_np(kspace_generated)
       images_ = np.abs(np.sum(resulting_images_coil_separate*np.conj(coil_maps_),axis=2))
       
       return images_[np.newaxis][np.newaxis]


#----------------------------------------------------------------------------
