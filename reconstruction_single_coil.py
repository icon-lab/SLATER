# single coil reconstruction class
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

from training import misc

class Reconstructor:
    def __init__(self):
        
        # inference settings
        self.num_steps                  = 1000          # number of optimization / inference steps
        self.dlatent_avg_samples        = 10000         # number of latent samples to be used to calculate average latent 
        self.initial_learning_rate      = 0.1           # initial learning rate to reach with ramp up
        self.lr_rampdown_length         = 0.25          # learning rate ramp down length ( 0.25 * 1000 = last 250 steps) 
        self.lr_rampup_length           = 0.05          # learning rate ramp up length (0.1 * 1000 = first 100 steps)
        
        # main settings
        self.verbose                    = False         # enable prints & reports for user
        self.clone_net                  = True          # clone network (beneficial for weight optimization)
        self._cur_step                  = None          # current step of inference


#************************************************************************************************************
# fourier operations defined for numpy arrays
        
    # single-array centered fft    
    def fft2c_np(self,im):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(im))) 
    
    # single-array centered ifft
    def ifft2c_np(self,d):
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(d)))
    
#************************************************************************************************************
# fourier operations defined for TensorFlow tensors   
    
    # single-array centered fft    
    def fft2c(self, im):
        return tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(im))) 
    
    # single-array centered ifft
    def ifft2c(self, d):
        return tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(d)))
    
#************************************************************************************************************
    def _info(self, *args):
        if self.verbose:
            print("Reconstructor: ", *args)
            
#************************************************************************************************************
# configure network and optimization environment including loss and variables
            
    def set_network(self, Gs, minibatch_size = 1):
        assert minibatch_size == 1
        #Gs.reset_vars()
        self._Gs = Gs
        self.initial_Gs = Gs.clone()
        print(Gs)
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
        dlatent_samples = self._Gs.components.mapping.run(latent_samples, None,latent_pos,None,is_training=False)[:, :, :1, :] # [N, 1, 512]
        
        # average latent vectors
        self._dlatent_avg = np.mean(dlatent_samples, axis = 0, keepdims = True) # [1, 1, 512]

#************************************************************************************************************
# construct noise variables and initializer ops.
        
        self._noise_vars = []
        noise_init_ops = []
        noise_normalize_ops = []
        while True:
            n = "G_synthesis/noise%d" % len(self._noise_vars)
            if n not in self._Gs.vars:
                break
            v = self._Gs.vars[n]
            self._noise_vars.append(v)
            noise_init_ops.append(tf.assign(v, tf.random_normal(tf.shape(v), dtype = tf.float32)))
            noise_mean = tf.reduce_mean(v)
            noise_std = tf.reduce_mean((v - noise_mean)**2)**0.5
            noise_normalize_ops.append(tf.assign(v, (v - noise_mean) / noise_std))
            self._info(n, v)
        self._noise_init_op = tf.group(*noise_init_ops)
        self._noise_normalize_op = tf.group(*noise_normalize_ops)
        
        
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
# necessary settings and image output graph
        
        self.mask = tf.Variable(tf.zeros([256,256], dtype=tf.complex64),dtype=tf.complex64)
        
        # hold intermediate latent vectors in a single TensorFlow variable ( 1 global + k local components)                             
        self._dlatents_var = tf.Variable(tf.zeros([1,17,15,32]), name = "dlatents_var")
        
        # latent positional embeddings (not important at this step)
        self.latent_pos = tf.Variable(tf.zeros([16,32]))
        
        # get generated images from synthesizer to graph
        self._images_expr, self.attention_maps = self._Gs.components.synthesis.get_output_for(self._dlatents_var, self.latent_pos,None,randomize_noise = False, use_pos=False)

        # convert generated magnitude images to [0, 1] range
        proc_images_expr = (self._images_expr + 1)  / 2
    
#************************************************************************************************************
# build loss graph
        
        self._target_images_var = tf.Variable(tf.zeros(proc_images_expr.shape), name = "target_images_var")
        
        # convert target images to complex tensors
        self.target_images_var_complex = tf.cast(self._target_images_var, dtype=tf.complex64)
        
        # take centered 2d fft of target images
        self.full_kspace_org_image = self.fft2c(self.target_images_var_complex[0,0,:,:])
        
        # undersample target images
        self.undersampled_kspace_org_image = tf.math.multiply(self.full_kspace_org_image, self.mask)

        # same operations as above for generated images
        self.proc_images_expr_complex = tf.cast(proc_images_expr, dtype=tf.complex64)
        self.full_kspace_gen_image = self.fft2c(self.proc_images_expr_complex[0,0,:,:])
        self.undersampled_kspace_gen_image = tf.math.multiply(self.full_kspace_gen_image,self.mask)

        self._loss = tf.reduce_mean(tf.abs(self.undersampled_kspace_org_image - self.undersampled_kspace_gen_image))


#************************************************************************************************************
# set up the optimizer
        self._lrate_in = tf.placeholder(tf.float32, [], name='lrate_in')   # adjust learning rate variable to be able to change in every step
        self._opt = dnnlib.tflib.Optimizer(learning_rate=self._lrate_in)   # initalize optimizer
        self._opt.register_gradients(self._loss, [self._dlatents_var] 
        + self.weights_ops + self._noise_vars)                             # draw gradient descent way by registering gradients 
        self._opt_step = self._opt.apply_updates()                         # define a single optimization step

#************************************************************************************************************
    def start(self, target_imgs, mask):
        assert self._Gs is not None

        # convert target images' range to [0, 1]
        self.target_images_initial = target_imgs.copy()
        target_imgs = np.asarray(target_imgs, dtype = "float32")
        target_imgs = (target_imgs + 1) / 2
        print(target_imgs.shape)

        # initialize optimization state.
        tflib.set_vars({self._target_images_var: target_imgs,
            self._dlatents_var: np.tile(self._dlatent_avg, (self._minibatch_size, 1, 15, 1)), self.mask: mask, self.latent_pos:np.random.normal(0,1,[16,32])})
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
            self._info("Running...")

        # learning schedule hyperparameters.
        t = self._cur_step / self.num_steps
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        learning_rate = self.initial_learning_rate * lr_ramp

        # run a single step (normalize noises back to normal)
        feed_dict = {self._lrate_in: learning_rate}
        _, loss_value = tflib.run([self._opt_step, self._loss], feed_dict)
        tflib.run(self._noise_normalize_op)

        self._cur_step += 1
        
#************************************************************************************************************

    def get_cur_step(self):
        return self._cur_step

    def get_mask(self):
        return tflib.run(self.mask)

    def get_dlatents(self):
        return tflib.run(self._dlatents_expr)

    def get_noises(self):
        return tflib.run(self._noise_vars)
    
    def untouched_images(self):
        return tflib.run(self._images_expr)

    def get_attention_maps(self):
        return tflib.run(self.attention_maps)

    # perform data consistency and return images
    def get_images(self):
       gen_im = tflib.run(self._images_expr)
       # get current mask
       mask = self.get_mask()
       # adjust range of images to [0,1] before data-consistency
       image_adjusted = misc.adjust_dynamic_range(gen_im[0,0,:,:], [np.min(gen_im[0,0,:,:]), np.max(gen_im[0,0,:,:])], [0,1])
       target_images_ = self.target_images_initial.copy()
       target_images_ = (target_images_ + 1) / 2
       kspace__ = self.fft2c_np(image_adjusted)
       target_images_var_complex = np.complex64(target_images_[0,0,:,:])
       full_kspace_org_image = self.fft2c_np(target_images_var_complex)
       # apply data-consistency
       kspace__[mask>0] = full_kspace_org_image[mask>0]
       images_ = np.float32(np.abs(self.ifft2c_np(kspace__)))
       images_[images_>1]=1
       # make non-brain regions zero (not necessary)
       images_[:,0:56] = 0
       images_[:,200:256] = 0
       return images_[np.newaxis][np.newaxis]
