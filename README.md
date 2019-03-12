
## Instructions

Download facades dataset:

```
bash download_dataset.sh facades
```
```
mkdir data/facades_processed
python preprocess.py
```


### Implementation Notes

- [x] switch out generator
- [x] switch out discriminator (check patch size)
- [x] delete obsolete model code in pix2pix.py
- [x] rm superfluous input to gan
- [x] sample discriminator predictions
- [x] check 32x32 patch size is correct
- [x] implement instance norm
- [x] check biases in generator
- [x] is generator maximising discriminator loss? (i've got a feeling ti might be opposite)
- [x] should discriminator/generator train switch be at batch level or epoch level? yes
- [x] are the discriminators weights are being updated correctly in gan? yes, see `check_discriminator_weights_match`
- [x] check loss parsing
- [x] write training progress to disk
- [x] check data augmentation
- [x] switch out data loader
- [ ] learning rate decay (see options pytorch)
- [ ] check hyperparams
- [ ] implement instance normalisation with from keras_contrib.layers.normalization import InstanceNormalization

- [ ] implement sce and mse loss


### Param check

https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py

- [x] normalisation : 
    - momentum: 0.9 -> 0.1
    - gamma initialisation added   
- [x] instance normalisation implemented correctly?
    - training = True (use inference stats)
- [x] dropout on at test time?
- [x] generate evaluation samples with batch size 1, to ensure bn stats are same
- [ ] loss functions
    - seems to implement label switichin fake=>1 in gan fake => 0 in discrim
    - applies exponential moving average to losses with decay 0.99
- [ ] loss weights

## gan hacks
- [x] one-sided label smoothing
- [x] minibatch discrimination - implemented as minibatch standard deviation
- [ ] feature matching
- [ ] historical averaging

# pytorch ref
- [x] run pytorch reference
- [x] match logging
- [x] match params
- [x] match params printout
- [x] match g loss function
- [x] understand why rgb=>lab colour space conversion is needed
    - it is only used in colourspace implementation
- [x] match d loss function
- [ ] training baseline images from pytorch implementation
- [ ] training review webpage with side-by-sides 
- [ ] training curves


- [ ] match discriminator deconvolutions
- [ ] match generator deconvolutions

- [ ] check data normalisation
- [ ] check activation functions
- [] copy visualizers
- [] implement model saving
- [] save more images
- [] check hd space
- [] implement n critic
- [] run experiments




### Cleanup
- [ ] conda freeze
- [ ] report keras training bug on dropout and batch norm layers


What is a ReLU?

- a Rectified Linear Unit is an activation function that sets all negative values to zero. Positive values are treated linearly. Similar to an ideal rectifier in electronics. 

What is a Leaky ReLU?

- Leaky ReLUs allow a small positive gradient for negative values. 
- e.g. if x>0: f(x) = x, else: f(x) = 0.01x

Should activations be passed to skip connection before ReLU is applied?
- The pix2pix code applies ReLU at input to each block, rather than at ouput.
- The pix2pix code is difficult to interpret here, not sure if skips should be pro-or post activation? The difference would be in whether LeakyReLU or ReLU is applied to skip activations. Not sure. Will try later

Should innermost layers of U-net have a skip connection?
- see my sketch in appendix to show that no, it shouldn't 
- pix2pix code excludes skip connections from innermost outermost layers

Where should skip connections be concatenated?

Is dropout in pix2pix repo as described in paper?

What are model adjustments for color transfer?

Should u-net have odd number of layers?
- unet in orginal paper has odd number of layers, but pix2pix has two unconnected layers at bottom of U

How should upsampling be performed in the U-net generator?
- [30] suggests "fractional-strided convolutions", which I assume means for example a stride of 0.5. It is suggested that downsampling and upsampling with strided convolutions is superior to spatial pooling functions because they allow the network to learn its own spatial downsampling and upsampling.
- [34] unet papers suggests 'Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2 convolution (“up-convolution”) that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions, each fol- lowed by a ReLU.", must assume stride=1, else no upsampling would be achieved
- The pix2pix implementation uses the `ConvTranspose2d` pytorch layer for upsampling, the docs for which note this function is otherwise known as fractional-strided convolution as mentioned by [30], so that ties together.
- Keras has a `Conv2DTranspose` layer which seems to be equivalent so I'll use that.

What is softmax function?

What is entropy?

What is the receptive field of a convolutional neural network?
- Theoretical receptive field (RF): 
    - https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
    - https://mathematica.stackexchange.com/questions/133927/how-to-compute-the-receptive-field-of-a-neuron/151825#151825
- Effective receptive field (ERF):
    - https://papers.nips.cc/paper/6203-understanding-the-effective-receptive-field-in-deep-convolutional-neural-networks.pdf

What is the difference between the L1 norm and the L2 norm

Whats is Jensen–Shannon divergence?

What is the central limit theorem (CLT)?

What is earth-mover distance (EM)?

What is the Lipschitz condition and why do standar feed-forward neural nets satisfy it?

What is cross-entropy loss?

What is the difference between **fractionally-strided convolution**  and **dilated convolution**? 

Which layers should include bias?

Why are only 2 output channels used in colorization?




### References

- [30] A. Radford, L. Metz, and S. Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015.
- [34] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In MIC- CAI, pages 234–241. Springer, 2015. 2, 3, 4
- https://arxiv.org/pdf/1611.07004.pdf
- https://richzhang.github.io/ideepcolor/
- http://richzhang.github.io/colorization/


### Pytorch Params

----------------- Options ---------------
               batch_size: 1                       #      
                    beta1: 0.5                     #     
          checkpoints_dir: ./checkpoints                 
           continue_train: False                         
                crop_size: 256                     #      
                 dataroot: ./datasets/facades                   [default: None]
             dataset_mode: aligned                 ?     
                direction: BtoA                                 [default: AtoB]
              display_env: main                          
             display_freq: 400                     #      
               display_id: 1                             
            display_ncols: 4                             
             display_port: 8097                          
           display_server: http://localhost              
          display_winsize: 256                           
                    epoch: latest                        
              epoch_count: 1                             
                 gan_mode: vanilla                 # means bceloss      
                  gpu_ids: 0                             
                init_gain: 0.02                    # torch.nn.init.normal_(mean=0.0, std=0.02)    
                init_type: normal                  # init type normal distribution     
                 input_nc: 3                             
                  isTrain: True                                 [default: None]
                lambda_L1: 100.0                   #     
                load_iter: 0                                    [default: 0]
                load_size: 286                           
                       lr: 0.0002                 #       
           lr_decay_iters: 50                            
                lr_policy: linear                        
         max_dataset_size: inf                           
                    model: pix2pix                              [default: cycle_gan]
               n_layers_D: 3                             
                     name: facades_pix2pix                      [default: experiment_name]
                      ndf: 64                            
                     netD: basic                 # => patchgan RF=70        
                     netG: unet_256              # check        
                      ngf: 64                            
                    niter: 100                           
              niter_decay: 100                           
               no_dropout: False                # TODO: could check drouput is working        
                  no_flip: False                         
                  no_html: False                         
                     norm: batch               # batch, matched pytorch defaults
                                               # notes: not confident its applied at inference time by training=True param         
              num_threads: 4                             
                output_nc: 3                             
                    phase: train                         
                pool_size: 0                             
               preprocess: resize_and_crop               
               print_freq: 100                           
             save_by_iter: False                         
          save_epoch_freq: 5                             
         save_latest_freq: 5000                          
           serial_batches: False                         
                   suffix:                               
         update_html_freq: 1000                          
                  verbose: False                         
----------------- End -------------------
dataset [AlignedDataset] was created
The number of training images = 400
initialize network with normal
initialize network with normal
model [Pix2PixModel] was created
---------- Networks initialized -------------
[Network G] Total number of parameters : 54.414 M   # 54,414,019 match
[Network D] Total number of parameters : 2.769 M    # 2,769,601 match
-----------------------------------------------
