
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