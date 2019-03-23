
# pix2pix in Tensorflow and Keras

An implementation of the pix2pix [paper](https://arxiv.org/pdf/1611.07004.pdf) using Keras to build models and Tensorflow to train.

The model is trained on the façades dataset. In this setting the model is provided with a diagram of a buildings' facade – showing the layout of windows, doors, balconies, mantels - with the objective being to generate a photo-realistic rendering.

A webpage is updated during training so that you can watch the model learn. You can see the development of concepts such as reflective windows, dampness and mildew on render, stonework detail, and shadows under balconies. Here's a few examples from the end of training. 

- Input: the diagram provided to model as reference
- Authors' Pytorch: Generated output of model provided by the authors of original paper
- This Implementation: Generated output of this model
- Target: A real photograph of the building
- Patchgan: A heatmap visualisation showing which parts of the generated image in 3rd column the discriminator classifies as real (white) and fake (grey).

![](./results/end_of_training.png)

See the full training results by downloading this repo and opening `results/index.html` in your browser. Or train the model yourself by following the steps below.

## Install dependencies

Tensorflow 1.13.1 requires CUDA 10 drivers if running on GPU, installation steps [here](https://www.tensorflow.org/install/gpu#install_cuda_with_apt). If running on CPU, change `tensorflow-gpu` to `tensorflow` in `requirements.txt`.

Setup python environment:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train model

Download facades dataset:

```
bash download_dataset.sh façades
```

Preprocess data:

```
python preprocess.py
```

Train:
```
python train.py --experiment_title my_experiment
```

## Watch the model learn!

To view results as the model trains open `results/index.html` in your browser to view training progress visualisations, including training plots and checkpoint images for each epoch.

## References

- **The 'pix2pix' paper on which this implementation is based:** P. Isola, J. Zhu, T. Zhou, A. Efros. Image-to-Image Translation with Conditional Adversarial Networks. (https://arxiv.org/pdf/1611.07004.pdf)
    - Authors' [PyTorch implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
    - Authors' original [Lua implementation](https://github.com/phillipi/pix2pix)
- **The original GAN paper** I. Goodfellow et al. Generative Adversarial Networks (https://arxiv.org/abs/1406.2661)
- **U-net architecture used by generator:** O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In MIC- CAI, pages 234–241. Springer, 2015. 2, 3, 4 (https://arxiv.org/abs/1505.04597)
- **Insights on receptive field theory exploited by Patchgan discriminator:** W. Luo, Y. Li, R. Urtasun, R. Zemel. Understanding the Effective Receptive Field in Deep Convolutional Neural Networks (https://arxiv.org/abs/1701.04128)
- **Useful overview of theory challenges and tricks when training GANS:** T. Salimans et al. Improved Techniques for Training GANs (https://arxiv.org/pdf/1606.03498.pdf)


