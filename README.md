# Exploring mode collapse in GANS

## What is Mode Collapse?

Usually, you want your GAN to produce a wide variety of outputs. You want, for example, a different face for every random input to your face generator.However, if a generator produces an especially plausible output, the generator may learn to produce only that output.

If the generator starts producing the same output (or a small set of outputs) over and over, the discriminator’s best strategy is to learn to always reject that output. But if the next iteration of discriminator gets stuck in a local minimum and doesn’t find the best strategy, then  it’s easy for the next generator iteration to find the most plausible output for the current discriminator. Each iteration of generator over-optimizes for a particular discriminator and the discriminator never manages to learn its way out of the trap. As a result, the generators rotate through a small set of output results. This form of GAN failure is called mode collapse.

![DC GAN Flow](https://github.com/rohitpatwa/gans-mode-collapse/blob/main/media/GANs.png)

## Approach

Following is the DCGAN architecture we used to reproduce and fix the mode collapse issue.

```
Generator(
(main): Sequential(
(0) ConvTranspose2d(100, 512, kernel s ize = (4, 4), stride = (1, 1), bias = F alse)
(1) : BatchN orm2d(512, eps = 1e − 05, momentum = 0.1, af f ine = T rue)
(2) : ReLU (inplace = T rue)
(3) : ConvT ranspose2d(512, 256, kernel s ize = (4, 4), stride = (2, 2), padding = (1, 1))
(4) : BatchN orm2d(256, eps = 1e − 05, momentum = 0.1, af f ine = T rue)
(5) : ReLU (inplace = T rue)
(6) : ConvT ranspose2d(256, 128, kernel s ize = (4, 4), stride = (2, 2), padding = (1, 1))
(7) : BatchN orm2d(128, eps = 1e − 05, momentum = 0.1, af f ine = T rue)
(8) : ReLU (inplace = T rue)
(9) : ConvT ranspose2d(128, 64, kernel s ize = (4, 4), stride = (2, 2), padding = (1, 1))
(10) : BatchN orm2d(64, eps = 1e − 05, momentum = 0.1, af f ine = T rue)
(11) : ReLU (inplace = T rue)
(12) : ConvT ranspose2d(64, 1, kernel s ize = (4, 4), stride = (2, 2), padding = (1, 1))
(13) : T anh()
))

Discriminator(
(main) : Sequential(
(0) : Conv2d(1, 64, kernel s ize = (4, 4), stride = (2, 2), padding = (1, 1))
(1) : LeakyReLU (negative s lope = 0.2, inplace = T rue)
(2) : Conv2d(64, 128, kernel s ize = (4, 4), stride = (2, 2), padding = (1, 1))
(3) : BatchN orm2d(128, eps = 1e − 05, momentum = 0.1, af f ine = T rue)
(4) : LeakyReLU (negative s lope = 0.2, inplace = T rue)
(5) : Conv2d(128, 256, kernel s ize = (4, 4), stride = (2, 2), padding = (1, 1))
(6) : BatchN orm2d(256, eps = 1e − 05, momentum = 0.1, af f ine = T rue)
(7) : LeakyReLU (negative s lope = 0.2, inplace = T rue)
(8) : Conv2d(256, 512, kernel s ize = (4, 4), stride = (2, 2), padding = (1, 1))
(9) : BatchN orm2d(512, eps = 1e − 05, momentum = 0.1, af f ine = T rue)
(10) : LeakyReLU (negative s lope = 0.2, inplace = T rue)
(11) : Conv2d(512, 1, kernel s ize = (4, 4), stride = (1, 1))
(12) : Sigmoid()
))
```

Below are the set of hyperparameters used to train the DCGAN.
```
BatchSize = 64
ImageSize = 64
nz = 100 # dimension of latent vector z
ngf = 64 # number of filters in generator
ndf = 64 # number of filters in discriminator
niter = 70 # number of epochs
lr = 0.0002 # learning rate
```

Notice that the number of features used by the Discriminator are too less (on the basis of which it makes a prediction). Generally, the generator exploits these features and produces a gibberish output which fools the discriminator to believe that it is a real image. Thus to avoid overfitting the discriminator, we added a noisy label. We did this by flipping a certain percentage of labels being fed to the discriminator as the ground truth. This avoids overfitting and ultimately prevents mode collapse.

![Labels flipped](https://github.com/rohitpatwa/gans-mode-collapse/blob/main/media/labels.png)

## Results

We tried 6 experiments on MNIST dataset with erroneous components in the discriminator.
All experiments were repeated 5 times and the average result was considered to determine at which epoch did the mode collapse.

| Percentage error | Epochs for mode collapse |
| :---------------:| :-----------------------:|
| 0%               | 13                       |
| 1%               | 15                       |
| 5%               | 25                       |
| 10%              | 70+                      |
| 15%              | 70+                      |
| 20%              | 70+                      |

The quality of generated images for all the 6 experiments were comparable and the only noticeable change was the number of epochs it took for mode to collapse.

![Results](https://github.com/rohitpatwa/gans-mode-collapse/blob/main/media/results.png)

## Technologies

* Python
* PyTorch
* Google Colab
* MNIST and FashionMNIST dataset

## Authors

* **Rohit Patwa** [\[LinkedIn\]](https://www.linkedin.com/in/rohitpatwa/)
* **Ativeer Patni** [\[LinkedIn\]](https://www.linkedin.com/in/ativeer-patni/)

## References

* NIPS 2016 Tutorial: Generative Adversarial Networks by Ian Goodfellow  
[https://arxiv.org/pdf/1701.00160.pdf](https://arxiv.org/pdf/1701.00160.pdf)

* ON CONVERGENCE AND STABILITY OF GANS by Naveen Kodali, Jacob Abernethy, James Hays Zsolt Kira  
[https://arxiv.org/pdf/1705.07215.pdf](https://arxiv.org/pdf/1705.07215.pdf)
