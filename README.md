
# Generative Adversarial Network (GAN) Example using Keras

The goal of this GAN is to generate fake date from a dataset shown below. 

![Original Dataset](images/dataset.png)

<br>

This is a shapshot of how the training gradually improves. 

![GAN Training](images/GAN_training.gif)

<br>

The final results are shown below. 
NOTE that the model is not able to recrease the entire sampling distribution. 
This is due to mode colapse, which is a hot topic in regards to GANs. 

![](images/good.png)

<br>

A worse mode colapse is shown below. 

![Mode Colapse](images/mode_colapse.png)

<br>

Adding regularization and dropout seem to enforce mode collapse. 

![Regularization](images/regularization.png)

