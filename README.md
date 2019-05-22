
# Generative Adversarial Network (GAN) Example using Keras. 

The goal of this GAN is to generate fake date from a dataset shown below. 

![Original Dataset](images/dataset.png)

<br><br>

This is a shapshot of how the training gradually improves. 

![GAN Training](images/GAN_training.gif)

<br><br>

The final results are shown below. 

![](images/good.png)

<br><br>

A known issue with GAN is mode collapse. 
This is when the generator learns to generate a subset of the real distribution. 

![Mode Colapse](images/mode_colapse.png)

![Regularization](images/regularization.png)

![Strange learning](images/line.png)
