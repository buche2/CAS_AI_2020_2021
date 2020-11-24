# CAS Artificial intelligence 2020-2021

CAS about artificial intelligence made at BFH in 2020 and 2021.

- [CAS Artificial intelligence 2020-2021](#cas-artificial-intelligence-2020-2021)
- [Ressources](#ressources)
  * [Interative course script](#interative-course-script)
  * [Data](#data)
  * [Videos](#videos)
  * [Tutorial](#tutorial)
  * [Book](#book)
- [2020-10-20 1.Einführung in AI Grundtechniken: Gradient Descent,partielle Ableitungen, Matrix Algebra, AI Frameworks](#2020-10-20-1einf-hrung-in-ai-grundtechniken--gradient-descent-partielle-ableitungen--matrix-algebra--ai-frameworks)
  * [Theory](#theory)
    + [General AI principle](#general-ai-principle)
    + [Notes](#notes)
  * [Home work](#home-work)
- [2020-10-26 2.Tensorflow und PyTorch Frameworks](#2020-10-26-2tensorflow-und-pytorch-frameworks)
  * [Theory](#theory-1)
    + [Data loading](#data-loading)
    + [Tensorflow introduction](#tensorflow-introduction)
    + [Pytorch introduction](#pytorch-introduction)
    + [Notes](#notes-1)
  * [Homework](#homework)
- [2020-11-03 3.Fundamentale Neuronale Netze: MLP und Autoencoder](#2020-11-03-3fundamentale-neuronale-netze--mlp-und-autoencoder)
  * [Theory](#theory-2)
  * [Homework](#homework-1)
- [2020-11-10 4.Convolutional Neuronal Networks (CNN) und Variational Autoencoders (VAE)](#2020-11-10-4convolutional-neuronal-networks--cnn--und-variational-autoencoders--vae-)
  * [Theory](#theory-3)
    + [Notes](#notes-2)
  * [Homework](#homework-2)
- [2020-11-17 5.Transfer-Learning Methoden und Deep ConvolutionalGenerative Adversarial Network (DCGAN)](#2020-11-17-5transfer-learning-methoden-und-deep-convolutionalgenerative-adversarial-network--dcgan-)
  * [Theory](#theory-4)
    + [Notes](#notes-3)
  * [Homework](#homework-3)
- [2020-11-24 6.Recurrent Neuronal Networks](#2020-11-24-6recurrent-neuronal-networks)
  * [Theory](#theory-5)
    + [Notes](#notes-4)
  * [Homework](#homework-4)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

  
# Ressources

## Interative course script

[Script as notebook](https://colab.research.google.com/drive/10vkq0wYDpZJkxRWzlxwcRFY1q0tOooCO?usp=sharing)

## Data

[Machine learning data repository](https://archive.ics.uci.edu/ml/datasets.php)

[IRIS](https://gist.github.com/curran/a08a1080b88344b0c8a7)

## Videos

## Tutorial

[Colab](https://colab.research.google.com/notebooks/intro.ipynb)

## Book

1. Machine Learning, Tom Mitchell, McGraw Hill, 1997 http://www.cs.cmu.edu/~tom/mlbook.html
2. Deeplearning, Ian Goodfellow https://www.deeplearningbook.org/contents/mlp.html


# 2020-10-20 1.Einführung in AI Grundtechniken: Gradient Descent,partielle Ableitungen, Matrix Algebra, AI Frameworks

## Theory

### General AI principle

Y: A label is the thing we're predicting

X: A feature is an input variable (can be a list)

x: An example is a particular instance of data (a vector of values for a feature)

A labeled example includes both feature(s) and the label.

An unlabeled example contains features but not the label.

Once we've trained our model with labeled examples, we use that model to predict the label on unlabeled examples.

General AI function: Y = wX + b

### Notes

|Doc| |
|-|-|
| ![Neuron](https://drive.google.com/uc?export=view&id=1DwdFEqoWczZEbDKUhDQhCEMDo7SZGNxr) | ![General AI function](https://drive.google.com/uc?export=view&id=14XPXX2nXUVCz81rqK5ckGQ5UzJRzR-fC) | 
| ![General AI function](https://drive.google.com/uc?export=view&id=1_XggjXSH_h0OYAoQxD_dQrs1g09yZKHu) | ![Chain rule](https://drive.google.com/uc?export=view&id=1Vp2-yuAsETZs7lAdVvE_PE0WD31uEUkb) |
| ![m derivative](https://drive.google.com/uc?export=view&id=19go_kg4CVdg-1tcDrP1LjFkeq5aX0mYq) | ![MSE / loss](https://drive.google.com/uc?export=view&id=1uTUqh4q_4N3ol0JtYLidJMlltD9HkcP6) |
| ![Activation function](https://drive.google.com/uc?export=view&id=1hTFvIL_e87F1OMQ9nM2vrFPBVbuRunDD) |

## Home work

Compute a gradient descent for a complex function and determine iteratively m and b: [Colab](https://colab.research.google.com/drive/1PMeiTzdSRWs6JGV14g-OoAHgNkGns458#scrollTo=4h5c2kvByWnu)

# 2020-10-26 2.Tensorflow und PyTorch Frameworks

## Theory

### Data loading

[Loading data in Colab](https://colab.research.google.com/drive/1nKFULrpkmL9GVfBcQFhPHljPzBt99oNm?usp=sharing)

### Tensorflow introduction

[Tensorflow quickstart for beginners](https://colab.research.google.com/drive/1DljkM90FBgpx0PwvopJ3t-Or30y5noST?usp=sharing)

[Predict fuel efficiency with a Tensorfol regressition. Dataset: MPG](https://colab.research.google.com/drive/1b3qdzQkU7cEh8jlbc7q5QurSJIV4u5tp?usp=sharing)

### Pytorch introduction

[Pytorch quickstart for beginners](https://colab.research.google.com/drive/1xgmYHV9uOi02jOy1qmhj5HsizxYQ4S9C?usp=sharing)

### Notes

|Doc| |
|-|-|
| ![Neocortex](https://drive.google.com/uc?export=view&id=1dzMwhtuwjrwrd1cdhPcDTfqMJOMUbWzg) | ![Neocortex](https://drive.google.com/uc?export=view&id=1_SbVgl-HMgXQv3-EM57a7GsVO_sRbwVr) |
| ![Activation function](https://drive.google.com/uc?export=view&id=13pAh_1bmVCckd8cK2VKRzE6vD3wWpo5C) | ![Activation function](https://drive.google.com/uc?export=view&id=1XCWJ6CLFizXwzTh6Ioz-pBn56zwQrZoN) |
| ![AI Neuronal network](https://drive.google.com/uc?export=view&id=1epAEvP662CtCbPLLPN9PxloO6fWidZGF) | ![AI Neuronal network: encoder](https://drive.google.com/uc?export=view&id=1xDGYZdWf1wBDwS0w3WGAQN9tw-m9U0Qx) |

## Homework

[Pytorch two layers NN](https://colab.research.google.com/drive/1MA56h5oApLPNRhnhiL4DRq788RRj6UD6?usp=sharing)
- Change values N, D_in, H, D
- Add a new layer/activation function with H hidden size

# 2020-11-03 3.Fundamentale Neuronale Netze: MLP und Autoencoder

## Theory

[Pytorch gradients - trainer notebook / ](https://colab.research.google.com/drive/18OZn0m_B0wX08bNSEr0SCQc4bx1pE8ww?usp=sharing)
[exercice notebook](https://colab.research.google.com/drive/1mX60kg1XVWKzU6LKFZgNDA5_GGMf9UHT?usp=sharing)

[Pytorch linear regressions - trainer notebook / ](https://colab.research.google.com/drive/1pD88GTLZL5gWPSYUy0kBYn5nRXtgmrxw?usp=sharing)
[exercice notebook](https://colab.research.google.com/drive/1_ciaDcHobrGikIfT7jj7XQjtpDV4YiD7?usp=sharing)

[Pytorch NN - trainer notebook / ](https://colab.research.google.com/drive/1ZR6taszFoUqR21yx855uNHNseSL0CSyC?usp=sharing)
[exercice notebook](https://colab.research.google.com/drive/16vH91qnn5Ha5xOa-2AYTegq48rXDx88r?usp=sharing)

[Pytorch datasets management - trainer notebook / ](https://colab.research.google.com/drive/1___ATZ-PS960ZztS39iuBDu-4oZktmAW?usp=sharing)
[exercice notebook](https://colab.research.google.com/drive/1dPCuyNFAf1Iy-mRzLu8C13ABG1DsDyLL?usp=sharing)

[Tensorflow quickstart for experts - trainer notebook / ](https://colab.research.google.com/drive/1wytjvqaDjSheG0fhmw2TWRW3RsooXS8i?usp=sharing)
[exercice notebook](https://colab.research.google.com/drive/1bQIroEU6h0w02WrRJzdR0xgFIJ0IminM?usp=sharing)

[Tensorflow autoencoder - trainer notebook / ](https://colab.research.google.com/drive/1ZuDM5AusueBZ9twN6IVVB3vZQPQFSS0F?usp=sharing)
[exercice notebook](https://colab.research.google.com/drive/1uUaSvh7-BXcQdgxF0AbCFWiWwOnPNd-E?usp=sharing)

## Homework

Migrate Iris exercice to Tensorflow: https://colab.research.google.com/drive/1gPMNk24EuvBKun5oCfV_mGrCtUA2rmoy?usp=sharing

# 2020-11-10 4.Convolutional Neuronal Networks (CNN) und Variational Autoencoders (VAE)

## Theory

[Tensorflow Stacked MLP autoencode - trainer notebook / ](https://colab.research.google.com/drive/1ZuDM5AusueBZ9twN6IVVB3vZQPQFSS0F?usp=sharing)
[exercice notebook](https://colab.research.google.com/drive/1tSKesyDf2vdsWogCkLNZ_dvjQBYGSrwy?usp=sharing)

### Notes

[Notes as PDF](https://drive.google.com/file/d/1UueiU2uD6rlEkXy6degJgurlpciWgLl8/view?usp=sharing)

## Homework

[Original autoencoder for ECG validation](https://colab.research.google.com/drive/1SrKSrs4m_D3SqKG2ffJCOQnobQE3atWg?usp=sharing)

[A Gentle Introduction to Anomaly Detection with Autoencoders](https://anomagram.fastforwardlabs.com/#/)

Goal of the exercice is to use another data set: 
[Anomaly detection for credit card with autoencoder](https://colab.research.google.com/drive/1qVjOp5hCO_vGkW8L_koCyNHXrvYT-VJj#scrollTo=SNQib7gi6nzE)

# 2020-11-17 5.Transfer-Learning Methoden und Deep ConvolutionalGenerative Adversarial Network (DCGAN)

## Theory

[Variational autoencode - trainer notebook / ](https://colab.research.google.com/drive/1H6tbiqSOdeNIuhMGxLJRoglw4sFSsEGI?usp=sharing)
[exercice notebook / ](https://colab.research.google.com/drive/17zT9lYKsnIbyxbLKwwJzIh292r1zkJKP)[video](https://www.youtube.com/watch?v=fcvYpzHmhvA)

[Dense function in TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/core.py#L1079)

[Exercice: use electrocardiogram data with previous variational autoencoder](https://colab.research.google.com/drive/1C1Ne-t7WO8kPI6e8RuOwzEkZbjxTjSIz?usp=sharing)

[Divergence de Kullback-Leibler / ](https://fr.wikipedia.org/wiki/Divergence_de_Kullback-Leibler)[video](https://www.youtube.com/watch?v=LJwtEaP2xKA)

[6 Different Ways of Implementing VAE with TensorFlow 2 and TensorFlow Probability](https://towardsdatascience.com/6-different-ways-of-implementing-vae-with-tensorflow-2-and-tensorflow-probability-9fe34a8ab981)

### Notes

[Notes as PDF](https://drive.google.com/file/d/1mUu6kiYjpzDBSWQVTHysuF_E2xbNSXCy/view?usp=sharing)

## Homework

[Howework description](https://colab.research.google.com/drive/1H6tbiqSOdeNIuhMGxLJRoglw4sFSsEGI?usp=sharing)

1. Basierend auf dem vorliegenden [Notebook](https://colab.research.google.com/drive/17zT9lYKsnIbyxbLKwwJzIh292r1zkJKP#scrollTo=eMTx8CpYOjyt), VAE mit den ECG Daten trainieren.

2. Training von VAE

3. Anomaly Detection mit VAE und Vergleich von drei Metrics - accuracy, precision und recall von einem vanilla Autoencoder aus [Aufgabe 4](https://colab.research.google.com/drive/1SrKSrs4m_D3SqKG2ffJCOQnobQE3atWg?usp=sharing#scrollTo=VvK4NRe8sVhE)

[Robust Variational Autoencoder trainer notebook / ](https://colab.research.google.com/drive/1INEafIhKPABUMbV_c967VkDcCmre6EVB?usp=sharing)
[my homework notebook](https://colab.research.google.com/drive/1C1Ne-t7WO8kPI6e8RuOwzEkZbjxTjSIz?usp=sharing)

[Very good homework](https://colab.research.google.com/drive/1DLcT-6VwrjwWBTlUsy4OkPWzmrKgBMUc?usp=sharing)

# 2020-11-24 6.Recurrent Neuronal Networks

## Theory

Convolution is a serie of scalar product.

![Convolution](https://www.researchgate.net/profile/Wooje_Lee2/publication/337106643/figure/fig2/AS:822942304858122@1573216143265/A-schematic-diagram-of-convolution-and-max-pooling-layer-In-the-convolution-layer-a.png)
 
[Notebook for understanding convolutions](https://colab.research.google.com/github/Gurubux/CognitiveClass-DL/blob/master/2_Deep_Learning_with_TensorFlow/DL_CC_2_2_CNN/2.1-Review-Understanding_Convolutions.ipynb)

[Tensorboard](https://colab.research.google.com/drive/1uBJFZfE2ep8V0mE7yLmnh7xNCKyHyjRt?usp=sharing)

[CNN basic](https://colab.research.google.com/drive/16Yq21OIy9mm2HjRT9g3gfDyZj467F0hW?usp=sharing)

[CNN advanced](https://colab.research.google.com/drive/1btoc8FpHjgTl8G51OxIUIRJw-fdRg2ZJ?usp=sharing)

[Softmax: compute probability of vector element](https://colab.research.google.com/drive/14xiHp_9aavkuMhiZgDSbyjwaE00rGKZv?usp=sharing)

### Notes

[Notes as PDF](TBD)

## Homework

[CNN homework](https://colab.research.google.com/drive/158q9rg9_fIColIym9U3fGVpVlyDeXuIT?usp=sharing)












