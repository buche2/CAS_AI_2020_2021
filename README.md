# CAS Artificial intelligence 2020-2021

CAS about artificial intelligence made at BFH in 2020 and 2021.

- [CAS Artificial intelligence 2020-2021](#cas-artificial-intelligence-2020-2021)
- [Ressources](#ressources)
  * [Data](#data)
  * [Videos](#videos)
  * [Tutorial](#tutorial)
  * [Book](#book)
- [2020-10-20 Einführung in AI Grundtechniken: Gradient Descent,partielle Ableitungen, Matrix Algebra, AI Frameworks](#2020-10-20-einf-hrung-in-ai-grundtechniken--gradient-descent-partielle-ableitungen--matrix-algebra--ai-frameworks)
  * [Theory](#theory)
    + [General AI principle](#general-ai-principle)
    + [Notes](#notes)
  * [Home work](#home-work)
- [2020-10-26 Tensorflow und PyTorch Frameworks](#2020-10-26-tensorflow-und-pytorch-frameworks)
  * [Theory](#theory-1)
    + [Data loading](#data-loading)
    + [Tensorflow introduction](#tensorflow-introduction)
    + [Pytorch introduction](#pytorch-introduction)
    + [Notes](#notes-1)
  * [Homework](#homework)
- [Fundamentale Neuronale Netze: MLP und Autoencoder](#fundamentale-neuronale-netze--mlp-und-autoencoder)
  * [Theory](#theory-2)
  * [Homework](#homework-1)
- [2020-11-10 Convolutional Neuronal Networks (CNN) und Variational Autoencoders (VAE)](#2020-11-10-convolutional-neuronal-networks--cnn--und-variational-autoencoders--vae-)
  * [Theory](#theory-3)
    + [Notes](#notes-2)
  * [Homework](#homework-2)
  
# Ressources

## Data

[Machine learning data repository](https://archive.ics.uci.edu/ml/datasets.php)
[IRIS](https://gist.github.com/curran/a08a1080b88344b0c8a7)

## Videos

## Tutorial

[Colab](https://colab.research.google.com/notebooks/intro.ipynb)

## Book

1. Machine Learning, Tom Mitchell, McGraw Hill, 1997 http://www.cs.cmu.edu/~tom/mlbook.html
2.	Deeplearning, Ian Goodfellow https://www.deeplearningbook.org/contents/mlp.html


# 2020-10-20 Einführung in AI Grundtechniken: Gradient Descent,partielle Ableitungen, Matrix Algebra, AI Frameworks

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

![Neuron](https://drive.google.com/uc?export=view&id=1DwdFEqoWczZEbDKUhDQhCEMDo7SZGNxr)
![General AI function](https://drive.google.com/uc?export=view&id=14XPXX2nXUVCz81rqK5ckGQ5UzJRzR-fC)
![General AI function](https://drive.google.com/uc?export=view&id=1_XggjXSH_h0OYAoQxD_dQrs1g09yZKHu)
![Chain rule](https://drive.google.com/uc?export=view&id=1Vp2-yuAsETZs7lAdVvE_PE0WD31uEUkb)
![m derivative](https://drive.google.com/uc?export=view&id=19go_kg4CVdg-1tcDrP1LjFkeq5aX0mYq)
![MSE / loss](https://drive.google.com/uc?export=view&id=1uTUqh4q_4N3ol0JtYLidJMlltD9HkcP6)
![Activation function](https://drive.google.com/uc?export=view&id=1hTFvIL_e87F1OMQ9nM2vrFPBVbuRunDD)

## Home work

Compute a gradient descent for a complex function and determine iteratively m and b: [Colab](https://colab.research.google.com/drive/1PMeiTzdSRWs6JGV14g-OoAHgNkGns458#scrollTo=4h5c2kvByWnu)

# 2020-10-26 Tensorflow und PyTorch Frameworks

## Theory

### Data loading

[Loading data in Colab](https://colab.research.google.com/drive/1nKFULrpkmL9GVfBcQFhPHljPzBt99oNm?usp=sharing)

### Tensorflow introduction

[Tensorflow quickstart for beginners](https://colab.research.google.com/drive/1DljkM90FBgpx0PwvopJ3t-Or30y5noST?usp=sharing)

[Predict fuel efficiency with a Tensorfol regressition. Dataset: MPG](https://colab.research.google.com/drive/1b3qdzQkU7cEh8jlbc7q5QurSJIV4u5tp?usp=sharing)

### Pytorch introduction

[Pytorch quickstart for beginners](https://colab.research.google.com/drive/1xgmYHV9uOi02jOy1qmhj5HsizxYQ4S9C?usp=sharing)

### Notes

![Neocortex](https://drive.google.com/uc?export=view&id=1dzMwhtuwjrwrd1cdhPcDTfqMJOMUbWzg)
![Neocortex](https://drive.google.com/uc?export=view&id=1_SbVgl-HMgXQv3-EM57a7GsVO_sRbwVr)
![Activation function](https://drive.google.com/uc?export=view&id=13pAh_1bmVCckd8cK2VKRzE6vD3wWpo5C)
![Activation function](https://drive.google.com/uc?export=view&id=1XCWJ6CLFizXwzTh6Ioz-pBn56zwQrZoN)
![AI Neuronal network](https://drive.google.com/uc?export=view&id=1epAEvP662CtCbPLLPN9PxloO6fWidZGF)
![AI Neuronal network: encoder](https://drive.google.com/uc?export=view&id=1xDGYZdWf1wBDwS0w3WGAQN9tw-m9U0Qx)

## Homework

[Pytorch two layers NN](https://colab.research.google.com/drive/1MA56h5oApLPNRhnhiL4DRq788RRj6UD6?usp=sharing)
- Change values N, D_in, H, D
- Add a new layer/activation function with H hidden size

# Fundamentale Neuronale Netze: MLP und Autoencoder

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

# 2020-11-10 Convolutional Neuronal Networks (CNN) und Variational Autoencoders (VAE)

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










