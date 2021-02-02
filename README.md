# CAS Artificial intelligence 2020-2021

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
- [2020-11-10 4.Anomaly Detection mit Autoencoder](#2020-11-10-4anomaly-detection-mit-autoencoder)
  * [Theory](#theory-3)
    + [Notes](#notes-2)
  * [Homework](#homework-2)
- [2020-11-17 5.Variational Autoencoder](#2020-11-17-5variational-autoencoder)
  * [Theory](#theory-4)
    + [Notes](#notes-3)
  * [Homework](#homework-3)
- [2020-11-24 6.CNN](#2020-11-24-6cnn)
  * [Theory](#theory-5)
    + [Notes](#notes-4)
  * [Homework](#homework-4)
- [2020-12-01 7.CNN - Transfer Learning](#2020-12-01-7cnn---transfer-learning)
  * [Theory](#theory-6)
    + [Notes](#notes-5)
  * [Homework](#homework-5)
- [2020-12-08 8.GAN](#2020-12-08-8gan)
  * [Theory](#theory-7)
    + [Notes](#notes-6)
  * [Homework](#homework-6)
- [2020-12-15 9.RNN](#2020-12-15-9rnn)
  * [Theory](#theory-8)
    + [Notes](#notes-7)
  * [Homework](#homework-7)
- [2021-01-05 10.RL Grundlagen](#2021-01-05-10rl-grundlagen)
  * [Theory](#theory-9)
    + [Setup](#setup)
    + [Categories of machine learning](#categories-of-machine-learning)
    + [Use cases of machine learning](#use-cases-of-machine-learning)
    + [Categories of reinforcement learning](#categories-of-reinforcement-learning)
    + [Notes](#notes-8)
  * [Homework](#homework-8)
- [2021-01-12 11.RL Cross Entropy](#2021-01-12-11rl-cross-entropy)
  * [Theory](#theory-10)
    + [Q-Learning intro](#q-learning-intro)
    + [Notes](#notes-9)
  * [Homework](#homework-9)
- [2021-01-19 12.RL Value iteration](#2021-01-19-12rl-value-iteration)
  * [Theory](#theory-11)
    + [Notes](#notes-10)
  * [Homework](#homework-10)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


CAS about artificial intelligence made at BFH in 2020 and 2021.
  
# Ressources

## Interative course script

[Script as notebook](https://colab.research.google.com/drive/10vkq0wYDpZJkxRWzlxwcRFY1q0tOooCO?usp=sharing)

## Data

[Machine learning data repository](https://archive.ics.uci.edu/ml/datasets.php)

[IRIS](https://gist.github.com/curran/a08a1080b88344b0c8a7)

[EPFL](https://www.epfl.ch/labs/mmspg/downloads/)

## Videos

[Courses video recording](https://teams.microsoft.com/_#/school/files/G%C3%A9n%C3%A9ral?threadId=19%3Ab816fda1d3d7401faeffba576f427581%40thread.tacv2&ctx=channel&context=Recordings&rootfolder=%252Fsites%252Fbfh-bfh-2020-hscasartificalintelligence%252FFreigegebene%2520Dokumente%252FGeneral%252FRecordings)

## Tutorial

[Colab](https://colab.research.google.com/notebooks/intro.ipynb)

[Deep Learning (CAS machine intelligence, 2019)]()https://tensorchiefs.github.io/dl_course_2018/

## Book

1. Machine Learning, Tom Mitchell, McGraw Hill, 1997 http://www.cs.cmu.edu/~tom/mlbook.html
2. Deeplearning, Ian Goodfellow https://www.deeplearningbook.org/contents/mlp.html

## Tricks

Update python packages

```
pip list --outdated
pip3 list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip3 install -U
```

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

# 2020-11-10 4.Anomaly Detection mit Autoencoder

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

# 2020-11-17 5.Variational Autoencoder

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

# 2020-11-24 6.CNN

## Theory

Convolution is a serie of scalar product.

![Convolution](https://www.researchgate.net/profile/Wooje_Lee2/publication/337106643/figure/fig2/AS:822942304858122@1573216143265/A-schematic-diagram-of-convolution-and-max-pooling-layer-In-the-convolution-layer-a.png)
 
[Notebook for understanding convolutions](https://colab.research.google.com/github/Gurubux/CognitiveClass-DL/blob/master/2_Deep_Learning_with_TensorFlow/DL_CC_2_2_CNN/2.1-Review-Understanding_Convolutions.ipynb)

[Tensorboard / ](https://colab.research.google.com/drive/1uBJFZfE2ep8V0mE7yLmnh7xNCKyHyjRt?usp=sharing)[My tensorboard](https://colab.research.google.com/drive/1oD5jpRKbSGlOdFOvLcBz5aNEF5mjhvlp)

[CNN basic](https://colab.research.google.com/drive/16Yq21OIy9mm2HjRT9g3gfDyZj467F0hW?usp=sharing)

[CNN advanced](https://colab.research.google.com/drive/1btoc8FpHjgTl8G51OxIUIRJw-fdRg2ZJ?usp=sharing)

[Softmax: compute probability of vector element](https://colab.research.google.com/drive/14xiHp_9aavkuMhiZgDSbyjwaE00rGKZv?usp=sharing)

[CIFAR10 explanation](https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c)

### Notes

[Notes as PDF](https://drive.google.com/file/d/1tVBzKR80kTm7ijtlocn_e9_4CGRBGnxb/view?usp=sharing)

## Homework

[CNN homework #1](https://colab.research.google.com/drive/158q9rg9_fIColIym9U3fGVpVlyDeXuIT?usp=sharing)

[CNN homework with preprocessing -> finally cancelled](https://colab.research.google.com/drive/1qfJRIbBeVg8QImTZPauraDXsKgL_tvpn?usp=sharing)

[CNN Advanced homework #2 - With CIFAR 10](https://colab.research.google.com/drive/1wIRSExQCUM4uV0TDbJWp4v7O1ACXdJPZ?usp=sharing)

[Conv Variational Autoencoder homework -  With CIFAR 10 -> finally cancelled](https://colab.research.google.com/drive/1li43rVySv0XUc0hwBiY6Zu8tn-9hYw2O)

[Robist Conv Variational Autoencoder homework - with MNIST - Trainer notebook / ](https://colab.research.google.com/drive/1t6v9ILqZVuqP5Fv01wNKcSJd5ou7Vfvp?usp=sharing)[Homework #3](https://colab.research.google.com/drive/1GHYzU7i11U4HHEa8OW4hYHnJV8QQE2OY#scrollTo=2M7LmLtGEMQJ)

# 2020-12-01 7.CNN - Transfer Learning

## Theory

[VGG16 – Convolutional Network for Classification and Detection](https://neurohive.io/en/popular-networks/vgg16/)

### Notes

[Notes as PDF](https://drive.google.com/file/d/1RoR5Y9wZTAqHTYG6zEUlM997wpDyLOfS/view?usp=sharing)

## Homework

[07 TF2.0 Transfer Learning - Special - Trainer notebook / ](https://colab.research.google.com/drive/1Id41sXdKCB9llQbAlWmZW1bbqDTVKcSb?usp=sharing)[My notebook](https://colab.research.google.com/drive/1yXLNlCs6-AKYsHrydXuFWLWm9fG3liAZ?usp=sharing)

[07 TF2.0 Transfer Learning with Data Augmentation - Classic - Trainer notebook / ](https://colab.research.google.com/drive/1Lgy4fBCBFvxmXxz4YaTdwCjW3nUGz3q6#scrollTo=d9W7Yjjl3XUO)[My notebook](https://colab.research.google.com/drive/1ThNobRaEDr4O13mxdxupMeXDMFg5pwmp?usp=sharing)

[07 TF2.0 Transfer Learning - Special CIFAR / ](https://colab.research.google.com/drive/1qmELcVBdKGKRQcujJfh8NfcB-CmfQSp_?usp=sharing)[Trainer notebook](https://colab.research.google.com/drive/1-RjVGy6nnjSGz7ghqz3sf1hrLqd-0VcV?usp=sharing)

# 2020-12-08 8.GAN

## Theory

[Generative adversarial network](https://en.wikipedia.org/wiki/Generative_adversarial_network)

[A Friendly Introduction to Generative Adversarial Networks (GANs) - Video](https://www.youtube.com/watch?v=8L11aMN5KY8)

[A guide to convolution arithmetic for deep learning](https://www.arxiv-vanity.com/papers/1603.07285/)

[Conv2D explanation](https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/)

### Notes

[Notes as PDF](https://drive.google.com/file/d/1O6htvSGNUjagp6tHac5yCVPjIi4i4erD/view?usp=sharing)

## Homework

[Dense GAN - My notebook](https://colab.research.google.com/drive/1AnOqf2PcqABZ0e_KNOBR-2mNy8cJR2J4?usp=sharing)

[DCGAN - CIFAR - My notebook](https://colab.research.google.com/drive/1mjT37cgtZlTEoSkfsy1mpzC6wXTqNtyj?usp=sharing)

# 2020-12-15 9.RNN

## Theory

[Cheatsheet recurrent neural networks](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)

[RNN for calligraphy / ](https://www.calligrapher.ai/)[Source code](https://github.com/sjvasquez/handwriting-synthesis)

[RNN Video - ](https://www.youtube.com/watch?v=EL439RMv3Xc)[LSTM Video](https://www.youtube.com/watch?v=3xgYxrNyE54)

[LTSM illustrated](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)

[RNN with pytorch - Trainer notebook](https://colab.research.google.com/drive/1rM-acG1lXw68_-mFcx275SlvO2asdAWG?usp=sharing)

[LSTM with pytorch - Trainer notebook](https://colab.research.google.com/drive/1z3sjL0D7klwbRSlIz4Grdpy9rT0pvrHi?usp=sharing)

[Stock return - Trainer notebook](https://colab.research.google.com/drive/1bW9xhVVoKJg28s_Go7EhwJTmMrnn2yqv?usp=sharing)

### Notes

[Notes as PDF](https://drive.google.com/file/d/1NQSSe1fOoLMB2kSUtgpJiux38TWW7FV1/view?usp=sharing)

## Homework

[RNN with pytorch - Homework](https://colab.research.google.com/drive/1y44MXu-ynNyE69hFbwt6ZZtwy06xIVd6#scrollTo=rWl7Nb3W0kgH)

[Stock prediction by Boris Banushev](https://github.com/borisbanushev/stockpredictionai)

# 2021-01-05 10.RL Grundlagen

## Theory

[Reinforcement Learning algorithms — an intuitive overview](https://medium.com/@SmartLabAI/reinforcement-learning-algorithms-an-intuitive-overview-904e2dff5bbc)

[Lilian Weng blog](https://lilianweng.github.io/lil-log/)

[OpenAI Gym - CartPole](https://gym.openai.com/envs/CartPole-v1/)

### Setup

We will work with Anaconda and PyCharm

A new python environment has been installed with Anaconda: casaai2020

Install gym and pygame:

```
source activate casaai2020
pip install gym
pip install pygame
```

### Categories of machine learning

![Categories Machine Learning](https://drive.google.com/uc?export=view&id=1_Jnm9qZjWripNE8eWqAHJ4dFuB0VQ47h)

### Use cases of machine learning

![Use cases of machine learning](https://drive.google.com/uc?export=view&id=1LftDhSV7G_ZCn71LZUIk23Oj3MdpntRh)

### Categories of reinforcement learning

![Categories Reinforcement Learning](https://drive.google.com/uc?export=view&id=172oahhZToFNiIgV-1VM6tRkw1p-sCwu5)

### Notes

[Notes as PDF](https://drive.google.com/file/d/1YNKJyifMraLWDXggRJbVrlQNRgqvedEE/view?usp=sharing)

## Homework

Take the [trainer notebook ](https://colab.research.google.com/drive/1XLYVhsT3u5cPPy3Jhc4O-54Sv7ONzRKd?usp=sharing) and make it working on pycharm -> export to .py. In order to work with PyCharm, several components had to be installed with pip: opencv, opencv-python, torchvision, cmake and atari-py

[My notebook](https://colab.research.google.com/drive/1Fe0NyGIob4jXmUYnoduGHMLG-84Jg2vh?usp=sharing)

Tip: load tensorboard by starting it in the PyCharm console:

```
tensorboard --logdir=runs
```

# 2021-01-12 11.RL Cross Entropy

## Theory

[GYM environments - Trainer notebook / ](https://colab.research.google.com/drive/1ODQ46vODyLsumYzK6_aBC1a-uO4p693W?usp=sharing)[My notebook](https://colab.research.google.com/drive/1IE9q8tpo4A96-O5eVhdPQ0m4YiF_Dv1M)

[CrossEntropy example - Cart Pole agent and mountain car agent](https://github.com/cedricmoullet/CAS_AI_2020_2021/tree/main/20210112_CrossEntropy)

[GYM environments](https://gym.openai.com/envs/#classic_control)

Tip: python environment library version:

```
source activate casaai2020
pip list
```

### Q-Learning intro

[Wikipedia for Q-Learning](https://fr.wikipedia.org/wiki/Q-learning)

[Carneggie Mellon course](https://www.cs.cmu.edu/~mgormley/courses/10601-s17/slides/lecture26-ri.pdf) 

[Q-Learning video](https://www.youtube.com/watch?v=a0bVIyIJ074)

[Gamma value](https://colab.research.google.com/drive/1a2-SwkvyPjlx7PiJKUN485JOAGAm_g6Y?usp=sharing)

[Policy Gradient with gym-MiniGrid](https://goodboychan.github.io/chans_jupyter/python/pytorch/reinforcement_learning/2020/08/06/03-Policy-Gradient-With-Gym-MiniGrid.html)

### Notes

[Notes as PDF](https://drive.google.com/file/d/1b42gu2H8sxv9jIg3nBHja62VD1aj9m1c/view?usp=sharing)

## Homework

[Moodle description / ](https://moodle.bfh.ch/mod/assign/view.php?id=1258398)[mini-grid code](https://github.com/cedricmoullet/CAS_AI_2020_2021/blob/main/20210112_CrossEntropy/miniGrid.py)

# 2021-01-19 12.RL Value iteration

## Theory

[Keras RL2 / ](https://github.com/wau/keras-rl2)[trainer notebook](https://colab.research.google.com/drive/1KzTM0Wxb97Zqv0Tqm4V-wrs1BtKni2nr?usp=sharing)

Comments about value [iteration example](https://github.com/cedricmoullet/CAS_AI_2020_2021/blob/main/20210119_ValueIteration/agent4.py):

- Observations: the possible agent positions in the 4x4 grids. 16 possibilities.
- Actions: the possible actions made by the agent: up, down, left and right
- Rewards: the possible rewards, depending of the currrent state, the next state and the action: 16 * 16 * 4
- Transitions: the possible paths, depending of the currrent state, the next state and the action:: 16 * 16 * 4
- Values: the Q-values, depending og the state and the action: 16 * 4

[GridWorld env](https://github.com/addy1997/Gridworld)

### Notes

[Notes as PDF](https://drive.google.com/file/d/1JefLVsqWmIKaQTQyCq-ftzuiTqXwGA5x/view?usp=sharing)

## Homework

[RL2 notebook](https://colab.research.google.com/drive/1H7pkvLjlBXvdtS5JVz-wQLFpUZiFQRlR?usp=sharing)

[Finance processing with AI](https://github.com/firmai/financial-machine-learning)

[PyTorch uncertainty estimation - trainer notebook / ](https://colab.research.google.com/drive/1lhZu_AKW83PVLu76VvzRranPEL_I0M45?usp=sharing)[my notebook / ](https://colab.research.google.com/drive/1fEGWQjmR5iK7RMiBQtQdsXvbcNjPrYWN?usp=sharing)[video](https://youtu.be/mM0QyaAEu9Q)

# 2021-01-26 13. Tabular Q-Learning

## Theory

[Tabular Q-Learning source code](https://github.com/cedricmoullet/CAS_AI_2020_2021/tree/main/20210126_Tabular_QLearning)

[Holt winters forecasting](https://towardsdatascience.com/holt-winters-exponential-smoothing-d703072c0572)

[Theta model](https://www.statsmodels.org/stable/examples/notebooks/generated/theta-model.html#:~:text=The%20Theta%20model%20of%20Assimakopoulos,to%20produce%20the%20final%20forecast.)

### Notes

## Homework

[Avocado exercise - trainer notebook / ](https://colab.research.google.com/drive/15kA3uh34pwyHERM1xa3aOl8fahpOV_MW?usp=sharing)[my notebook](https://colab.research.google.com/drive/1DqSa4fHoay2vQzQablJ0G-f-hngGnlue?usp=sharing)

# 2021-02-02 14. Deep Q Learning

## Theory

[Advanced Forecasting with LSTM and verification of results using Multi-Step Forecasting Devoir - trainer notebook / ](https://colab.research.google.com/drive/1CcpEtz1mdcUmYYAU5uTuJTPbfLFD0yz-?usp=sharing)[my notebook](https://colab.research.google.com/drive/1xGGbUc7pfu6T4EGF-L6LlrtI4Im7wg0_?usp=sharing)

[DQN Video lesson](https://www.youtube.com/watch?v=OYhFoMySoVs)

[DQN Cart pole](https://github.com/cedricmoullet/CAS_AI_2020_2021/tree/main/20210202_DQN)

### Notes

## Homework




















