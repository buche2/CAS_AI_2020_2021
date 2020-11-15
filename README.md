# CAS Artificial intelligence 2020-2021

CAS about artificial intelligence made at BFH in 2020 and 2021.

- [CAS Artificial intelligence 2020-2021](#cas-artificial-intelligence-2020-2021)
- [2020-10-20 Einführung in AI Grundtechniken: Gradient Descent,partielle Ableitungen, Matrix Algebra, AI Frameworks](#2020-10-20-einf-hrung-in-ai-grundtechniken--gradient-descent-partielle-ableitungen--matrix-algebra--ai-frameworks)
  * [Theory](#theory)
    + [General AI principle](#general-ai-principle)
    + [Notes](#notes)
  * [Home work](#home-work)

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
