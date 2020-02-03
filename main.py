#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:31:41 2020

@author: dycon
"""

__author__ = "Borjan Geshkovski"
__version__ = "0.1"

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random as rand
import numpy as np
import copy
from sklearn import datasets
from sklearn.datasets import make_classification
import neural_net as nn
import seaborn as sns
sns.set(style="darkgrid")
#sns.set(style="whitegrid")

# Colormap options: plt.cm.jet, plt.cm.coolwarm, plt.cm.viridis

def rand_pts(n, c, r):
    """
    Returns n random points in a disk of radius r, centered at c = (x, y)
    """
    x, y = c
    points = []
    colors = []
    for i in range(n):
        theta = 2*math.pi*rand.random()
        s = r*rand.random()
        points.append((x+s*math.cos(theta), y+s*math.sin(theta)))
        k = rand.randint(0, 1)
        colors.append(k)
    return points, colors 

def generate_points(n, centers, labels, amplitude):
    """
    Write down. It's for generating the chess points.
    """
    x1_noisy = []
    x2_noisy = []
    y = []
    for center, c in zip(centers, labels):
        x1, x2 = center
        noise_x1 = np.random.rand(n)
        noise_x2 = np.random.rand(n)
        x1_noisy += (noise_x1 + x1).tolist()
        x2_noisy += (noise_x2 + x2).tolist()
        y += [c] * n
    
    X = list(zip(x1_noisy, x2_noisy))
    X = np.array(X)
    y = np.array(y)
    y = np.expand_dims(y, 1)
    
    X = X.T
    y = y.T
    return X, y 

def plot_history(history):
    """
    We plot some graphs of the loss functional over iterations.
    """
    n = history['epochs']
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    n = 4000
    plt.plot(range(history['epochs'])[:n], history['train_loss'][:n], label='train_loss')
    plt.plot(range(history['epochs'])[:n], history['test_loss'][:n], label='value of functional')
    plt.title('value of the functional')
    plt.grid(1)
    plt.xlabel('iterations')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(history['epochs'])[:n], history['train_acc'][:n], label='train_acc')
    plt.plot(range(history['epochs'])[:n], history['test_acc'][:n], label='test_acc')
    plt.title('train & test accuracy')
    plt.grid(1)
    plt.xlabel('iterations')
    plt.legend()

def lambdas(A, b, x):
    return np.dot(A, x) + b    

def sigmas(A, b, x):
    return nn.sigmoid(np.dot(A, x)+b)

def simulate(samples, features=2, data_="blobs", architecture=[2, 2, 1]):
    """
    """
    # Perhaps encode a forcing of an exception
    # in case the data_ is not in a desired list
    # of predefined strings
    # By default, cluster_std=1.0 in blobs and random_state = 2
    
    datasets_ = {'blobs': datasets.make_blobs(n_samples=samples, n_features=features, centers=2, cluster_std=5, random_state=3), 
                 'spirals': datasets.make_moons(n_samples=samples, noise=0.2),
                 'chess': generate_points(samples, [(0, 0), (0, 1), (1, 0), (1, 1)], [0, 1, 1, 0], 0.005)}
    data = datasets_[data_]
    
    if data_ == "chess":
        X, y = data
    else:
        X = data[0].T
        y = np.expand_dims(data[1], 1).T
    
#    if features == 1:
#        aux = np.array( [ [0 for i in range(len(X[0]))]  ]  )
#        X = np.concatenate((X, aux))
        
    neural_net = nn.NeuralNetwork(architecture, seed=0)
    history = neural_net.train(X=X, y=y, batch_size=16, epochs=5000, learning_rate=0.3, 
                               print_every=1000, validation_split=0.2, tqdm_=False,
                               plot_every=2500)
    weights, biases = history['weights'], history['biases']
     
    # Initialize the neural network scheme
    z0 = X
    if np.shape(z0)[0] == 1:
        red = list()
        blue = list()
   
        for i, x in enumerate(z0[0]):
            if y[0][i] == 0:
                red.append(x)
            else:
                blue.append(x)
                
        plt.figure()
        plt.plot(blue, len(blue)*[0], 'o', c='r')
        
        plt.plot(red, len(red)*[0], 'o', c='b')
        plt.title('The data points', fontdict={'fontsize':12})
        plt.show()
    else:
         #####We plot the data points
        plt.figure()
        plt.scatter(z0.T[:, 0], z0.T[:, 1], c=y.T.reshape(-1), cmap = plt.cm.coolwarm, alpha=0.55)
        plt.xlabel(r'$x_1$ coordinate', fontdict = {'fontsize' : 12})
        plt.ylabel(r'$x_2$ coordinate', fontdict = {'fontsize' : 12})
        plt.title(r'The {} data points'.format(samples), fontdict = {'fontsize' : 18})
    
    # Faut que je repare ça 
    #plt.savefig('{}/{}/z0.png'.format(data_, samples), dpi=450)
    
    lambda_ = list()
    layers = list()
    layers.append(z0)
    
    if len(architecture)>2:
        for k in range(1, len(architecture)):
            # For the moment, the code here is not pretty
            # and quite repetitive. I would like to declutter it
            # quite a bit.
            
            # We can only visualize up to 3d
            if architecture[k]==1: 
                print(np.shape(layers[k-1]))
                #_ = copy.deepcopy( [  lambdas(weights[k-1], biases[k-1],  ]  )
                _ = copy.deepcopy(  [ lambdas(weights[k-1], biases[k-1], 
                                      [[layers[k-1][0][i]], [layers[k-1][1][i]]] ) 
                                      for i in range(samples)]  )

                lambda_.append(np.array([l[0][0] for l in _]))
#                
#                # Then we store the activation of these linear transformations
                __ = copy.deepcopy(  [ sigmas(weights[k-1], biases[k-1], 
                                      [[layers[k-1][0][i]], [layers[k-1][1][i]]] ) 
                                      for i in range(samples)]  )
                layers.append(np.array([l[0][0] for l in __]))
            
            elif architecture[k]==2:
                # We first store the linear transformations to be plotted
                if len(layers[k-1])==1:
                    _ = copy.deepcopy([ lambdas(weights[k-1], biases[k-1], 
                                     [layers[k-1][0][i]]) 
                                      for i in range(samples)])
                    lambda_.append(np.array( [[l[0][0] for l in _], [l[1][0] for l in _]] ))
                
                # Then we store the activation of these linear transformations
                    __ = copy.deepcopy([ sigmas(weights[k-1], biases[k-1], 
                                     [layers[k-1][0][i]]) 
                                      for i in range(samples)])
                    layers.append(np.array( [[s[0][0] for s in __], [s[1][0] for s in __]] ))
                
                else:
                    _ = copy.deepcopy([ lambdas(weights[k-1], biases[k-1], 
                                     [ [layers[k-1][0][i]], [layers[k-1][1][i]] ]) 
                                      for i in range(samples)])
                    lambda_.append(np.array( [[l[0][0] for l in _], [l[1][0] for l in _]] ))
                
                    # Then we store the activation of these linear transformations
                    __ = copy.deepcopy([ sigmas(weights[k-1], biases[k-1], 
                                     [ [layers[k-1][0][i]], [layers[k-1][1][i]] ]) 
                                      for i in range(samples)])
                    layers.append(np.array( [[s[0][0] for s in __], [s[1][0] for s in __]] ))
            
            elif architecture[k]==3:
                _ = copy.deepcopy( [ lambdas(weights[k-1], biases[k-1], 
                                     [ [layers[k-1][0][i]], [layers[k-1][1][i]],  [layers[k-1][2][i]] ]) 
                                      for i in range(samples)] )
                lambda_.append( np.array( [[l[0][0] for l in _], [l[1][0] for l in _], [l[2][0] for l in _]] ) )
                
                # Then we store the activation of these linear transformations
                __ = copy.deepcopy([ sigmas(weights[k-1], biases[k-1], 
                                     [ [layers[k-1][0][i]], [layers[k-1][1][i]], [layers[k-1][2][i]] ]) 
                                      for i in range(samples)])
                layers.append(np.array( [[s[0][0] for s in __], [s[1][0] for s in __], [s[2][0] for s in __]] ))
            else:
                pass
    else:
        # For the moment, we do nothing in this case
        pass
    
    # Plotting the linear transitions
    for i, lbd in enumerate(lambda_):
        if len(lbd) == 2:
            plt.figure()
            plt.scatter(lbd.T[:, 0], lbd.T[:, 1], c=y.T.reshape(-1), cmap=plt.cm.coolwarm, alpha=0.55)
            plt.xlabel(r'$x_1$ coordinate', fontdict = {'fontsize' : 12})
            plt.ylabel(r'$x_2$ coordinate', fontdict = {'fontsize' : 12})
            plt.title(r'Transformed data: $\Lambda_{} = A^{} x + b^{}$'.format(i, i, i), fontdict = {'fontsize' : 18})
        else:
            red_ = list()
            blue_ = list()
            for j, ix in enumerate(lbd):
                if y[0][j] == 0:
                    red_.append(ix)
                else:
                    blue_.append(ix)
            plt.figure()
            plt.plot(blue_, len(blue_)*[0], 'o', c='r')
        
            plt.plot(red_, len(red_)*[0], 'o', c='b')
            plt.xlabel(r'$x_1$ coordinate', fontdict = {'fontsize' : 12})
            plt.ylabel(r'$x_2$ coordinate', fontdict = {'fontsize' : 12})
            plt.title(r'Transformed data: $\Lambda_{} = A^{} x + b^{}$'.format(i, i, i), fontdict = {'fontsize' : 18})
        
    
    
    print(weights[0])
    for j, sig in enumerate(layers):
        if j==0:
            # Don't want to show input layer as we already plot it
            pass
        else:
            if len(sig) == 2:
                plt.figure()
                plt.scatter(sig.T[:, 0], sig.T[:, 1], c=y.T.reshape(-1), cmap=plt.cm.coolwarm, alpha=0.55)
                plt.xlabel(r'$x_1$ coordinate', fontdict = {'fontsize' : 12})
                plt.ylabel(r'$x_2$ coordinate', fontdict = {'fontsize' : 12})
                plt.title(r'{}st hidden layer: $z^{} = \sigma(A^{} z^{} + b^{})$'.format(j, j, j-1, j-1, j-1), 
                      fontdict = {'fontsize' : 18})
            else:
                _red = list()
                _blue = list()
                for r, ix in enumerate(sig):
                    if y[0][r] == 0:
                        _red.append(ix)
                    else:
                        _blue.append(ix)
                plt.figure()
                plt.plot(_blue, len(_blue)*[0], 'o', c='r')
        
                plt.plot(_red, len(_red)*[0], 'o', c='b')
                plt.xlabel(r'$x_1$ coordinate', fontdict = {'fontsize' : 12})
                plt.ylabel(r'$x_2$ coordinate', fontdict = {'fontsize' : 12})
                if j == len(layers)-1:
                    plt.title(r'Output of neural network: $z^{} = \sigma(A^{} z^{} + b^{})$'.format(j, j, j-1, j-1, j-1), 
                              fontdict = {'fontsize' : 18})
                else:
                    plt.title(r'{}st hidden layer: $z^{} = \sigma(A^{} z^{} + b^{})$'.format(j, j, j-1, j-1, j-1), 
                              fontdict = {'fontsize' : 18})
            

if __name__ == "__main__":
    #simulate(150, data_='blobs')
    simulate(150, features=1, data_='blobs', architecture=[1, 2, 1])
    
#    # Just plot the activation functions 
#    x1 = np.linspace(-10, 10, 200)
#    x2 = np.linspace(-3, 3, 100)
#    y1 = list()
#    y2 = list()
#    for z in x1:
#        y1.append(nn.sigmoid(z))
#    for z in x2:
#        y2.append(nn.relu(z))
#    
#    for i in range(2):    
#        plt.figure()
#        plt.grid(True)
#        #plt.rc('grid', linestyle="-.", color='r')
#        
#        if i==0:
#            plt.plot(x1, y1, color='blue', linewidth=3, alpha=0.55, linestyle='-', label=r'$\sigma(x) = (1+e^{-x})^{-1}$')
#            plt.title(r'The sigmoid activation function', fontdict={'fontsize': 12})
#            plt.xlim(-10, 10)
#            plt.ylim(-0.01, 1.01)
#        else:
#            plt.plot(x2, y2, color='blue', linewidth=3, alpha=0.55, linestyle='-', label=r'$\sigma(x) = \max(x, 0)$')            
#            plt.title(r'The ReLU activation function', fontdict={'fontsize': 12})
#            plt.xlim(-3, 3)
#        plt.xlabel(r'x')
#        plt.ylabel(r'$\sigma(x)$')
#        plt.legend(loc=2, prop={'size': 14.5})
    
#    # Some plotting tests in 1d
#    data = datasets.make_blobs(n_samples=50, n_features=1, centers=3, random_state=2)
#
#    X = data[0].T
#    y = np.expand_dims(data[1], 1).T
#    
#    red = list()
#    blue = list()
#   
#    for i, x in enumerate(X[0]):
#        if y[0][i] == 0:
#            red.append(x)
#        else:
#            blue.append(x)
#            
#    plt.plot(blue, len(blue)*[0], 'x')
#    
#    plt.plot(red, len(red)*[0], 'r+')
#    plt.show()
#    
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#
#    # Prepare arrays x, y, z
#    x = np.linspace(-10, 10, 100)
#    x1 = nn.sigmoid(x)
#    x2 = nn.sigmoid(x)
#    
#    #plt.plot(x1, x2)
#    ax.plot(x1, x2, x, label='the parametric curve')
#    ax.legend()
#
#    plt.show()
    
    
