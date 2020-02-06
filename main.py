#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:31:41 2020

@author: dycon
"""

__author__ = "Borjan Geshkovski"
__version__ = "0.2"

from matplotlib import pyplot as plt
from matplotlib import rc
rc("text", usetex = True)
font = {'size' : 30}
rc('font', **font)
from mpl_toolkits.mplot3d import Axes3D
import math
import random as rand
import numpy as np
import copy
from sklearn import datasets
import neural_net as nn
import seaborn as sns
from pathlib import Path
#sns.set(style="darkgrid")
sns.set(style="whitegrid")


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

def generate_points_1d(nb, centers=[-1, 0, 1], labels=[1, 0, 1]):
    """
    We generate points on 1d    
    """
    x_noisy = list()
    y = []
    n = nb//len(centers)
    for center, c in zip(centers, labels):
        x = center
        noise_x = np.random.rand(n)
        x_noisy += (noise_x + x).tolist()
        y += [c] * n
    
    X = list(zip(x_noisy))
    X = np.array(X)
    y = np.array(y)
    y = np.expand_dims(y, 1)
    
    X = X.T
    y = y.T
    return X, y 

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

def simulate(samples, features=2, data_="blobs", architecture=[2, 2, 1]):
    """
    """
    # Perhaps encode a forcing of an exception
    # in case the data_ is not in a desired list
    # of predefined strings
    # By default, cluster_std = 1.0 in blobs and random_state = 2
    
    # Just to link the code variables with my mathematical notations:
    # Lplus2 = len(architecture)
    
    datasets_ = {'blobs': datasets.make_blobs(n_samples=samples, n_features=features, centers=2, cluster_std=1, random_state=2), 
                 'spirals': datasets.make_moons(n_samples=samples, noise=0.2),
                 'chess': generate_points(samples, [(0, 0), (0, 1), (1, 0), (1, 1)], [0, 1, 1, 0], 0.005),
                 'q_random': generate_points_1d(nb=samples, centers=[-1, 0, 1], labels=[1, 0, 1])}
    data = datasets_[data_]
    
    if data_ == "chess" or data_ == "q_random":
        X, y = data
    else:
        X = data[0].T
        y = np.expand_dims(data[1], 1).T   
        
    network = nn.NeuralNetwork(architecture, seed=0)
    history = network.train(X=X, y=y, batch_size=4, epochs=5000, learning_rate=0.3, 
                               print_every=1000, validation_split=0.2, tqdm_=False,
                               plot_every=5000)
    
    weights, biases = history['weights'], history['biases']
     
    # Initialize the neural network scheme
    # np.shape(z0) = (input_dimension, samples)
    # Donc c'est un "vecteur" de input_dimension lignes 
    # et samples colonnes. Géneralement input_dimension = 1, 2
    z0 = X
    
    # We begin by checking if it's 1d. If yes, then we cannot use
    # scatter plot.
    if np.shape(z0)[0] == 1:
        # Red will contain the points whose labels correspond to 1, 
        # which we will paint in red
        red = list()
        # Blue will contain the points whose labels correspond to 0, 
        # which we will paint in blue
        blue = list()
        for i, x in enumerate(z0[0]):
            if y[0][i] == 0:
                blue.append(x)
            elif y[0][i] == 1:
                red.append(x)
            else:
                raise ValueError
                
        x_min, x_max = min(blue+red)-0.1, max(blue+red)+0.1
        major_ticks_x = np.linspace(x_min, x_max, 11)
        minor_ticks_x = np.linspace(x_min, x_max, 21)
        major_ticks_y = np.linspace(-0.5, 0.5, 11)
        minor_ticks_y = np.linspace(-0.5, 0.5, 21)
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xticks(major_ticks_x)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(major_ticks_y)
        ax.set_yticks(minor_ticks_y, minor=True)
        
        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.35)
        ax.grid(which='major', alpha=1)
        
        ax.plot(red, len(red)*[0.5], 'o', c='r', alpha=0.0)
        ax.plot(red, len(red)*[-0.5], 'o', c='r', alpha=0.0)
        
        ax.plot(red, len(red)*[0], 'o', c='r', alpha=0.55)
        ax.plot(blue, len(blue)*[0], 'o', c='b', alpha=0.55)
        plt.xlabel(r'$x$', fontdict = {'fontsize' : 16})
        plt.xlim(x_min, x_max)
        plt.ylim(-0.5, 0.5)
        plt.title(r'The N={} data points'.format(samples), fontdict = {'fontsize' : 24})
        
    elif np.shape(z0)[0] == 2:
        x_min, x_max = min(z0.T[:, 0]), max(z0.T[:, 0])
        y_min, y_max = min(z0.T[:, 1]), max(z0.T[:, 1])
        major_ticks_x = np.linspace(x_min, x_max, 11)
        minor_ticks_x = np.linspace(x_min, x_max, 21)
        major_ticks_y = np.linspace(y_min, y_max, 11)
        minor_ticks_y = np.linspace(y_min, y_max, 21)
        
        fig = plt.figure()
        ax = fig.add_sublplot(1, 1, 1)
        ax.set_xticks(major_ticks_x)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(major_ticks_y)
        ax.set_yticks(minor_ticks_y, minor=True)
        
        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.35)
        ax.grid(which='major', alpha=1)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        plt.scatter(z0.T[:, 0], z0.T[:,1], c=y.T.reshape(-1), cmap = plt.cm.coolwarm, alpha=0.55)
        plt.xlabel(r'$x_1$ coordinate', fontdict = {'fontsize' : 16})
        plt.ylabel(r'$x_2$ coordinate', fontdict = {'fontsize' : 16})
        plt.title(r'The N={} data points'.format(samples), fontdict = {'fontsize' : 24})
    else:
        raise TypeError 
        
    # /!\ Faut que je repare ça /!\
    Path('figures/visual_trans/{}/{}/{}d'.format(data_, samples, architecture[1])).mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/visual_trans/{}/{}/{}d/0.svg'.format(data_, samples, architecture[1]), format='svg')
    
    # We are now in a position to store the information from every transition.
    # We store the linear transformations (\Lambda_1 z^0, \Lambda_2 z^1 etc.) 
    # in the following array
    transitions = []
    # We store in layers the activations of each linear transformation, i.e.
    # z^0, z^1 = \sigma(\Lambda_1 z^0), z^2 = \sigma(\Lambda_2 z^1) etc.
    # We also store z^{L+1}. (thus the list should be of length L+2)
    layers = [z0]
    
    # We have the scheme z^{k} = \sigma(A^{k-1} z^{k-1} + b^{k-1})
    # for k = 1, ..., L; k represents the layers
    for k in range(1, len(architecture)):
        zkmoins1_list = copy.deepcopy(layers[k-1])
        
        # We initialize \Lambda_{k} as an array with
        # N_{k} rows and samples columns, while 
        # z^{k} has N_k rows and samples columns.
        # They both have samples columns, because we 
        # vectorize and compute over all of the data points
        
        Lambdak_list = np.zeros((architecture[k], samples)) 
        zk_list = np.zeros((architecture[k], samples))
        
        for i in range(samples):
            # We first compute the linear transitions
            # The reshape has the effect of giving an array of N_{k-1} rows
            # and 1 column (1 column because we do this for all i)
            Lambdak_datai = network.Lambda(k, zkmoins1_list[:, i].reshape(architecture[k-1], 1))
            
            # We fill the i-th column of every row with this linear transition
            # Things should be consistent because Lambdak has N_k rows, and
            # every element will have N_{k-1} columns
            Lambdak_list[:, i] = Lambdak_datai.reshape(-1)
            zk_list[:, i] = nn.sigmoid(Lambdak_datai).reshape(-1)
        transitions.append(Lambdak_list)
        layers.append(zk_list)
            
    # We plot the transitions, up to dimension 3 (cannot do higher)
    for i, lbd in enumerate(transitions):
        if len(lbd) == 2:
            x_min, x_max = min(lbd[0,:])-0.1, max(lbd[0, :])+0.1
            y_min, y_max = min(lbd[1, :])-0.1, max(lbd[1, :])+0.1
            major_ticks_x = np.linspace(x_min, x_max, 11)
            minor_ticks_x = np.linspace(x_min, x_max, 21)
            major_ticks_y = np.linspace(y_min, y_max, 11)
            minor_ticks_y = np.linspace(y_min, y_max, 21)
        
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xticks(major_ticks_x)
            ax.set_xticks(minor_ticks_x, minor=True)
            ax.set_yticks(major_ticks_y)
            ax.set_yticks(minor_ticks_y, minor=True)
        
            # Or if you want different settings for the grids:
            ax.grid(which='minor', alpha=0.35)
            ax.grid(which='major', alpha=1)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            
            plt.scatter(lbd.T[:, 0], lbd.T[:, 1], c=y.T.reshape(-1), cmap=plt.cm.coolwarm, alpha=0.55)
            #plt.xlabel(r'$1$st coordinate', fontdict = {'fontsize' : 16})
            #plt.ylabel(r'$2$nd coordinate', fontdict = {'fontsize' : 16})
        elif len(lbd) == 3:
            x_min, x_max = min(lbd[0,:])-0.1, max(lbd[0, :])+0.1
            y_min, y_max = min(lbd[1, :])-0.1, max(lbd[1, :])+0.1
            z_min, z_max = min(lbd[2, :])-0.1, max(lbd[2, :])+0.1
            
            major_ticks_x = np.linspace(x_min, x_max, 11)
            minor_ticks_x = np.linspace(x_min, x_max, 21)
            major_ticks_y = np.linspace(y_min, y_max, 11)
            minor_ticks_y = np.linspace(y_min, y_max, 21)
            major_ticks_z = np.linspace(z_min, z_max, 11)
            minor_ticks_z = np.linspace(z_min, z_max, 21)
            
            ax.set_xticks(major_ticks_x)
            ax.set_xticks(minor_ticks_x, minor=True)
            ax.set_yticks(major_ticks_y)
            ax.set_yticks(minor_ticks_y, minor=True)
            ax.set_zticks(major_ticks_z)
            ax.set_zticks(minor_ticks_z, minor=True)
            
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.grid(which='minor', alpha=0.35)
            ax.grid(which='major', alpha=1)
            
            ax.scatter(lbd.T[:, 0], lbd.T[:, 1], lbd.T[:, 2], c=y.T.reshape(-1), cmap=plt.cm.coolwarm, alpha=0.55)
            #plt.xlabel(r'$1$st coordinate', fontdict = {'fontsize' : 16})
            #plt.ylabel(r'$2$nd coordinate', fontdict = {'fontsize' : 16})
        else: 
            red = list()
            blue = list()
            lbd = copy.copy(lbd.reshape(-1))
            for j, e in enumerate(lbd):
                if y[0][j] == 0:
                    blue.append(e)
                else:
                    red.append(e)
            x_min, x_max = min(blue+red)-0.1, max(blue+red)+0.1
            
            major_ticks_x = np.linspace(x_min, x_max, 11)
            minor_ticks_x = np.linspace(x_min, x_max, 21)
            major_ticks_y = np.linspace(-0.5, 0.5, 11)
            minor_ticks_y = np.linspace(-0.5, 0.5, 21)
        
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xticks(major_ticks_x)
            ax.set_xticks(minor_ticks_x, minor=True)
            ax.set_yticks(major_ticks_y)
            ax.set_yticks(minor_ticks_y, minor=True)
        
            # Or if you want different settings for the grids:
            ax.grid(which='minor', alpha=0.35)
            ax.grid(which='major', alpha=1)
        
            ax.plot(red, len(red)*[0.5], 'o', c='r', alpha=0.0)
            ax.plot(red, len(red)*[-0.5], 'o', c='r', alpha=0.0)
        
            ax.plot(red, len(red)*[0], 'o', c='r', alpha=0.55)
            ax.plot(blue, len(blue)*[0], 'o', c='b', alpha=0.55)
            #plt.xlabel(r'$x$', fontdict = {'fontsize' : 16})
            plt.xlim(x_min, x_max)
            plt.ylim(-0.5, 0.5)
            
            #plt.xlabel(r'$1$st coordinate', fontdict = {'fontsize' : 16})
        plt.title(r'Linear transformation nb.{}: $\Lambda_{}z^{} = A^{} z^{} + b^{}$'.format(i+1, i+1, i, i, i, i), fontdict = {'fontsize' : 24})
        plt.savefig('figures/visual_trans/{}/{}/{}d/{}.svg'.format(data_, samples, architecture[1], 2*i+1), format='svg')
    
    for i, sig in enumerate(layers):
        if i==0:
            pass
        else:
            if len(sig) == 2:
                major_ticks = np.arange(-0.1, 1.1, 0.1)
                minor_ticks = np.arange(-0.1, 1.1, 0.05)
        
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.set_xticks(major_ticks)
                ax.set_xticks(minor_ticks, minor=True)
                ax.set_yticks(major_ticks)
                ax.set_yticks(minor_ticks, minor=True)
        
                # Or if you want different settings for the grids:
                ax.grid(which='minor', alpha=0.35)
                ax.grid(which='major', alpha=1)
                plt.xlim(0.0, 1.05)
                plt.ylim(0.0, 1.0)
                
                plt.scatter(sig.T[:, 0], sig.T[:, 1], c=y.T.reshape(-1), cmap=plt.cm.coolwarm, alpha=0.55)
                plt.xlabel(r'$z^{}_1$ coordinate'.format(i), fontdict = {'fontsize' : 16})
                plt.ylabel(r'$z^{}_2$ coordinate'.format(i), fontdict = {'fontsize' : 16})
                plt.title(r'Hidden layer nb.{}: $z^{} = \sigma(A^{} z^{} + b^{})$'.format(i, i, i-1, i-1, i-1), 
                      fontdict = {'fontsize' : 24})
            elif len(sig) == 3:
                fig = plt.figure()
                ax = Axes3D(fig)
                
                major_ticks = np.arange(0.0, 1.01, 0.1)
                minor_ticks = np.arange(0.0, 1.01, 0.05)
        
                ax.set_xticks(major_ticks)
                ax.set_xticks(minor_ticks, minor=True)
                ax.set_yticks(major_ticks)
                ax.set_yticks(minor_ticks, minor=True)
                ax.set_zticks(major_ticks)
                ax.set_zticks(minor_ticks, minor=True)
        
                # Or if you want different settings for the grids:
                ax.grid(which='minor', alpha=0.35)
                ax.grid(which='major', alpha=1)
                
                ax.scatter(sig.T[:, 0], sig.T[:, 1], sig.T[:, 2], c=y.T.reshape(-1), cmap=plt.cm.coolwarm, alpha=0.55)
                plt.xlabel(r'$z^{}_1$ coordinate'.format(i), fontdict = {'fontsize' : 16})
                plt.ylabel(r'$z^{}_2$ coordinate'.format(i), fontdict = {'fontsize' : 16})
                plt.title(r'Hidden layer nb.{}: $z^{} = \sigma(A^{} z^{} + b^{})$'.format(i, i, i-1, i-1, i-1), 
                      fontdict = {'fontsize' : 24})
            else:
                red = list()
                blue = list()
                sig = copy.copy(sig.reshape(-1))
                for j, e in enumerate(sig):
                    if y[0][j] == 0:
                        blue.append(e)
                    else:
                        red.append(e)
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                
                x_min, x_max = min(blue+red)-0.01, max(blue+red)+0.01
                major_ticks_x = np.linspace(x_min, x_max, 11)
                minor_ticks_x = np.linspace(x_min, x_max, 21)
                major_ticks_y = np.linspace(-0.5, 0.5, 11)
                minor_ticks_y = np.linspace(-0.5, 0.5, 21)
        
                ax.set_xticks(major_ticks_x)
                ax.set_xticks(minor_ticks_x, minor=True)
                ax.set_yticks(major_ticks_y)
                ax.set_yticks(minor_ticks_y, minor=True)
        
                # Or if you want different settings for the grids:
                ax.grid(which='minor', alpha=0.35)
                ax.grid(which='major', alpha=1)
                #plt.yticks(color='w')
                
                plt.plot(blue, len(blue)*[-0.5], 'o', c='b', alpha=0.0)
                plt.plot(red, len(red)*[0.5], 'o', c='r', alpha=0.0)
                
                plt.plot(blue, len(blue)*[0], 'o', c='b', alpha=0.55)
                plt.plot(red, len(red)*[0], 'o', c='r', alpha=0.55)
                plt.xlabel(r'$z^{}$ coordinate'.format(i), fontdict = {'fontsize' : 16})
                plt.xlim(x_min, x_max)
                plt.ylim(-0.5, 0.5)
                
                if i == len(layers)-1:
                    plt.title(r'Output of network: $z^{} = \sigma(A^{} z^{} + b^{}) \in [0, 1]$'.format(i, i-1, i-1, i-1, i-1), 
                              fontdict = {'fontsize' : 24})
                else:
                    plt.title(r'Hidden layer nb.{}: $z^{} = \sigma(A^{} z^{} + b^{}) \in [0, 1]^{{}}$'.format(i, i, i-1, i-1, i-1, architecture[i]), 
                              fontdict = {'fontsize' : 24})  
            plt.savefig('figures/visual_trans/{}/{}/{}d/{}.svg'.format(data_, samples, architecture[1], 2*i), format='svg')

                
    #### We generate a 2d grid
    x_list = np.arange(-0.1, 1.1, 0.01)
    y_list = np.arange(-0.1, 1.1, 0.01)
    
    if architecture[1] == 2:
        xx, yy = np.meshgrid(x_list, y_list)
        
        f1 = lambda x, y: nn.sigmoid(network.Lambda(2, np.array([[x], [y]])  ))[0,0]
        res = np.array([ [f1(x, y) for x in x_list] for y in y_list] )
    
        major_ticks = np.arange(-0.1, 1.1, 0.1)
        minor_ticks = np.arange(-0.1, 1.1, 0.05)
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        
        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.35)
        ax.grid(which='major', alpha=1)
    
        plt.scatter(layers[1].T[:, 0], layers[1].T[:, 1], c=y.T.reshape(-1), cmap=plt.cm.coolwarm, alpha=0.55)
        plt.contourf(xx, yy, res, cmap = plt.cm.coolwarm, alpha=0.35)
        plt.colorbar()
        droite = plt.contour(xx, yy, res, levels=[0.5], c = 'b', linewidth = 2.25, alpha=0.8)
        plt.clabel(droite, inline=1, fontsize=16)
      
        droite.collections[0].set_label(r'Level line $\{ x \in [0, 1]^2 \colon \sigma(A^1 x + b^1) = 0.5  \}$')
        
        plt.xlabel(r'$x_1$', fontdict = {'fontsize': 16})
        plt.ylabel(r'$x_2$', fontdict = {'fontsize': 16})
        plt.title(r'Level sets of $x \mapsto \sigma(A^1 x + b^1)$', fontdict = {'fontsize': 24})
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075),
                      fancybox=True, shadow=True, ncol=5, fontsize=18)
        plt.savefig('figures/visual_trans/{}/{}/{}d/{}.svg'.format(data_, samples, architecture[1], '2-5'), format='svg')
    
    if architecture[1] == 3:
        # It doesn't work
        
        a, b, c, d = weights[1][0][0], weights[1][0][1], weights[1][0][2], biases[1][0][0]
        X,Y = np.meshgrid(x_list,y_list)
        Z = (d - a*X - b*Y) / c
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, color='gray', alpha=0.35)
        ax.scatter(layers[1].T[:, 0], layers[1].T[:, 1], layers[1].T[:, 2], c=y.T.reshape(-1), cmap=plt.cm.coolwarm, alpha=0.55)
        
        #for angle in range(0,360): 
        ax.view_init(30, 60)
        
        plt.xlabel(r'$z^{}_1$ coordinate'.format(i), fontdict = {'fontsize' : 16})
        plt.ylabel(r'$z^{}_2$ coordinate'.format(i), fontdict = {'fontsize' : 16})
        plt.title(r'Hidden layer nb.{}: $z^{} = \sigma(A^{} z^{} + b^{})$'.format(i, i, i-1, i-1, i-1), 
                     fontdict = {'fontsize' : 24})
        
    
if __name__ == "__main__":
    #simulate(8, data_='blobs')
    #simulate(25, features=1, data_='blobs', architecture=[1, 2, 1])
    simulate(12, features=1, data_='q_random', architecture=[1, 2, 1])
    
    
###    # Just plot the activation functions 
#    x1 = np.linspace(-10, 10, 500)
#    x2 = np.linspace(-3, 3, 500)
#    y1 = list()
#    y2 = list()
#    for z in x1:
#        y1.append(nn.sigmoid(z))
#    for z in x2:
#        y2.append(nn.relu(z))
#    
#    for i in range(2):    
#        fig = plt.figure()
#        ax = fig.add_subplot(1, 1, 1)
#        
#        if i==0:
#            major_ticks_x = np.arange(-10, 10.15, 2.5)
#            minor_ticks_x = np.arange(-10, 10.15, 1.25)
#            major_ticks_y = np.arange(0, 1.15, 0.125)
#            minor_ticks_y = np.arange(0, 1.15, 0.0625)
#            ax.set_xticks(major_ticks_x)
#            ax.set_xticks(minor_ticks_x, minor=True)
#            ax.set_yticks(major_ticks_y)
#            ax.set_yticks(minor_ticks_y, minor=True)
#            
#            # And a corresponding grid
#            ax.grid(which='both')
#
#            # Or if you want different settings for the grids:
#            ax.grid(which='minor', alpha=0.25)
#            ax.grid(which='major', alpha=1)
#            ax.plot(x1, y1, color='purple', linewidth=5, alpha=0.45, linestyle='-', label=r'$\sigma(x) = (1+e^{-x})^{-1}$')
#            plt.title(r'The sigmoid activation function', fontdict={'fontsize': 20})
#            plt.xlim(-10, 10)
#            plt.ylim(-0.05, 1.05)
#        
#            # Put a legend to the right of the current axis
#            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075),
#                      fancybox=True, shadow=True, ncol=5, fontsize=18)
#            plt.savefig('figures/sigmoid.svg', format='svg')
#        else:
#            major_ticks_x = np.arange(-3, 3.1, 1)
#            minor_ticks_x = np.arange(-3, 3.1, 0.5)
#            major_ticks_y = np.arange(0, 3.01, 0.5)
#            minor_ticks_y = np.arange(0, 3.01, 0.25)
#            ax.set_xticks(major_ticks_x)
#            ax.set_xticks(minor_ticks_x, minor=True)
#            ax.set_yticks(major_ticks_y)
#            ax.set_yticks(minor_ticks_y, minor=True)
#            
#            # Different settings for the grids:
#            ax.grid(which='minor', alpha=0.25)
#            ax.grid(which='major', alpha=1)
#            plt.plot(x2, y2, color='purple', linewidth=5, alpha=0.45, linestyle='-', label=r'$\sigma(x) = \max(x, 0)$')            
#            plt.title(r'The ReLU activation function', fontdict={'fontsize': 20})
#            plt.xlim(-3, 3)
#            
#            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075),
#                      fancybox=True, shadow=True, ncol=5, fontsize=18)
#            plt.savefig('figures/relu.svg', format='svg')
#        plt.xlabel(r'$x$', fontdict = {'fontsize': 16})
#        plt.ylabel(r'$\sigma(x)$', fontdict = {'fontsize': 16})
        
        