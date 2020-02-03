#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:31:41 2020

@author: dycon
"""

__author__ = "Borjan Geshkovski"
__version__ = "0.1"

# Import relevant modules
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
from matplotlib import pyplot as plt
import numpy as np

# won't use scikit but some help functions 
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# tqdm is progress-bar. make sure it's installed: pip install tqdm
from tqdm import tqdm
from IPython import display

# Several of the standard activation functions
def sigmoid(z, derivative=False):
    """
    Sigmoid activation function, defined as
    \sigma(z) = 1/(1+exp(-z)).
    """
    if derivative == True:
        return sigmoid(z)*(1-sigmoid(z))
    else:
        return 1/(1 + np.exp(-z))
    
def relu(z, derivative=False):
    """
    ReLU activation function, defined as
    \sigma(z) = \max(z, 0)
    """
    if derivative == True:
        return (z>0)*1.0
    else:
        # this is the fastest way it appears. We could also use np.maximum(z, 0)
        return z*(z>0) 

# Here we define the output activation $\varphi$
# We distinguish 3 cases: 
#   - binary classification (use sigmoid)
#   - classification (use softmax)
#   - regression (use identity)

def softmax(z, derivative=False):
    """
    In construction
    """
    return 

def identity(z, derivative=False):
    """
    Linear activation function, defined as
    \sigma(z) = z
    """
    if derivative == True:
        return 1.0
    else:
        return z
        
# The cost functional     
def cost_function(y_true, y_pred):
    """
    Returns a scalar value representing the "loss"
    This will be the functional depending on the set 
    of parameters to be optimized.
    In our case, we take the simple example of the
    square L2 norm of the discrepencies
    """
    
    # Probablement y_pred est un vecteur ligne, donc n est bien egal au nombre de sorties m
    n = y_pred.shape[1]
    cost = (1./(2*n))*np.sum((y_true-y_pred)**2)
    return cost

def cost_function_prime(y_true, y_pred):
    """
    Computes the derivative of the loss function w.r.t the sigmoid of the output layer
    """
    cost_prime = y_pred - y_true
    return cost_prime

class NeuralNetwork(object):     
    '''
    Parameters
    ---
    size: list of number of neurons per layer

    Examples
    ---
    >>> import NeuralNetwork
    >>> nn = NeuralNetwork([2, 3, 4, 1])
    
    This means :
    1 input layer with 2 neurons
    1 hidden layer with 3 neurons
    1 hidden layer with 4 neurons
    1 output layer with 1 neuron
    
    '''

    def __init__(self, size, seed=42):
        '''
        Initialize the weights and biases of the network
        weights and biases are attributes of the NeuralNetwork class
        They are updated during the training
        '''
        self.seed = seed
        np.random.seed(self.seed)
        # size est une liste de la forme [N0, N1, .., NL] contenant
        # les largeurs des couches. L = len(size) designe ainsi la 
        # depth (profondeur) du reseau, et chacun des Nj represente
        # la largeur de la couche j.
        self.size = size
        
        self.weights = [np.random.randn(self.size[i], self.size[i-1])*np.sqrt(1/self.size[i-1]) for i in range(1, len(self.size))]
        self.biases = [np.random.rand(n, 1) for n in self.size[1:]]

    def forward(self, input_):
        '''
        Perform a feed forward computation 

        Parameters
        ---
        input: data to be fed to the network with, in occurence == batch_x
        shape: (input_shape, batch_size)

        Returns
        ---
        x: ouptut activation (output_shape, batch_size)
        pre_activations: list of pre-activations per layer
        each of shape (n[l], batch_size), where n[l] is the number 
        of neuron at layer l
        activations: list of sigmoids per layer
        each of shape (n[l], batch_size), where n[l] is the number 
        of neuron at layer l

        '''
        x = input_
        pre_activations = []
        activations = [x]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, x) + b
            x  = sigmoid(z)
            pre_activations.append(z)
            activations.append(x)
        # Donc x est la derniere transition, donc le output du reseau.
        # pre_activations est une lite contenant toutes les transformations
        # lineaires, et activations contient toutes les activations evalues
        # dans ces transformations lineaires, donc ce sont les valeurs des 
        # neurones par couche 
        return x, pre_activations, activations

    def compute_deltas(self, pre_activations, y_true, y_pred):
        """
        Computes the partial derivatives of the cost functional w.r.t. z^l
        Namely, we usually denote z^l = W_l x + b^l, at layer l, where x is 
        the input from the previous layer l-1.
        Then here we compute the partial derivatives w.r.t. z^l, in principle.
        
        The parameters:
        ---
        pre_activations: list of of pre-activations. each corresponding to a layer
        y_true: values of the labels from data
        y_pred: prediction values of the labels
        Returns:
        ---
        partials: a list of the partial derivatives per layer
        In fact, all of these computations can be find synthetically written
        in the paper "Deep Learning: an Introduction for Applied Mathematicians"
        in Siam Review, 2018. The notations in this code are more or less the 
        same as in this paper.
        """
        # We begin by computing the gradient of the (sub)functional with respect to z^L, 
        # so we start by the last layer. We use the variable delta_L to denote
        # the vector $\del C/\del_{z^L}$, which has components $\del C/\del_{z^L_j}$
        # for j = 1, .., N_L (width).
        # The list deltas will contain the remaining $\delta_l$, defined analogously.
        # Using the chain rule, one can show that $\delta_L = \sigma'(z^L)\circ (a^L-y)$
        # where \circ denotes the Hadamard (componentwise) product of two vectors.
        delta_L = cost_function_prime(y_true, y_pred)*sigmoid(pre_activations[-1], derivative=True)
        deltas = [0]*(len(self.size)-1)
        deltas[-1] = delta_L
        
        for l in range(len(deltas)-2, -1, -1):
            # These formulas can be found in the above-cited paper.
            delta = np.dot(self.weights[l + 1].transpose(), deltas[l + 1])*sigmoid(pre_activations[l], derivative=True) 
            deltas[l] = delta
        return deltas

    def backpropagate(self, deltas, pre_activations, activations):
        """
        Applies back-propagation and computes the gradient of the loss
        w.r.t the weights and biases of the network

        Parameters:
        ---
        deltas: list of deltas computed by compute_deltas
        pre_activations: a list of pre-activations per layer
        activations: a list of sigmoids per layer
        Returns:
        ---
        dW: list of gradients w.r.t. the weight matrices of the network
        db: list of gradients w.r.t. the biases (vectors) of the network
    
        """        
        dW = []
        db = []
        deltas = [0] + deltas
        for l in range(1, len(self.size)):
            dW_l = np.dot(deltas[l], activations[l-1].transpose()) 
            db_l = deltas[l]
            dW.append(dW_l)
            db.append(np.expand_dims(db_l.mean(axis=1), 1))
        return dW, db

    def plot_decision_regions(self, X, y, iteration, train_loss, val_loss, train_acc, val_acc, res=0.01):
        """
        Plots the decision boundary at each iteration 

        Parameters:
        ---
        X: the input data
        y: the labels
        iteration: the epoch (optimization iteration) number
        train_loss: value of the training loss
        val_loss: value of the validation loss
        train_acc: value of the training accuracy
        val_acc: value of the validation accuracy
        res: resolution of the plot
        Returns:
        ---
        None: this function plots the decision boundary
        """
        X, y = X.T, y.T 
        
        if np.shape(X)[1] >=2:
        
            # We define the delimitors for the figure
            x_min, x_max = X[:, 0].min()-0.5, X[:, 0].max()+0.5
            y_min, y_max = X[:, 1].min()-0.5, X[:, 1].max()+0.5
            
            # We generate a 2d grid
            xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                                np.arange(y_min, y_max, res))
            
            Z = self.predict(np.c_[xx.ravel(), yy.ravel()].T)
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap = plt.cm.coolwarm, alpha=0.35)
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            # We superpose the data points
            plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), cmap = plt.cm.coolwarm, alpha=0.55)
            message = r'Truncated output of $\sigma(A^L z^L+b^L)$ at iteration: {} '.format(iteration+1)
            plt.xlabel(r'$x_1$ coordinate', fontdict = {'fontsize' : 12})
            plt.ylabel(r'$x_2$ coordinate', fontdict = {'fontsize' : 12})
            plt.title(message, fontdict = {'fontsize' : 18})
        
        else:
            
            x_min = X[:, 0].min() -0.5
            x_max = X[:, 0].max() +0.5
            
            xx = np.arange(x_min, x_max, res)
            
            Z = self.predict(np.c_[xx.ravel()].T)
            Z = Z.reshape(xx.shape)
            #print(xx[Z>0])
            
            #plt.contourf(xx, yy, Z, cmap = plt.cm.coolwarm, alpha=0.35)
            plt.xlim(xx.min(), xx.max())
            plt.ylim(-0.1, 1.1)
            # We superpose the data points
            red = list()
            blue = list()
   
            z0 = X.T
            #print(z0[0])
            for i, ix in enumerate(z0[0]):
                if y.T[0][i] == 0:
                    red.append(ix)
                else:
                    blue.append(ix)
                
            plt.plot(blue, len(blue)*[0], 'o', c='r')
        
            plt.plot(red, len(red)*[0], 'o', c='b')
            
            #plt.plot(xx, Z)
            plt.plot(xx[Z>0], (Z[Z>0]), c= 'r', linewidth=5)
            plt.plot(xx[Z==0], (Z[Z==0]+np.ones(len(Z[Z==0]))), c='b', linewidth=5)
            
            #plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), cmap = plt.cm.coolwarm, alpha=0.55)
            message = r'Truncated output of $\sigma(A^L z^L+b^L)$ at iteration: {} '.format(iteration+1)
            plt.xlabel(r'$x_1$ coordinate', fontdict = {'fontsize' : 12})
            #plt.ylabel(r'$x_2$ coordinate', fontdict = {'fontsize' : 12})
            plt.title(message, fontdict = {'fontsize' : 18})
            
        
        # Fix this
        #plt.savefig('figures/fig_%s.png' % iteration, dpi=450)                                                                                                  val_acc)

    def train(self, X, y, batch_size, epochs, learning_rate, validation_split=0.2, print_every=10, tqdm_=True, plot_every=None):
        """

        Parameters:
        X: input data
        y: input labels
        batch_size: number of data points to process in each batch (batch = ?)
        epochs: number of epochs for the training (epoch == iteration)
        learning_rate: value of the learning rate
        validation_split: percentage of the data for validation
        print_every: the number of epochs by which the network logs the loss and accuracy metrics for train and validations splits
        tqdm_: use tqdm progress-bar
        plot_every: the number of epochs by which the network plots the decision boundary
    
        Returns:
        ---
        history: dictionary of train and validation metrics per epoch
            train_acc: train accuracy
            test_acc: validation accuracy
            train_loss: train loss
            test_loss: validation loss
        """
        history_train_losses = []
        history_train_accuracies = []
        history_test_losses = []
        history_test_accuracies = []


        # train_test_split est un module predefini de scikit-learn
        x_train, x_test, y_train, y_test = train_test_split(X.T, y.T, test_size=validation_split, )
        x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T 

        # Simply not necessary
        if tqdm_:
            epoch_iterator = tqdm(range(epochs))
        else:
            epoch_iterator = range(epochs)

        # On fait les iterations
        for e in epoch_iterator:
            if x_train.shape[1] % batch_size == 0:
                n_batches = int(x_train.shape[1]/batch_size)
            else:
                n_batches = int(x_train.shape[1]/batch_size)-1

            x_train, y_train = shuffle(x_train.T, y_train.T)
            x_train, y_train = x_train.T, y_train.T

            batches_x = [x_train[:, batch_size*i:batch_size*(i+1)] for i in range(0, n_batches)]
            batches_y = [y_train[:, batch_size*i:batch_size*(i+1)] for i in range(0, n_batches)]

            train_losses = []
            train_accuracies = []
            
            test_losses = []
            test_accuracies = []

            dw_per_epoch = [np.zeros(w.shape) for w in self.weights]
            db_per_epoch = [np.zeros(b.shape) for b in self.biases] 
            
            for batch_x, batch_y in zip(batches_x, batches_y):
                # Là on calcule le output du reseau de neurones
                batch_y_pred, pre_activations, activations = self.forward(batch_x)
                # On calcule les derivées partielles de la fonctionelle cout aux paramètres
                deltas = self.compute_deltas(pre_activations, batch_y, batch_y_pred)
                dW, db = self.backpropagate(deltas, pre_activations, activations)
                for i, (dw_i, db_i) in enumerate(zip(dW, db)):
                    dw_per_epoch[i] += dw_i / batch_size
                    db_per_epoch[i] += db_i / batch_size

                batch_y_train_pred = self.predict(batch_x)

                train_loss = cost_function(batch_y, batch_y_train_pred)
                train_losses.append(train_loss)
                train_accuracy = accuracy_score(batch_y.T, batch_y_train_pred.T)
                train_accuracies.append(train_accuracy)

                batch_y_test_pred = self.predict(x_test)

                test_loss = cost_function(y_test, batch_y_test_pred)
                test_losses.append(test_loss)
                test_accuracy = accuracy_score(y_test.T, batch_y_test_pred.T)
                test_accuracies.append(test_accuracy)


            # weight update using the batch gradient descent scheme
            for i, (dw_epoch, db_epoch) in enumerate(zip(dw_per_epoch, db_per_epoch)):
                self.weights[i] = self.weights[i] - learning_rate * dw_epoch
                self.biases[i] = self.biases[i] - learning_rate * db_epoch

            history_train_losses.append(np.mean(train_losses))
            history_train_accuracies.append(np.mean(train_accuracies))
            
            history_test_losses.append(np.mean(test_losses))
            history_test_accuracies.append(np.mean(test_accuracies))


            if not plot_every:
                if e % print_every == 0:    
                    print('Epoch {} / {} | train loss: {} | train accuracy: {} | val loss : {} | val accuracy : {} '.format(
                        e, epochs, np.round(np.mean(train_losses), 3), np.round(np.mean(train_accuracies), 3), 
                        np.round(np.mean(test_losses), 3),  np.round(np.mean(test_accuracies), 3)))
            else:
                if e % plot_every == 0:
                    self.plot_decision_regions(x_train, y_train, e, 
                                                np.round(np.mean(train_losses), 4), 
                                                np.round(np.mean(test_losses), 4),
                                                np.round(np.mean(train_accuracies), 4), 
                                                np.round(np.mean(test_accuracies), 4), 
                                                )
                    plt.show()                    
                    display.display(plt.gcf())
                    display.clear_output(wait=True)

        self.plot_decision_regions(X, y, e, 
                                    np.round(np.mean(train_losses), 4), 
                                    np.round(np.mean(test_losses), 4),
                                    np.round(np.mean(train_accuracies), 4), 
                                    np.round(np.mean(test_accuracies), 4), 
                                    )

        history = {'epochs': epochs,
                   'train_loss': history_train_losses, 
                   'train_acc': history_train_accuracies,
                   'test_loss': history_test_losses,
                   'test_acc': history_test_accuracies,
                   'pre_activ': pre_activations,
                   'activ': activations,
                   'weights': self.weights,
                   'biases': self.biases
                   }
        
        return history

    def predict(self, a):
        '''
        Use the current state of the network to make predictions

        Parameters:
        ---
        a: input data, shape: (input_shape, batch_size)

        Returns:
        ---
        predictions: vector of output predictions
        '''
        
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = sigmoid(z)
        # We threshold the output of the neural network; 
        # as it's a float in [0, 1]; we cut it to obtain the final result.
        predictions = (a > 0.5).astype(int)
        return predictions
