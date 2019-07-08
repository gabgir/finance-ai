# Author: Victor BG
# Date: June 20, 2019

# This script trains and tests a deep neural network
# Heavily inspired by the Coursera course "Neural networks and deep learning"
# by Andrew Ng

# Source for L2 regularization
# Deep learning in finance, J. Heaton et al., arXiv, 2018
# https://towardsdatascience.com/how-to-improve-a-neural-network-with-regularization-8a18ecda9fe3

#---------#
# IMPORTS #
#---------#
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import cv2

#-------#
# CLASS #
#-------#
class Neural:
    def __init__(self):
        self.X = None
        self.Y = None
        self.As = None

        self.regularization_factor = 0.

    #----------------#
    # INITIALIZATION #
    #----------------#
    def load_dataset(self, inpath):
        # Loading dataset
        with open(inpath, "r") as f:
            dataset = json.load(f)
        self.X = np.array(dataset["X"])
        self.Y = np.array(dataset["Y"])

        # The dimensions of the training dataset
        self.num_input, self.num_example = self.X.shape

        if self.As == None:
            raise ValueError("A neural network needs to be defined before loading a dataset")
        else:
            self.As[0] = self.X
            print("New dataset loaded")


    def initialize_neural_network(self,
            regularization_factor,
            learning_rate=0.1,
            hidden_layers=[],
            factor_init=0.01):

        # Hyperparameters
        self.num_neurons = [self.num_input] + hidden_layers + [1]
        self.num_layers = len(self.num_neurons)
        self.learning_rate = learning_rate
        self.regularization_factor = regularization_factor
        self.factor_init = factor_init
        self.actfun = [None] + [self.sigmoid]*(self.num_layers-1)
        # self.actfun = [None]\
        #     + [self.tanh for _ in range(self.num_layers-2)]\
        #     + [self.sigmoid]

        # Parameters
        # self.Ws = [None]+[np.random.rand(self.num_neurons[ind],\
        #     self.num_neurons[ind-1])*factor_init \
        #     for ind in range(1, self.num_layers)]
        self.Ws = [None]+[\
            (np.random.rand(self.num_neurons[ind],\
            self.num_neurons[ind-1])-0.5)*self.factor_init \
            for ind in range(1, self.num_layers)]
        self.bs = [None]+[(np.random.rand(self.num_neurons[ind], 1)-0.5)*\
            self.factor_init \
            for ind in range(1, self.num_layers)]
        self.Zs = [None]*self.num_layers
        self.As = [None]*self.num_layers
        self.As[0] = self.X

        # Derivatives
        self.dWs = [None]*self.num_layers
        self.dbs = [None]*self.num_layers
        self.dZs = [None]*self.num_layers
        self.dAs = [None]*self.num_layers

    #--------------------#
    # SAVING AND LOADING #
    #--------------------#
    def save_network(self, outpath):
        W = [self.Ws[ind].tolist() for ind in range(1, self.num_layers)]
        b = [self.bs[ind].tolist() for ind in range(1, self.num_layers)]

        dumpfile = {
            "W": W,
            "b": b}

        with open(outpath, "w") as f:
            json.dump(dumpfile, f)

    def load_network(self, inpath):
        with open(inpath, "r") as f:
            d = json.load(f)

        W = d["W"]
        b = d["b"]
        W = [None] + [np.array(W[ind]) for ind in range(len(W))]
        b = [None] + [np.array(b[ind]) for ind in range(len(b))]

        self.num_neurons = [W[1].shape[1]] + [W[ind].shape[0] \
            for ind in range(1, len(W))]
        self.num_layers = len(self.num_neurons)

        self.Ws = W
        self.bs = b

        self.actfun = [None] + [self.sigmoid]*(self.num_layers-1)

        # self.actfun = [None]\
        #     + [self.tanh for _ in range(self.num_layers-2)]\
        #     + [self.sigmoid]


        self.Zs = [None]*self.num_layers
        self.As = [None]*self.num_layers

    #-------------#
    # SUBROUTINES #
    #-------------#
    def sigmoid(self, input, fwd=True):
        if fwd:
            return 1/(1+np.exp(-input))
        else:
            return np.exp(-input) / (1+np.exp(-input))**2

    def tanh(self, input, fwd=True):
        if fwd:
            return np.tanh(input)
        else:
            return 1 - np.tanh(input)**2

    def cost(self, A, Y):
        return -(Y*np.log(A) + (1-Y)*np.log(1-A))

    #----------#
    # ROUTINES #
    #----------#
    def forward_propagation(self):
        for ind in range(1, self.num_layers):
            self.Zs[ind] = np.dot(self.Ws[ind], self.As[ind-1]) + self.bs[ind]
            self.As[ind] = self.actfun[ind](self.Zs[ind], fwd=True)

        # The cost function
        loss = self.cost(self.As[-1], self.Y).mean()
        reg = [np.sum(self.Ws[ind]**2) for ind in range(1, self.num_layers)]
        reg = self.regularization_factor*np.sum(reg)/(2*self.num_example)

        return loss + reg

    def backward_propagation(self):
        for ind in range(1, self.num_layers):
            if ind==1:
                self.dZs[-ind] = self.As[-ind] - self.Y
            else:
                self.dZs[-ind] = np.dot(self.Ws[-ind+1].T, self.dZs[-ind+1]) *\
                    self.actfun[-ind](self.Zs[-ind], fwd=False)

            # Computing the derivatives
            self.dWs[-ind] = 1/self.num_example\
                * np.dot(self.dZs[-ind], self.As[-ind-1].T)\
                + self.regularization_factor*self.Ws[-ind]/self.num_example

            self.dbs[-ind] = 1/self.num_example\
                * np.sum(self.dZs[-ind], axis=1, keepdims=True)

        # Updating the parameters
        self.Ws = [None]+[self.Ws[ind] - self.learning_rate*self.dWs[ind]\
            for ind in range(1, self.num_layers)]
        self.bs = [None]+[self.bs[ind] - self.learning_rate*self.dbs[ind]\
            for ind in range(1, self.num_layers)]

    def train_network(self, num_iter, interval=100):
        print("Iteration\tCost\t\tTime")
        t0 = time.time()
        for iter_grad in range(num_iter):
            L = self.forward_propagation()
            self.backward_propagation()

            # Printing the results
            if iter_grad%interval == 0:
                t = time.time()
                print("{}\t\t{:.5f}\t\t{:.2f}".format(iter_grad, L, t-t0))
                # cv2.imshow('frame', self.Ws[1])

    def evaluate_dataset(self):
        self.forward_propagation()

        A = np.around(self.As[-1])
        success_rate = np.mean((1 - abs(A-self.Y)**2))
        print("The success rate of the neural network: {:.0f}%".format(success_rate*100))

     # Not super clean, would require some rework of Neural class to clean up
    def apply_network(self, vector):
        # Saving the current values
        temp_A = self.As
        temp_Y = self.Y

        # Loading the desired test values
        self.X = vector.reshape(len(vector), 1)
        self.As[0] = self.X
        self.Y = 1.

        # The dimensions of the training dataset
        self.num_input, self.num_example = self.X.shape

        if self.As == None:
            raise ValueError("A neural network needs to be defined before loading a dataset")
        else:
            self.As[0] = self.X
            # print("New dataset loaded")

        self.forward_propagation()
        return self.As[-1]


#------#
# MAIN #
#------#
if __name__=="__main__":
    nrl = Neural()

    nrl.load_dataset("./config/dataset.json")
    print("The average number of successes in the training dataset: {:.0f}%".format(100*np.mean(nrl.Y)))
    nrl.initialize_neural_network(hidden_layers=[100, 50],
        regularization_factor=0.05,
        learning_rate=0.1,
        factor_init=0.1)
    nrl.evaluate_dataset()
    nrl.train_network(num_iter=100000, interval=500)
    nrl.save_network("neural.json")

    # nrl.load_network("neural_100k.json")
    nrl.load_dataset("./config/test_dataset.json")
    nrl.evaluate_dataset()

    # aa = nrl.Ws[1]
    # print(aa.shape)
    # plt.imshow(aa, cmap='gray')
    # plt.show()
