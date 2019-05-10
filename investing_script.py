#!/usr/bin/env python3

# Author: Victor BG
# Date: March 10th 2018

# This script intends to show a simple example of decision-taking using sigmoid neurons.
# The inputs are 5-day, 120-day, and 1-year averages.
# We use the 5-day average to avoid a ponctual decrease in stock values
# We use the 120-day average to invest while the stock is currently under-valued.
# We use the 1-year average to see if the stock is generally increasing in value.

# Sources for the time serie:
# https://finance.yahoo.com/quote/AMZN/
# NOTE: We use the opening price

#---------#
# IMPORTS #
#---------#
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys

#---------------#
# INTIALIZATION #
#---------------#
# Loading the file
try:
    file_stock_1 = sys.argv[1]
except:
    warnings.warn('No file name given as argument, using the default AMZN.csv')
    file_stock_1 = 'AMZN.csv'
dat = np.genfromtxt(file_stock_1, delimiter=',', dtype='float32', skip_header=1, usecols=(1))

# The daily growth
dif = (dat[1:]/dat[:-1] - 1)

# The moving averages
win_005 = np.ones(5)/5
win_120 = np.ones(120)/120
win_365 = np.ones(365)/365

avg_005 = np.convolve(dat, win_005, 'valid')
avg_120 = np.convolve(dat, win_120, 'valid')
avg_365 = np.convolve(dat, win_365, 'valid')

dif_005 = np.convolve(dif, win_005, 'valid')
dif_120 = np.convolve(dif, win_120, 'valid')
dif_365 = np.convolve(dif, win_365, 'valid')

# The current averages for decision-taking
dec_dat_ref = dat[-1]
dec_dat_005 = avg_005[-1]
dec_dat_120 = avg_120[-1]
dec_dat_365 = avg_365[-1]

dec_dif_ref = dif[-1]
dec_dif_005 = dif_005[-1]
dec_dif_120 = dif_120[-1]
dec_dif_365 = dif_365[-1]

# Selecting only the data for a few days for plotting
plot_window = 365
dat = dat[-plot_window:]
avg_005 = avg_005[-plot_window:]
avg_120 = avg_120[-plot_window:]
avg_365 = avg_365[-plot_window:]

dif = dif[-plot_window:]
dif_005 = dif_005[-plot_window:]
dif_120 = dif_120[-plot_window:]
dif_365 = dif_365[-plot_window:]

#-----------------#
# DECISION-TAKING #
#-----------------#
# A class for a sigmoid neuron
class SigmoidNeuron():
    def __init__(self, beta, vec_weights=None, vec_inputs=None):
        self.beta = beta
        self.weights = vec_weights
        self.input = vec_inputs

    def set_weights(self, vector):
        self.weights = vector

    def set_input(self, vector):
        self.input = vector

    def compute_output(self):
        # Checking if inputs and weights are defined
        if not(self.weights) or not(self.input):
            raise ValueError('Error in SigmoidNeuron.compute_output(): Either the weight vector or the input vector, or both, are not defined')
        out = np.dot(self.weights, self.input)
        out += self.beta
        out = 1/(1+np.exp(-out))
        self.output = out

# Formatting the input data
in_005_dat = dec_dat_ref/dec_dat_005 - 1
in_005_dif = dec_dif_005

in_120_dat = dec_dat_ref/dec_dat_120 - 1
in_120_dif = dec_dif_120

in_365_dat = dec_dat_ref/dec_dat_365 - 1
in_365_dif = dec_dif_365

print('-------')
# We use one neuron for each the short-term, middle-term, and long-term data
sig_005 = SigmoidNeuron(beta=0)
sig_005.set_input([in_005_dat, in_005_dif])
sig_005.set_weights([0.01, 1000.0])
sig_005.compute_output()
print('The 5-day neuron:')
print('-> We don\'t look at the value for this neuron')
print('-> We desire growth and observe a {0:.2f}% growth.'.format(in_005_dif*100))
print('-> The 5-day neuron output is: {0:.4f}'.format(sig_005.output))
print('')

sig_120 = SigmoidNeuron(beta=-1)
sig_120.set_input([in_120_dat, in_120_dif])
sig_120.set_weights([-10.0, 0.01])
sig_120.compute_output()
print('The 120-day neuron: ')
print('-> We desire a negative value and observe a {0:.2f} comparative value.'.format(in_120_dat*100))
print('-> We don\'t look at growth for this neuron')
print('-> The 120-day neuron output is: {0:.4f}'.format(sig_120.output))
print('')

sig_365 = SigmoidNeuron(beta=-1)
sig_365.set_input([in_365_dat, in_365_dif])
sig_365.set_weights([0.01, 1000.0])
sig_365.compute_output()
print('The 365-day neuron: ')
print('-> We don\'t look at the value for this neuron')
print('-> We desire growth and observe a {0:.2f}% growth.'.format(in_365_dif*100))
print('-> The 365-day neuron output is: {0:.4f}'.format(sig_365.output))
print('')

sig_dec = SigmoidNeuron(beta=-2.25)
sig_dec.set_input([sig_005.output, sig_120.output, sig_365.output])
sig_dec.set_weights([1.0,1.0,1.0])
sig_dec.compute_output()

print('For the final neuron, we expect at least a 75% score from each neuron.')
print('The final output: {0:.4f}'.format(sig_dec.output))

if sig_dec.output > 0.5:
    print('Buy!')
else:
    print('Sell!')

print('')
print('Note: looking at the plot, and general knowledge of Amazon\'s expected future growth may tell us that the right choice is to buy. Further development is necessary')
print('-------')
#----------#
# PLOTTING #
#----------#
bool_plot = True
if bool_plot:
    fig = plt.figure(figsize=(16,9))

    ax1 = fig.add_subplot(211)
    ax1.set_title('Moving averages for the AMZN stock\'s value')
    ax1.plot(dat, '-k', label='AMZN value')
    ax1.plot(avg_005, '-r', label='5-day average')
    ax1.plot(avg_120, '-g', label='3-month average')
    ax1.plot(avg_365, '-b', label='1-year average')
    ax1.plot([0,plot_window-1], [dec_dat_ref, dec_dat_ref], '--k')
    ax1.plot([0,plot_window-1], [dec_dat_005, dec_dat_005], '--r')
    ax1.plot([0,plot_window-1], [dec_dat_120, dec_dat_120], '--g')
    ax1.plot([0,plot_window-1], [dec_dat_365, dec_dat_365], '--b')
    ax1.set_xlim(0, plot_window-1)
    ax1.set_ylabel('Stock value [USD]')
    ax1.legend(loc='upper right', ncol=4)

    ax2 = fig.add_subplot(212)
    ax2.set_title('Moving averages for the AMZN stock\'s growth')
    ax2.plot(dif*100, '-k', label='AMZN growth')
    ax2.plot(dif_005*100, '-r', label='5-day average')
    ax2.plot(dif_120*100, '-g', label='3-month average')
    ax2.plot(dif_365*100, '-b', label='1-year average')
    ax2.plot([0,plot_window-1], [dec_dif_ref*100, dec_dif_ref*100], '--k')
    ax2.plot([0,plot_window-1], [dec_dif_005*100, dec_dif_005*100], '--r')
    ax2.plot([0,plot_window-1], [dec_dif_120*100, dec_dif_120*100], '--g')
    ax2.plot([0,plot_window-1], [dec_dif_365*100, dec_dif_365*100], '--b')
    ax2.set_xlim(0,plot_window-1)
    ax2.set_ylabel('Daily stock growth [%]')
    ax2.set_xlabel('Time [days]')
    ax2.legend(loc='upper right', ncol=4)

    plt.show()

# Joke: if AI gets applied in psychology, we will have to use Sigmoid Freud neurons.
