from utils.dataManager import DataManager
from utils.log import Log
from nn.foamSVGD import FoamSVGD
import matplotlib.pyplot as plt
import os
import torch as th
from nn.turbnn import TurbNN
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


if __name__ == '__main__':


    lg = Log()

    # Define data location and timesteps
    trainingDir = ['rans-uncertainity/training-data/converge-diverge']
    trainingDir = [os.path.join(os.getcwd(), dir0) for dir0 in trainingDir]
    ransTimes = [60, 90, 60, 60, 60]
    lesTimes = [200, 1000, 250, 1700, 170]
    dataManager = DataManager(trainingDir, ransTimes, lesTimes)
    nsamples = 10
    foamNN = FoamSVGD(nsamples, 16) # Number of SVGD particles
    # Load pre-trained neural networks
    #foamNN.loadNeuralNet('./torchNets/foamNet')

    # First set up validation dataset
    #foamNN.getTestingPoints(dataManager, n_data=500, n_mb=256)
    dirs = '/mnt/c/Users/z5027487/Downloads/deepak_data/deepak_data/'
    XTdirs = [dirs+'deepak_X_train_data.csv']
    YTdirs = [dirs+'deepak_y_train_data.csv']
    Xdirs  = [dirs+'deepak_X_test_data.csv']
    Ydirs  = [dirs+'deepak_y_test_data.csv']

    n_mb=32
    foamNN.getDataPoints(dataManager, XTdirs, YTdirs, Xdirs, Ydirs, stp=2, n_mb=n_mb)

    lg.log('Batch size is ' + str(n_mb))
    n = 1 # Number of training sets
    n_data = [1000 for i in range(n)] # Number of data per training set
    n_mb = [1024 for i in range(n)] # Mini-batch size
    n_epoch = [100 for i in range(n)] # Number of epochs per training set

    foamNN.extra = "-" + str(foamNN.prior_w_shape) + "-" + str(foamNN.prior_w_rate) + "-lr-" + str(foamNN.lr) + "-" \
                    + str(foamNN.lr_noise) + "-bs-" +  str(foamNN.n_mb) + "-" + str(nsamples) + "-64neu-Vode"


    # Training loop
    for i in range(n):
        # Parse data and create data loaders
        #foamNN.getTrainingPoints(dataManager, n_data = n_data[i], n_mb = n_mb[i])

        lg.log('Training data-set number: '+str(i+1))
        foamNN.train(n_epoch[i], gpu=False)
        # Save neural networks
        foamNN.saveNeuralNet('foamNet'+foamNN.extra)
