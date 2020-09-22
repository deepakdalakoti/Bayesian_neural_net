"""
Main run file for svgd invariant nn.
===
Distributed by: Notre Dame CICS (MIT Liscense)
- Associated publication:
url: https://www.sciencedirect.com/science/article/pii/S0021999119300464
doi: https://doi.org/10.1016/j.jcp.2019.01.021
github: https://github.com/cics-nd/rans-uncertainty
===
"""

from utils.dataManager import DataManager
from utils.log import Log
from nn.foamSVGD import FoamSVGD
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import r2_score
import numpy as np
if __name__ == '__main__':


    nparts = int(sys.argv[1])
    training = int(sys.argv[2])
    # Initialize logger
    lg = Log()

    # Define data location and timesteps
    trainingDir = ['../training-data/converge-diverge','../training-data/periodic-hills', \
                '../training-data/square-cylinder', '../training-data/square-duct', \
		'../training-data/tandem-cylinders']
    trainingDir = [os.path.join(os.getcwd(), dir0) for dir0 in trainingDir]
    ransTimes = [60, 90, 60, 60, 60]
    lesTimes = [200, 1000, 250, 1700, 170]
    dataManager = DataManager(trainingDir, ransTimes, lesTimes)
        
    foamNN = FoamSVGD(nparts) # Number of SVGD particles
    # Load pre-trained neural networks
    #foamNN.loadNeuralNet('./torchNets/foamNet')
    
    # First set up validation dataset
    #foamNN.getTrainingPoints(dataManager, n_data=500, n_mb=256)
    #foamNN.getTestingPoints(dataManager, n_data=500, n_mb=256)

    XTdirs = ['../../IgnDelay/xdataTr']
    YTdirs = ['../../IgnDelay/ydataTr']
    Xdirs = ['../../IgnDelay/xdataTe']
    Ydirs = ['../../IgnDelay/ydataTe']



    foamNN.getDataPoints(dataManager, XTdirs, YTdirs, Xdirs, Ydirs, stp=1, n_mb=64)
    foamNN.loadNeuralNet('torchNets/foamNet-0D')
    print(foamNN.models[0])
    mean, var, _ = foamNN.predict2(training,gpu=False)
    if(training):
        y_data = foamNN.trainingLoader.dataset.target_tensor
        x_data = foamNN.trainingLoader.dataset.x_tensor
    else:
        y_data = foamNN.testingLoaders[0].dataset.target_tensor
        x_data = foamNN.testingLoaders[0].dataset.x_tensor
    print(x_data.shape)    
    R2Score = np.zeros(y_data.shape[1])
    for i in range(R2Score.shape[0]):
        R2Score[i] = r2_score(y_data[:,i].detach().numpy(),mean[:,i].detach().numpy())
    print(R2Score)
    plt.figure()
    plt.scatter(y_data[:,4].detach().numpy(),mean[:,4].detach().numpy(),marker='.')
    #plt.ylim(-5,25)
    plt.savefig('test-mean.png')
    plt.clf()
    plt.scatter(y_data[:,4].detach().numpy(),var[:,4].detach().numpy(),marker='.')
    plt.savefig('test-var.png')
