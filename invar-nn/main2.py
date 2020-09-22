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
import torch as th
from nn.turbnn import TurbNN
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
if __name__ == '__main__':

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
    
    foamNN = FoamSVGD(20) # Number of SVGD particles
    # Load pre-trained neural networks
    #foamNN.loadNeuralNet('./torchNets/foamNet')
    
    # First set up validation dataset
    #foamNN.getTrainingPoints(dataManager, n_data=500, n_mb=1024)
    XTdirs = ['../../IgnDelay/xtrain0D.dat']
    YTdirs = ['../../IgnDelay/ytrain0D.dat']
    Xdirs = ['../../IgnDelay/xtrain0D.dat']
    Ydirs = ['../../IgnDelay/ytrain0D.dat']

    foamNN.getDataPoints(dataManager, XTdirs, YTdirs, Xdirs, Ydirs, stp=1,n_mb=1024)

    y_data = foamNN.trainingLoader.dataset.target_tensor
    x_data = foamNN.trainingLoader.dataset.x_tensor
    print(x_data.shape,y_data.shape)
    x_data = Variable(x_data)
    y_data = Variable(y_data)
    
    loss_func = th.nn.MSELoss()
    nn = TurbNN(D_in=55, H=128, D_out=54).double()
    optimizer = th.optim.Adam(nn.parameters(), lr=0.005)

    for i in range(200):
        out = nn.forward(x_data)
        loss = loss_func(out,y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        MSE = (out-y_data).pow(2).sum()
        print("EPOCH {}, LOSS {} ".format(i,MSE/(y_data.shape[0]*y_data.shape[1])))

    R2Scores = np.zeros(y_data.shape[1])
    for i in range(R2Scores.shape[0]):
        R2Scores[i] = r2_score(y_data[:,i],out[:,i].detach().numpy())
    print(R2Scores)
    plt.figure()
    plt.scatter(out[:,4].detach().numpy(),y_data[:,4],marker='.')
    plt.savefig('testfig.png')
