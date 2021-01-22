"""
Data manager file is used for reading and parsing flow data for training
===
Distributed by: Notre Dame CICS (MIT Liscense)
- Associated publication:
url: https://www.sciencedirect.com/science/article/pii/S0021999119300464
doi: https://doi.org/10.1016/j.jcp.2019.01.021
github: https://github.com/cics-nd/rans-uncertainty
===
"""
from utils.log import Log
from utils.foamReader import FoamReader
from utils.torchReader import TorchReader
from fluid.invariant import Invariant

import sys, random, re, os
import torch as th
import numpy as np
import pandas as pd
# Default tensor type
dtype = th.DoubleTensor

class DataManager():

    def __init__(self):
        """
        Manages the training data, training data
        """
        self.lg = Log()

    def getDataPoints2D(self,fnameX,fnameY,stp):
        #dataT = self.read_data(fnameX,56)
        #reacT = self.read_reaction(fnameY)
        dataT = pd.read_csv(fnameX).to_numpy()
        reacT = pd.read_csv(fnameY).to_numpy()
        #return dataT[0:-1:stp,2:55], reacT[0:-1:stp,2:54]
        return dataT, reacT

    def getDataPoints2DTest(self,fnameXT,fnameYT,fnameX,fnameY):
        dataT = self.read_data(fnameX,56)
        reacT = self.read_reaction(fnameY)
        dataTe = self.read_data(fnameX,56)
        reacTe = self.read_reaction(fnameY)

        nc=55
        datanormT = self.do_normalization(dataTe[:,0:nc],dataT[:,0:nc],'std')
        nc2 = 54
        RRnormT = self.do_normalization(reacTe[:,0:nc2],reacT[:,0:nc2],'std')
        return th.from_numpy(datanormT[0:-1:10,:]).double(), th.from_numpy(RRnormT[0:-1:10,53,None]).double()


    def do_normalization(self,dataTr,dataT,which):

        if(which=='std'):
            datanormTr = (dataTr-np.mean(dataTr,0))/(np.std(dataTr,0))
            datanormT = (dataT-np.mean(dataTr,0))/(np.std(dataTr,0))
        if(which=='minmax'):
            datanormTr = (dataTr-np.min(dataTr,0))/(np.max(dataTr,0)-np.min(dataTr,0))
            datanormT = (dataT-np.min(dataTr,0))/(np.max(dataTr,0)-np.min(dataTr,0))



        return datanormTr, datanormT

            

    def read_data(self,fname,nc):    
            data = np.fromfile(fname,dtype=np.single)
            data = np.reshape(data,(int(data.size/nc),nc))
            #HRR = data[:,0]
            data = np.delete(data,0,1)
            return data
    def read_reaction(self,fname):
            data = np.fromfile(fname,dtype=np.single)
            data = np.reshape(data,(int(data.size/56),56))
            HRR = data[:,0]
            data = np.delete(data,0,1)
            data[:,53]=HRR
            data = np.delete(data,54,1)
            return data
