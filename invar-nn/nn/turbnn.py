'''
PyTorch turbulence neural network (TurbNN). To change the neural network 
architecture used change this file!
===
Distributed by: Notre Dame CICS (MIT Liscense)
- Associated publication:
url: https://www.sciencedirect.com/science/article/pii/S0021999119300464
doi: https://doi.org/10.1016/j.jcp.2019.01.021
github: https://github.com/cics-nd/rans-uncertainty
===
'''
import torch as th
import torch.nn.functional as F
import numpy as np
if th.cuda.is_available():
    dtype = th.cuda.DoubleTensor
else:
    dtype = th.DoubleTensor

# Turbulence Neural network
class TurbNN(th.nn.Module):

    def __init__(self, D_in, H, D_out):
        """
        Architecture of the turbulence deep neural net
        Args:
            D_in (Int) = Number of input parameters
            H (Int) = Number of hidden paramters
            D_out (Int) = Number of output parameters
        """
        super(TurbNN, self).__init__()
        self.linear1 = th.nn.Linear(D_in, H)
        self.f1 = th.nn.LeakyReLU()
        #self.linear2 = th.nn.Linear(H, H)
        #self.f2 = th.nn.LeakyReLU()
        #self.linear3 = th.nn.Linear(H, H)
        #self.f3 = th.nn.LeakyReLU()
        self.linear4 = th.nn.Linear(H, H)
        self.f4 = th.nn.LeakyReLU()
        self.linear5 = th.nn.Linear(H, H)
        self.f5 = th.nn.LeakyReLU()
        #self.linear6 = th.nn.Linear(H, H)
        #self.f6 = th.nn.LeakyReLU()
        self.linear7 = th.nn.Linear(H, D_out)
        self.f7 = th.nn.Sigmoid()
    def forward(self, x):
        """
        Forward pass of the neural network
        Args:
            x (th.DoubleTensor): [N x D_in] column matrix of training inputs
        Returns:
            out (th.DoubleTensor): [N x D_out] matrix of neural network outputs
        """
        lin1 = self.f1(self.linear1(x))
        #lin2 = self.f2(self.linear2(lin1))
        #lin3 = self.f3(self.linear3(lin2))
        lin4 = self.f4(self.linear4(lin1))
        lin5 = self.f5(self.linear5(lin4))
        #lin6 = self.f6(self.linear6(lin5))
        out  = self.linear7(lin5)
        return self.f7(out)

    def reset_parameters2(self):
        """
        Resets the weights of the neural network, samples from a normal guassian.
        """
        for x in self.modules():
            if isinstance(x, th.nn.Linear):
                x.weight.data = th.normal(th.zeros(x.weight.size()), th.zeros(x.weight.size())+1.0).type(dtype)
                x.bias.data = th.zeros(x.bias.size()).type(dtype)

    def reset_parameters(self, shape, rate):
        """
        Sample from student t distribution 
        """
        m=th.distributions.studentT.StudentT(2.0*shape)
        transforms = [th.distributions.transforms.AffineTransform(loc=0.0, scale=np.sqrt(rate/shape))]
        m2 = th.distributions.transformed_distribution.TransformedDistribution(m,transforms)
        for x in self.modules():
            if isinstance(x,th.nn.Linear):
                x.weight.data = m2.sample(x.weight.size()).type(dtype)
                x.bias.data = m2.sample(x.bias.size()).type(dtype)

