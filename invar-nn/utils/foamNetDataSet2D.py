'''
Custom datasets
===
Distributed by: Notre Dame CICS (MIT Liscense)
- Associated publication:
url: https://www.sciencedirect.com/science/article/pii/S0021999119300464
doi: https://doi.org/10.1016/j.jcp.2019.01.021
github: https://github.com/cics-nd/rans-uncertainty
===
'''
import torch as th
import torch.utils.data

class FoamNetDataset2D(th.utils.data.Dataset):
    """
    Dataset wrapping data and target tensors.
    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Args:
        invar_tensor (Tensor): contains invariant inputs
        tensor_tensor (Tensor): contains tensor basis functions
        k_tensor (Tensor): RANS TKE
        target_tensor (Tensor): Target anisotropic data tensor
    """

    def __init__(self, x_tensor, target_tensor):
        assert x_tensor.size(0) == target_tensor.size(0)
        self.x_tensor = x_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.x_tensor[index], self.target_tensor[index]

    def __len__(self):

        return self.x_tensor.size(0)

