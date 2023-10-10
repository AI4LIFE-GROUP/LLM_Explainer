import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import Module
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dim_per_layer, activation_per_layer):
        super().__init__()
        '''
            dim_per_layer: list of the dimensions per layer
        '''

        self.dim_per_layer = dim_per_layer

        # variable number of hidden layers (loop over each layer)
        self.layers = nn.ModuleList()
        for l in range(len(self.dim_per_layer) - 1):  # loop over each layer to add
            self.layers.append(nn.Linear(self.dim_per_layer[l], self.dim_per_layer[l+1]))

            activation = activation_per_layer[l]
            if activation is not None:  # Do we have an activation function?
                assert isinstance(activation, Module), "Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)
    
    def predict(self, data):
        """
        predict method for CFE-Models which need this method.
        :param data: torch or list
        :return: np.array with prediction
        """
        if not torch.is_tensor(data):
            input = torch.from_numpy(np.array(data)).float()
        else:
            input = torch.squeeze(data)
            
        output = self.forward(input).detach().numpy()

        return output


    def forward(self, x):
        outputs = self._forwardPass(x)
        return outputs


    def _forwardPass(self, x):
        # COMMENT FROM DL: I commented out the following line. Seems bugged.
        # x = x.view(x.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        x = F.softmax(x, dim=-1)  # also changed from dim=1 to dim=-1
        return x
