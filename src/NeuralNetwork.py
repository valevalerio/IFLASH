''' Neural Network class to use as models '''
import torch.nn as nn
import torch.nn.functional as F
# pylint: disable=invalid-name
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=unused-argument

class ModularNet(nn.Module):
    def __init__(self,input_size=4,layers = [16,16,16],num_targets=3,dropout=0.05):
        super(ModularNet, self).__init__()
        self.layers = nn.ModuleList()
        self.hidden_units = layers
        self.dropout = dropout
        layers = [input_size] + layers
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.final = nn.Linear(layers[-1], num_targets)
    def forward(self, x):
        for fc in self.layers:
            x = F.selu(fc(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final(x)
        output = F.log_softmax(x,dim=1)
        return output
