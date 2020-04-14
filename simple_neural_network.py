import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc0 = nn.Linear(state_size, 8)
        self.fc1 = nn.Linear(8, action_size)

    def forward(self, x):
        x = x.float()
        x = nn.ReLU()(self.fc0(x))
        x = nn.ReLU()(self.fc1(x))
        return x

    def num_parameters(self):
        n = 0
        for param_name, param in self.state_dict().items():
            if not "weight" in param_name:
                continue
            n += param.view(-1).shape[0]
        return n

    def replace_parameters(self, param_vector):
        param_vector = torch.tensor(param_vector)
        # https://stackoverflow.com/a/49448065
        assert(len(param_vector.shape) == 1)
        assert(param_vector.shape[0] == self.num_parameters())
        state_dict = self.state_dict()
        for param_name, param in state_dict.items():
            if not "weight" in param_name:
                continue
            n = param.view(-1).shape[0]
            new_param = param_vector[:n].view(param.shape)
            param_vector = param_vector[n:]
            state_dict[param_name].copy_(new_param)
        assert(param_vector.shape[0] == 0)