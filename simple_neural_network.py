import numpy as np

class SimpleNeuralNetwork():
    def __init__(self, state_size, action_size):
        self.fc0 = np.zeros((8, state_size))
        self.fc1 = np.zeros((action_size, 8))
        self.timestep = 0
        self.update_interval = 4

    def forward(self, x):
        x0 = self.fc0 @ x
        x1 = np.maximum(x0, 0.0)
        x2 = self.fc1 @ x1
        return x2

    def step(self, obs):
        if self.timestep % self.update_interval == 0:
            self.cached = self.forward(obs)
        self.timestep += 1
        return self.cached

    def num_parameters(self):
        return self.fc0.size + self.fc1.size

    def replace_parameters(self, param_vector):
        param_vector = np.array(param_vector)
        assert(len(param_vector.shape) == 1)
        assert(param_vector.shape[0] == self.num_parameters())
        for param in [self.fc0, self.fc1]:
            n = param.size
            param[...] = param_vector[:n].reshape(param.shape)
            param_vector = param_vector[n:]
        assert(param_vector.size == 0)