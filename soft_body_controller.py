import numpy as np

class MuscleController():
    def __init__(self, hidden_state_dim, joint_indices, adjacent_muscle_indices):
        self.joint_indices = joint_indices
        self.adjacent_muscle_indices = adjacent_muscle_indices
        self.hidden_state_dim = hidden_state_dim
        self.previous_state = np.zeros((hidden_state_dim))
        self.current_state = np.zeros((2 * hidden_state_dim))
        self.obs_mapping = np.zeros((hidden_state_dim, len(self.joint_indices)))
        self.neighbour_weights = np.zeros((len(self.adjacent_muscle_indices)))
        self.act_mapping = np.zeros((3, 2 * hidden_state_dim))

    def get_joint_indices(self):
        return self.joint_indices

    def get_adjacent_muscle_indices(self):
        return self.adjacent_muscle_indices

    def observe(self, local_joint_state, previous_adjacent_states):
        self.current_state[:self.hidden_state_dim] = self.obs_mapping @ local_joint_state
        self.current_state[self.hidden_state_dim:] = self.neighbour_weights @ previous_adjacent_states

    def reset(self):
        self.current_state[...] = 0.0
        self.previous_state[...] = 0.0

    def get_previous_state(self):
        return self.previous_state

    def make_action(self):
        return self.act_mapping @ self.current_state

    def swap_state(self):
        self.previous_state = self.current_state[:self.hidden_state_dim]
        self.current_state[...] = 0.0

    def num_parameters(self):
        return self.obs_mapping.size + self.neighbour_weights.size + self.act_mapping.size

    def replace_parameters(self, param_vector):
        param_vector = np.array(param_vector)
        assert(len(param_vector.shape) == 1)
        assert(param_vector.shape[0] == self.num_parameters())
        for param in [self.obs_mapping, self.neighbour_weights, self.act_mapping]:
            n = param.size
            param[...] = param_vector[:n].reshape(param.shape)
            param_vector = param_vector[n:]
        assert(param_vector.size == 0)



class SoftBodyController():
    def __init__(self, the_soft_body, hidden_state_dim):
        self.muscle_controllers = []
        for i in range(the_soft_body.num_muscles()):
            joint_indices = the_soft_body.get_muscle(i).get_joint_indices()
            adjacent_muscle_indices = the_soft_body.get_adjacent_muscles(i)
            self.muscle_controllers.append(MuscleController(hidden_state_dim, joint_indices, adjacent_muscle_indices))
        self.n_joints = the_soft_body.num_edges()
            
    def step(self, obs):
        assert(len(obs.shape) == 1)
        assert(obs.shape[0] == (self.n_joints))

        # obs_matrix = obs.reshape(self.n_joints, 4)

        for msc in self.muscle_controllers:
            msc.swap_state()

        for msc in self.muscle_controllers:
            local_joint_state = obs[msc.get_joint_indices()].reshape(-1)
            adjacent_muscle_state = np.array([ self.muscle_controllers[i].get_previous_state() for i in msc.get_adjacent_muscle_indices() ])
            msc.observe(local_joint_state, adjacent_muscle_state)

        muscle_actions = []

        for msc in self.muscle_controllers:
            muscle_actions.append(msc.make_action())

        return np.array(muscle_actions).reshape(-1)

    def num_parameters(self):
        n = 0
        for msc in self.muscle_controllers:
            n += msc.num_parameters()
        return n

    def replace_parameters(self, param_vector):
        assert(len(param_vector.shape) == 1)
        assert(param_vector.shape[0] == self.num_parameters())
        for msc in self.muscle_controllers:
            n = msc.num_parameters()
            msc.replace_parameters(param_vector[:n])
            param_vector = param_vector[n:]
        assert(param_vector.size == 0)

    def reset(self):
        for msc in self.muscle_controllers:
            msc.reset()

