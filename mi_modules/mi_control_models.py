import torch
import numpy as np
from mpi_utils.mpi_utils import sync_grads, sync_networks


class MutualInformationControlEstimator:
    def __init__(self, env_params, policy, args):
        self.batch_size = args.batch_size
        self.env_params = env_params
        self.policy = policy
        self.nb_objects = args.n_blocks
        self.cuda = args.cuda
        self.n_coord_body = 5 # position + gripper state
        self.n_coord_obj = 3

        from mi_modules.networks import TNetworkSpatialFlat
        self.T_network = TNetworkSpatialFlat(args.hidden_dim) # Mutual Information Estimator

        if self.cuda: 
            self.T_network.cuda()
        
        sync_networks(self.T_network)

        self.mi_optim = torch.optim.Adam(self.T_network.parameters(), lr=args.mi_control_lr)

        # create variables for means and covs
        self.means = None
        self.covs = None

    def set_gaussian_parameters(self, means, covs):
        self.means = means
        self.stds = covs
        
    def train(self):
        """ Train the MI neural estimator """
        trajectories = self.policy.buffer.sample_trajectories(n=self.batch_size)

        obs_unorm = trajectories['obs']

        obs = self.policy.o_norm.normalize(obs_unorm)

        obs_agent = obs[:, :, :self.n_coord_body] # (batchsize x max_timesteps x dim_body)
        obs_object = np.stack([obs[:, :, self.env_params['body'] + i * self.env_params['obj']:
                                self.env_params['body'] + i * self.env_params['obj'] + self.n_coord_obj] 
                                for i in range(self.nb_objects)]) # (n_objects x batchsize x max_timesteps x dim_obj)
        
        # Move axis in order to shuffle using numpy
        obs_agent_shuffled = np.moveaxis(obs_agent.copy(), 0, -1)
        np.random.shuffle(obs_agent_shuffled)
        obs_agent_shuffled = np.moveaxis(obs_agent_shuffled, -1, 0) # axis back 

        # Expand body observation accross all objects 
        obs_agent_shuffled = np.repeat(np.expand_dims(obs_agent_shuffled, axis=0), self.nb_objects, axis=0)
        obs_agent = np.repeat(np.expand_dims(obs_agent, axis=0), self.nb_objects, axis=0)

        joint_inp = np.concatenate([obs_agent, obs_object], axis=-1) # (n_objects x batchsize x max_timesteps x dim_obj + dim_body)
        marginal_inp = np.concatenate([obs_agent_shuffled, obs_object], axis=-1) # (n_objects x batchsize x max_timesteps x dim_obj + dim_body)

        loss = self._update_mi_estimator(self.T_network, joint_inp, marginal_inp, self.mi_optim)

        return loss
    
    def _update_mi_estimator(self, T_network, joint, marginal, mi_optim):
        """ Update the Mutual Information Estimator """
        # Tensorize 
        input_joint_tensor = torch.tensor(joint, dtype=torch.float32)
        input_marginal_tensor = torch.tensor(marginal, dtype=torch.float32)
        if self.cuda:
            input_joint_tensor = input_joint_tensor.cuda()
            input_marginal_tensor = input_marginal_tensor.cuda()

        output_joint = T_network(input_joint_tensor) # (n_objects x batchsize x max_timesteps x 1)
        output_marginal = T_network(input_marginal_tensor) # (n_objects x batchsize x max_timesteps x 1)

        exp_output_marginal = torch.exp(output_marginal)

        # mean accross timesteps
        loss = - (output_joint.mean(-2) - torch.log(exp_output_marginal.mean(-2)))

        # loss = torch.reshape(loss, (loss.shape[0] * loss.shape[1], loss.shape[2]))
        # mean accross trajectories
        loss = loss.mean(-2)

        # sum accross objects
        loss = loss.sum(0)

        # start to update the network
        mi_optim.zero_grad()
        loss.backward()
        sync_grads(T_network)
        mi_optim.step()

        return loss.item()
    
    def _compute_intrinsic_rewards(self, obs_unorm, obs_next_unorm):
        """ Use Mutual Information Estimator to compute intrinsic rewards """
        # normalize
        obs = self.policy.o_norm.normalize(obs_unorm)
        obs_next = self.policy.o_norm.normalize(obs_next_unorm)
        # Current observation
        obs_agent = obs[:, :, :self.n_coord_body] # (batch_size, trajectory_len, features)
        # obs_objects = obs[:, :, self.env_params['dim_body_features']:self.env_params['dim_body_features'] + self.env_params['dim_obj_features']]
        obs_objects = np.stack([obs[:, :, self.env_params['body'] + i * self.env_params['obj']:
                                self.env_params['body'] + i * self.env_params['obj'] + self.n_coord_obj] 
                                for i in range(self.nb_objects)]) # (n_objects x batchsize x max_timesteps x dim_obj)

        obs_agent_shuffled = obs_agent.copy()
        obs_agent_shuffled = np.transpose(obs_agent_shuffled, axes=(1, 0, 2))
        np.random.shuffle(obs_agent_shuffled)
        obs_agent_shuffled = np.transpose(obs_agent_shuffled, axes=(1, 0, 2))

        # Next observation
        obs_next_agent = obs_next[:, :, :self.n_coord_body]
        # obs_next_objects = np.concatenate([obs_next[:, 4:6], obs_next[:, 10:11]], axis=-1)
        # obs_next_objects = obs_next[:, :, self.env_params['dim_body_features']:self.env_params['dim_body_features'] + self.env_params['dim_obj_features']]
        obs_next_objects = np.stack([obs_next[:, :, self.env_params['body'] + i * self.env_params['obj']:
                               self.env_params['body'] + i* self.env_params['obj'] + self.n_coord_obj] 
                               for i in range(self.nb_objects)])

        obs_next_agent_shuffled = obs_next_agent.copy()
        obs_next_agent_shuffled = np.transpose(obs_next_agent_shuffled, axes=(1, 0, 2))
        np.random.shuffle(obs_next_agent_shuffled)
        obs_next_agent_shuffled = np.transpose(obs_next_agent_shuffled, axes=(1, 0, 2))

        # Expand body observation accross all objects 
        obs_agent_shuffled = np.repeat(np.expand_dims(obs_agent_shuffled, axis=0), self.nb_objects, axis=0)
        obs_agent = np.repeat(np.expand_dims(obs_agent, axis=0), self.nb_objects, axis=0)
        obs_next_agent_shuffled = np.repeat(np.expand_dims(obs_next_agent_shuffled, axis=0), self.nb_objects, axis=0)
        obs_next_agent = np.repeat(np.expand_dims(obs_next_agent, axis=0), self.nb_objects, axis=0)

        # Infer distributions
        joint_inp = np.concatenate([obs_agent, obs_objects], axis=-1)
        marginal_inp = np.concatenate([obs_agent_shuffled, obs_objects], axis=-1)

        joint_next_inp = np.concatenate([obs_next_agent, obs_next_objects], axis=-1)
        marginal_next_inp = np.concatenate([obs_next_agent_shuffled, obs_next_objects], axis=-1)

        pair_joint_inp = np.stack([joint_inp[:, :, :-1, :], joint_next_inp], axis=-2)
        pair_marginal_inp = np.stack([marginal_inp[:, :, :-1, :], marginal_next_inp], axis=-2)

        input_joint_tensor = torch.tensor(pair_joint_inp, dtype=torch.float32)
        input_marginal_tensor = torch.tensor(pair_marginal_inp, dtype=torch.float32)

        if self.cuda:
            input_joint_tensor = input_joint_tensor.cuda()
            input_marginal_tensor = input_marginal_tensor.cuda()

        # flatten
        # input_joint_tensor = torch.reshape(input_joint_tensor, (input_joint_tensor.shape[0] * input_joint_tensor.shape[1], 
        #                                                       input_joint_tensor.shape[2], input_joint_tensor.shape[3]))

        # input_joint_tensor  = torch.permute(input_joint_tensor, (0, 2, 1))

        # input_marginal_tensor = torch.reshape(input_marginal_tensor, (input_marginal_tensor.shape[0] * input_marginal_tensor.shape[1], 
        #                                                       input_marginal_tensor.shape[2], input_marginal_tensor.shape[3]))

        # input_marginal_tensor  = torch.permute(input_marginal_tensor, (0, 2, 1))

        with torch.no_grad():
            output_joint = self.T_network(input_joint_tensor)
            output_marginal = self.T_network(input_marginal_tensor)

            exp_output_marginal = torch.exp(output_marginal)

            r = output_joint.mean(-2) - torch.log(exp_output_marginal.mean(-2))

            # r = torch.exp(r) - 1

            # r = torch.clamp(100*r, min=0, max=1)

            # sum accross objects when performing optimization of intrinsic rewards
            r = r.sum(0)
            # mutual information is non-negative
            # r = torch.relu(r)
        
        
        return r

    def compute_mutual_information(self, o):
        """ For experiment 2 """
        # o = np.expand_dims(o, axis=0)
        o_next = o[:, 1:, :]
        
        if self.cuda:
            return self._compute_intrinsic_rewards(o, o_next).cpu().numpy()
        else:
            return self._compute_intrinsic_rewards(o, o_next).numpy()


    def save(self, model_path, n):
        """ Save the model  """
        model_name = 'estimator_{}_{}_obj.pt'.format(self.architecture, n)
        torch.save([self.T_network.state_dict()], model_path + '/' + model_name)
    
    def load(self, model_path):
        """ Load model from path """
        m = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.T_network.load_state_dict(m[0])