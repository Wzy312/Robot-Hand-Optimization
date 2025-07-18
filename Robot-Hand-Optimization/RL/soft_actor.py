from rlkit.torch.sac.policies import TanhGaussianPolicy 
from rlkit.torch.networks.mlp import Mlp, ConcatMlp
import numpy as np
from .rl_algorithm import RL_algorithm
from rlkit.torch.sac.sac import SACTrainer as SoftActorCritic_rlkit
from rlkit.torch.sac.policies import MakeDeterministic
import rlkit.torch.pytorch_util as ptu
import torch
import utils

class SoftActorCritic(RL_algorithm):
    """
        Individual network: quickly adapt to the current form
        Population network: learn general strategies across forms
    """
    def __init__(self, config, env, replay, networks):
        
   
        super().__init__(config, env, replay, networks)

  
        self._variant_pop = config['rl_algorithm_config']['algo_params_pop']
        self._variant_spec = config['rl_algorithm_config']['algo_params']

        # Initialize individual networks
        self._ind_qf1 = networks['individual']['qf1']
        self._ind_qf2 = networks['individual']['qf2']
        self._ind_qf1_target = networks['individual']['qf1_target']
        self._ind_qf2_target = networks['individual']['qf2_target']
        self._ind_policy = networks['individual']['policy']

        # Initialize the swarm network
        self._pop_qf1 = networks['population']['qf1']
        self._pop_qf2 = networks['population']['qf2']
        self._pop_qf1_target = networks['population']['qf1_target']
        self._pop_qf2_target = networks['population']['qf2_target']
        self._pop_policy = networks['population']['policy']

        self._batch_size = config['rl_algorithm_config']['batch_size']
        self._nmbr_indiv_updates = config['rl_algorithm_config']['indiv_updates']
        self._nmbr_pop_updates = config['rl_algorithm_config']['pop_updates']

   
        self._algorithm_ind = SoftActorCritic_rlkit(
            env=self._env,
            policy=self._ind_policy,
            qf1=self._ind_qf1,
            qf2=self._ind_qf2,
            target_qf1=self._ind_qf1_target,
            target_qf2=self._ind_qf2_target,
            use_automatic_entropy_tuning=False,
            **self._variant_spec
        )

        self._algorithm_pop = SoftActorCritic_rlkit(
            env=self._env,
            policy=self._pop_policy,
            qf1=self._pop_qf1,
            qf2=self._pop_qf2,
            target_qf1=self._pop_qf1_target,
            target_qf2=self._pop_qf2_target,
            use_automatic_entropy_tuning=False,
            **self._variant_pop
        )

    def episode_init(self):
        
        self._algorithm_ind = SoftActorCritic_rlkit(
            env=self._env,
            policy=self._ind_policy,
            qf1=self._ind_qf1,
            qf2=self._ind_qf2,
            target_qf1=self._ind_qf1_target,
            target_qf2=self._ind_qf2_target,
            use_automatic_entropy_tuning=False,
            **self._variant_spec
        )


        if self._config['rl_algorithm_config']['copy_from_gobal']:
            utils.copy_pop_to_ind(
                networks_pop=self._networks['population'],
                networks_ind=self._networks['individual']
            )

    def single_train_step(self, train_ind=True, train_pop=False):

        if train_ind:
            # Training individual networks
            self._replay.set_mode('species')
            for _ in range(self._nmbr_indiv_updates):
                try:
                    batch = self._replay.random_batch(self._batch_size)
                    for key, value in batch.items():
                        if isinstance(value, np.ndarray) and np.isnan(value).any():
                            print(f"warning: {key} in the batch contains NaN values")

                            raise ValueError(f"{key} in the batch contains NaN values, stop training")
                    
                    self._algorithm_ind.train(batch)
                    
                    for name, param in self._ind_policy.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                print(f"Warning: Unexpected gradient of parameter {name}")
                                raise ValueError(f"The gradient of parameter {name} is abnormal, stop training")
                                
                except Exception as e:
                    print(f"Exception caught during training: {e}")
                    import traceback
                    traceback.print_exc()
                    raise e

        if train_pop:
            # Train the group network
            self._replay.set_mode('population')
            for _ in range(self._nmbr_pop_updates):
                try:
                    batch = self._replay.random_batch(self._batch_size)
           
                    for key, value in batch.items():
                        if isinstance(value, np.ndarray) and np.isnan(value).any():
                            print(f"warning: {key} in the batch contains NaN values")
                            raise ValueError(f"{key} in the swarm network batch contains NaN value, stop training")
                            
                    self._algorithm_pop.train(batch)
 
                    for name, param in self._pop_policy.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                print(f"Warning: Abnormal gradient of swarm network parameter {name}")
                                raise ValueError(f"The gradient of the group network parameter {name} is abnormal, stop training")
                                
                except Exception as e:
                    print(f"Exception caught during swarm network training: {e}")
                    import traceback
                    traceback.print_exc()
                    raise e

    @staticmethod
    def create_networks(env, config):
     
        network_dict = {
            'individual': SoftActorCritic._create_networks(env=env, config=config),
            'population': SoftActorCritic._create_networks(env=env, config=config),
        }
        return network_dict

    @staticmethod
    def _create_networks(env, config):

        # Get state and action dimensions
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))

        # Network structure parameters
        net_size = config['rl_algorithm_config']['net_size']
        hidden_sizes = [net_size] * config['rl_algorithm_config']['network_depth']

        # Creating a Q network - using ConcatMlp to handle the concatenation of states and actions
        qf1 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=hidden_sizes,
            hidden_activation=torch.nn.ReLU(),
            output_activation=torch.nn.Identity()
        ).to(device=ptu.device)

        qf2 = ConcatMlp(
            input_size=obs_dim + action_dim, 
            output_size=1,
            hidden_sizes=hidden_sizes,
            hidden_activation=torch.nn.ReLU(),
            output_activation=torch.nn.Identity()
        ).to(device=ptu.device)

        # Create a copy of the target Q network
        qf1_target = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=hidden_sizes,
            hidden_activation=torch.nn.ReLU(),
            output_activation=torch.nn.Identity()
        ).to(device=ptu.device)

        qf2_target = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=hidden_sizes,
            hidden_activation=torch.nn.ReLU(),
            output_activation=torch.nn.Identity()
        ).to(device=ptu.device)

        # Creating a policy network
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            hidden_activation=torch.nn.ReLU(),
        ).to(device=ptu.device)

        # Gradient clipping to improve training stability
        clip_value = 1.0
        for p in qf1.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
        for p in qf2.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
        for p in policy.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

        return {
            'qf1': qf1,
            'qf2': qf2,
            'qf1_target': qf1_target,
            'qf2_target': qf2_target,
            'policy': policy
        }

    @staticmethod
    def get_q_network(networks):

        return networks['qf1']

    @staticmethod
    def get_policy_network(networks):

        return networks['policy']
