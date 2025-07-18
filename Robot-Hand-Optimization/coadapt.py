from RL.soft_actor import SoftActorCritic
from DO.pso_batch import PSO_batch
from DO.pso_sim import PSO_simulation
from Environments import evoenvs
import utils
import time
from RL.evoreplay import EvoReplayLocalGlobalStart
import numpy as np
import os
import csv
import torch

def select_design_opt_alg(alg_name):

    if alg_name == "pso_batch":
        return PSO_batch
    elif alg_name == "pso_sim":
        return PSO_simulation
    else:
        raise ValueError("Design Optimization method not found.")

def select_environment(env_name):
  
    if env_name == 'HalfCheetah':
        return evoenvs.HalfCheetahEnv
    elif env_name == 'AllegroHand':
        return evoenvs.AllegroHandEnv
    else:
        raise ValueError("Environment class not found.")

def select_rl_alg(rl_name):

    if rl_name == 'SoftActorCritic':
        return SoftActorCritic
    else:
        raise ValueError('RL method not fund.')

class Coadaptation(object):
    """Main class for co-optimization of morphology and behavior"""

    def __init__(self, config):
 
        self._config = config
        utils.move_to_cuda(self._config)
        self._episode_length = self._config['steps_per_episodes']
        self._reward_scale = 1.0

        # Create environment
        self._env_class = select_environment(self._config['env']['env_name'])
        self._env = self._env_class(config=self._config)

        # Create experience replay buffer
        self._replay = EvoReplayLocalGlobalStart(
            self._env,
            max_replay_buffer_size_species=int(1e6),
            max_replay_buffer_size_population=int(1e7)
        )

        # Create RL algorithm
        self._rl_alg_class = select_rl_alg(self._config['rl_method'])
        self._networks = self._rl_alg_class.create_networks(env=self._env, config=config)
        self._rl_alg = self._rl_alg_class(
            config=self._config,
            env=self._env,
            replay=self._replay,
            networks=self._networks
        )

        # Create design optimizer
        self._do_alg_class = select_design_opt_alg(self._config['design_optim_method'])
        self._do_alg = self._do_alg_class(config=self._config, replay=self._replay, env=self._env)
        
        utils.move_to_cuda(self._config)
        self._last_single_iteration_time = 0
        self._design_counter = 0
        self._episode_counter = 0
        self._data_design_type = 'Initial'
        

    def initialize_episode(self):
        """Initialize episode"""
        if hasattr(self, '_data_rewards'):
            print(f"[INIT DEBUG] _data_rewards before reset: {self._data_rewards}, length: {len(self._data_rewards)}")
        
        self._rl_alg.episode_init()
        self._replay.reset_species_buffer()

        self._data_rewards = []

        print(f"[INIT DEBUG] _data_rewards after reset: {self._data_rewards}, length: {len(self._data_rewards)}")
        
        self._episode_counter = 0

    def single_iteration(self):
        """Execute single training step"""
        print("Time for one iteration: {}".format(time.time() - self._last_single_iteration_time))
        self._last_single_iteration_time = time.time()
        
        # Collect training data
        self._replay.set_mode("species")
        self.collect_training_experience()
        
        # Train networks
        train_pop = self._design_counter > 3
        if self._episode_counter >= self._config['initial_episodes']:
            self._rl_alg.single_train_step(train_ind=True, train_pop=train_pop)
        
        self._episode_counter += 1
        self.execute_policy()
        self.save_logged_data()
        self.save_networks()

    def collect_training_experience(self):
        """ Collect training data.

        This function executes a single episode in the environment using the
        exploration strategy/mechanism and the policy.
        The data, i.e. state-action-reward-nextState, is stored in the replay
        buffer.

        Modification: Add reward tracking debug information
        """
        state = self._env.reset()
        nmbr_of_steps = 0
        done = False

        # Select policy network - similar to execute_policy
        if self._episode_counter < self._config['initial_episodes']:
            policy_gpu_ind = self._rl_alg_class.get_policy_network(self._networks['population'])
        else:
            policy_gpu_ind = self._rl_alg_class.get_policy_network(self._networks['individual'])
        
        # Maintain randomness during training data collection, don't use deterministic policy
        self._policy_cpu = policy_gpu_ind

        # Use safe way to get config item
        if self._config.get('use_cpu_for_rollout',False):
            utils.move_to_cpu()
        else:
            utils.move_to_cuda(self._config)

        # Create list specifically for storing samples for batch adding
        episode_samples = []

        while not(done) and nmbr_of_steps <= self._episode_length:
            nmbr_of_steps += 1
            
            # Use policy to get action, completely rely on original policy
            action, _ = self._policy_cpu.get_action(state)            
            new_state, reward, done, info = self._env.step(action)            
            # print(f"[COADAPT DEBUG] Got Reward: {reward}")
            scaled_reward = float(reward) * self._reward_scale
            reward = np.array([scaled_reward], dtype=np.float32)          
            terminal = np.array([done], dtype=np.float32)
            sample = {
                'observation': state,
                'action': action,
                'reward': reward,
                'next_observation': new_state,
                'terminal': terminal
            }
            episode_samples.append(sample)
            state = new_state
        for sample in episode_samples:
            self._replay.add_sample(
                observation=sample['observation'],
                action=sample['action'],
                reward=sample['reward'],
                next_observation=sample['next_observation'],
                terminal=sample['terminal']
            )
            
        self._replay.terminate_episode()
        if self._config.get('use_cpu_for_rollout', False):
            utils.move_to_cuda(self._config)    
            
    def execute_policy(self):
        """Evaluate current policy"""
        state = self._env.reset()
        done = False
        reward_ep = 0.0
        reward_original = 0.0
        action_cost = 0.0
        nmbr_of_steps = 0

        # Select policy network
        if self._episode_counter < self._config['initial_episodes']:
            policy = self._rl_alg_class.get_policy_network(self._networks['population'])
        else:
            policy = self._rl_alg_class.get_policy_network(self._networks['individual'])

        from rlkit.torch.sac.policies import MakeDeterministic
        deterministic_policy = MakeDeterministic(policy)
        self._policy_cpu = deterministic_policy  

        if self._config.get('use_cpu_for_rollout', False):
            utils.move_to_cpu()
        else:
            utils.move_to_cuda(self._config)
 
        while not(done) and nmbr_of_steps <= self._episode_length:
            nmbr_of_steps += 1
            action, _ = self._policy_cpu.get_action(state)
            new_state, reward, done, info = self._env.step(action)
            action_cost += info.get('orig_action_cost', 0)
            reward_ep += float(reward)
            reward_original += float(info.get('orig_reward', 0))
            state = new_state
            
        if self._config.get('use_cpu_for_rollout', False):
            utils.move_to_cuda(self._config)
    
        # [New] Add reward debug information
        print(f"[EXECUTE DEBUG] Episode reward: {reward_ep}, Original reward: {reward_original}")
                
        self._data_rewards.append(reward_ep)
        
        # [New] Confirm reward has been added to list
        print(f"[EXECUTE DEBUG] _data_rewards after adding: {self._data_rewards}, length: {len(self._data_rewards)}")

    def save_networks(self):
        """Save network weights"""
        if not self._config['save_networks']:
            return

        checkpoints_pop = {}
        for key, net in self._networks['population'].items():
            checkpoints_pop[key] = net.state_dict()

        checkpoints_ind = {}
        for key, net in self._networks['individual'].items():
            checkpoints_ind[key] = net.state_dict()

        checkpoint = {
            'population': checkpoints_pop,
            'individual': checkpoints_ind,
        }
        
        file_path = os.path.join(self._config['data_folder_experiment'], 'checkpoints')
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        torch.save(checkpoint, os.path.join(file_path, f'checkpoint_design_{self._design_counter}.chk'))

    def save_logged_data(self):
        """Save training logs"""
        file_path = self._config['data_folder_experiment']
        current_design = self._env.get_current_design()

        with open(os.path.join(file_path, f'data_design_{self._design_counter}.csv'), 'w') as fd:
            cwriter = csv.writer(fd)
            cwriter.writerow(['Design Type:', self._data_design_type])
            cwriter.writerow(current_design)
            cwriter.writerow(self._data_rewards)

    def run(self):
        """Run complete training process"""
        iterations_init = self._config['iterations_init']
        iterations = self._config['iterations']
        design_cycles = self._config['design_cycles']
        exploration_strategy = self._config['exploration_strategy']

        self._intial_design_loop(iterations_init)
        self._training_loop(iterations, design_cycles, exploration_strategy)

    def _training_loop(self, iterations, design_cycles, exploration_strategy):
        """Main training loop
        
        Args:
            iterations: Training rounds per design
            design_cycles: Number of design optimization cycles
            exploration_strategy: Exploration strategy
        """
        self.initialize_episode()
        initial_state = self._env._env.reset()

        self._data_design_type = 'Optimized'

        # Initialize design parameters
        optimized_params = self._env.get_random_design()
        q_network = self._rl_alg_class.get_q_network(self._networks['population'])
        policy_network = self._rl_alg_class.get_policy_network(self._networks['population'])
        

        
        optimized_params = self._do_alg.optimize_design(optimized_params, q_network, policy_network)
        optimized_params = list(optimized_params)       
        
        
        print(f"\n[TRAINING DEBUG] Design cycle initialization, self._data_rewards: {self._data_rewards}")
        initial_reward = np.mean(self._data_rewards) if self._data_rewards else 0
        print(f"[TRAINING DEBUG] Initial average reward: {initial_reward}")
        
        # Main loop
        for i in range(design_cycles):
            self._design_counter += 1
            self._env.set_new_design(optimized_params)

            # Reinforcement learning phase
            for _ in range(iterations):
                self.single_iteration()

            print(f"\n[TRAINING DEBUG] Design cycle {i} completed")
            print(f"[TRAINING DEBUG] self._data_rewards: {self._data_rewards}")
            print(f"[TRAINING DEBUG] Reward statistics - Mean: {np.mean(self._data_rewards) if len(self._data_rewards) > 0 else 'N/A'}, Max: {np.max(self._data_rewards) if len(self._data_rewards) > 0 else 'N/A'}, Min: {np.min(self._data_rewards) if len(self._data_rewards) > 0 else 'N/A'}, Length: {len(self._data_rewards)}")
            print(f"\n[DESIGN DEBUG] ========== Design cycle {i}, Design ID {self._design_counter} ==========")
            print(f"[DESIGN DEBUG] Design parameters ({len(optimized_params)}): {optimized_params}")
            
            self._env.set_new_design(optimized_params)
            # Design optimization phase
            if i % 2 == 1:
                self._data_design_type = 'Optimized'
                q_network = self._rl_alg_class.get_q_network(self._networks['population'])
                policy_network = self._rl_alg_class.get_policy_network(self._networks['population'])
                
                # Record pre-optimization reward
                pre_opt_reward = np.mean(self._data_rewards) if self._data_rewards else 0
                print(f"[TRAINING DEBUG] Average reward for recording: {pre_opt_reward}")
                               
                optimized_params = self._do_alg.optimize_design(optimized_params, q_network, policy_network)
                optimized_params = list(optimized_params)
            else:
                self._data_design_type = 'Random'
                optimized_params = self._env.get_random_design()
                optimized_params = list(optimized_params)
                
                # Debug random design reward
                random_reward = np.mean(self._data_rewards) if self._data_rewards else 0
                print(f"[TRAINING DEBUG] Random design average reward: {random_reward}")
                
                
            self.initialize_episode()

    def _intial_design_loop(self, iterations):
        """Initial design training loop
        
        Args:
            iterations: Number of training rounds
        """
        self._data_design_type = 'Initial'
        for params in self._env.init_sim_params:
            self._design_counter += 1
            self._env.set_new_design(params)
            self.initialize_episode()

            # Reinforcement learning phase
            for _ in range(iterations):
                self.single_iteration()