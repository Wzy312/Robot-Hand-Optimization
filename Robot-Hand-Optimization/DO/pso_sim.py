import numpy as np
import torch
import rlkit.torch.pytorch_util as ptu
import pyswarms as ps
from .design_optimization import Design_Optimization

class PSO_simulation(Design_Optimization):
    """Simulation-based PSO optimizer"""
    
    def __init__(self, config, replay, env):
        """Initialize simulation-based PSO optimizer
        
        Args:
            config: Configuration dictionary
            replay: Experience replay buffer
            env: Environment instance
        """
        super().__init__(config=config, replay=replay, env=env)
        
        # Get simulation parameters
        self._episode_length = config['rl_algorithm_config']['algo_params'].get('num_steps_per_epoch', 1000)
        self._reward_scale = config['rl_algorithm_config']['algo_params'].get('reward_scale', 1.0)
    
    def optimize_design(self, design, q_network, policy_network):
        """
        PSO optimization using simulation evaluation
        
        Args:
            design: Current design parameters
            q_network: Not used, kept for interface consistency
            policy_network: Policy network for action generation
            
        Returns:
            Optimized design parameters
        """
        def get_reward_for_design(design):
            """Evaluate a design in the environment"""
            self._env.set_new_design(design)
            state = self._env.reset()
            reward_episode = []
            done = False
            nmbr_of_steps = 0
            
            # Execute a complete episode
            while not(done) and nmbr_of_steps <= self._episode_length:
                nmbr_of_steps += 1
                action, *_ = policy_network.get_action(state, deterministic=True)
                new_state, reward, done, info = self._env.step(action)
                reward = reward * self._reward_scale
                reward_episode.append(float(reward))
                state = new_state
                
            # Return average reward
            reward_mean = np.mean(reward_episode)
            return reward_mean
        
        def f_qval(x_input, **kwargs):
            """PSO fitness function"""
            shape = x_input.shape
            cost = np.zeros((shape[0],))
            for i in range(shape[0]):
                x = x_input[i,:]
                reward = get_reward_for_design(x)
                cost[i] = -reward  # Convert to minimization problem
            return cost
        
        # Set optimization bounds
        lower_bounds = [l for l, *_ in self._env.design_params_bounds]
        lower_bounds = np.array(lower_bounds)
        upper_bounds = [u for *_, u in self._env.design_params_bounds]
        upper_bounds = np.array(upper_bounds)
        bounds = (lower_bounds, upper_bounds)
        
        # Configure PSO
        options = {
            'c1': 0.5,
            'c2': 0.3,
            'w': 0.9
        }
        
        # Use fewer particles and iterations since each evaluation requires simulation
        optimizer = ps.single.GlobalBestPSO(
            n_particles=35,            # Reduced number of particles
            dimensions=len(design),
            bounds=bounds,
            options=options
        )
        
        # Execute optimization
        cost, new_design = optimizer.optimize(
            f_qval,
            print_step=100,
            iters=30,                  # Reduced number of iterations
            verbose=3
        )
        
        return new_design