import numpy as np
import torch
import rlkit.torch.pytorch_util as ptu
import pyswarms as ps
from .design_optimization import Design_Optimization

class PSO_batch(Design_Optimization):
    """PSO batch optimizer"""
    
    def __init__(self, config, replay, env):
        """Initialize PSO batch optimizer
        
        Args:
            config: Configuration dictionary
            replay: Experience replay buffer
            env: Environment instance
        """
        super().__init__(config=config, replay=replay, env=env)
        
        # Get batch size
        self._state_batch_size = config.get('state_batch_size', 32)
        self._coadapt = None
    
    def optimize_design(self, design, q_network, policy_network):
        """Optimize design using PSO
        
        Args:
            design: Current design parameters
            q_network: Q-network for evaluation
            policy_network: Policy network for action generation
            
        Returns:
            new_design: Optimized design parameters
        """
        trials_counter = 0
        self._replay.set_mode('start')
        initial_state = self._replay.random_batch(self._state_batch_size)
        initial_state = initial_state['observations']
        design_idxs = self._env.get_design_dimensions()
        
        from rlkit.torch.sac.policies import MakeDeterministic
        deterministic_policy = MakeDeterministic(policy_network)
        
        def f_qval(x_input, **kwargs):
            """
            Fitness function for PSO. Uses Q-network to evaluate design quality.
            
            Args:
                x_input: Design parameter array with shape (n_particles, n_dimensions)
                
            Returns:
                Negative Q-values for each design (smaller is better)
            """
            nonlocal trials_counter
            
            shape = x_input.shape
            cost = np.zeros((shape[0],))
            
            trials_counter += shape[0]
            
            with torch.no_grad():
                for i in range(shape[0]):
                    # Get current design parameters
                    x = x_input[i:i+1,:]
                    
                    # Build input state batch
                    state_batch = initial_state.copy()
                    state_batch[:,design_idxs] = x
                    
                    # Collect actions for each state using deterministic policy
                    actions = []
                    for state in state_batch:
                        action, *_ = deterministic_policy.get_action(state)
                        actions.append(action)
                    actions = np.array(actions)
                    
                    network_input = torch.from_numpy(state_batch).to(device=ptu.device, dtype=torch.float32)
                    action_tensor = torch.from_numpy(actions).to(device=ptu.device, dtype=torch.float32)
                    
                    output = q_network(network_input, action_tensor)
                    loss = -output.mean().sum()
                    cost[i] = float(loss.item())
                    
            return cost
        
        # Set optimization bounds
        lower_bounds = [l for l, *_ in self._env.design_params_bounds]
        upper_bounds = [u for *_, u in self._env.design_params_bounds]
        bounds = (np.array(lower_bounds), np.array(upper_bounds))
        
        # Configure PSO optimizer
        options = {
            'c1': 0.5,  # Cognitive learning factor
            'c2': 0.4,  # Social learning factor
            'w': 0.9    # Inertia weight
        }
        
        # Create optimizer
        optimizer = ps.single.GlobalBestPSO(
            n_particles=700,           # Number of particles
            dimensions=len(design),    # Parameter dimensions
            bounds=bounds,             # Parameter bounds
            options=options            # Optimizer configuration
        )
        
        # Execute optimization
        cost, new_design = optimizer.optimize(
            f_qval,              # Fitness function
            print_step=150,      # Print interval
            iters=300,           # Number of iterations
            verbose=3            # Verbosity level
        )
        
        return new_design