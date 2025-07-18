from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
import numpy as np

class EvoReplayLocalGlobalStart(ReplayBuffer):
    """
    Supports experience playback buffer for morphological optimization. Maintains three independent buffers:
1. species_buffer: stores experience of the current morphology
2. population_buffer: stores experience of all morphologies
3. init_state_buffer: stores initial state for design evaluation
    """
    def __init__(self, env, max_replay_buffer_size_species, max_replay_buffer_size_population):
        """          
            env: 
            max_replay_buffer_size_species: individual buffer size
            max_replay_buffer_size_population: population buffer size
        """
        self._species_buffer = EnvReplayBuffer(
            env=env, 
            max_replay_buffer_size=max_replay_buffer_size_species
        )
        self._population_buffer = EnvReplayBuffer(
            env=env, 
            max_replay_buffer_size=max_replay_buffer_size_population
        )
        self._init_state_buffer = EnvReplayBuffer(
            env=env, 
            max_replay_buffer_size=max_replay_buffer_size_population
        )
        
        self._env = env
        self._max_replay_buffer_size_species = max_replay_buffer_size_species
        self._mode = "species"  
        self._ep_counter = 0
        self._expect_init_state = True
        
        print("Using EvoReplayLocalGlobalStart replay buffer")

    def add_sample(self, observation, action, reward, next_observation, terminal, **kwargs):

  
        print(f"[REPLAY DEBUG] Storage Rewards: {reward}")
        
        
        self._species_buffer.add_sample(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            env_info={},
            **kwargs
        )
        
    
        if self._expect_init_state:
            self._init_state_buffer.add_sample(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminal=terminal,
                env_info={},
                **kwargs
            )
            self._init_state_buffer.terminate_episode()
            self._expect_init_state = False
            
      
        if self._ep_counter >= 0:
            self._population_buffer.add_sample(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminal=terminal,
                env_info={},
                **kwargs
            )
    def terminate_episode(self):
        """ Finish current episode"""
        if self._mode == "species":
            self._species_buffer.terminate_episode()
            self._population_buffer.terminate_episode()
            self._ep_counter += 1
            self._expect_init_state = True

    def num_steps_can_sample(self, **kwargs):
       
        if self._mode == "species":
            return self._species_buffer.num_steps_can_sample(**kwargs)
        elif self._mode == "population":
            return self._population_buffer.num_steps_can_sample(**kwargs)
        else:
            raise ValueError(f"Unknown mode: {self._mode}")

    def random_batch(self, batch_size):
   
        if self._mode == "species":
            # Mixed individual and group experiences
            species_batch_size = int(np.floor(batch_size * 0.9))
            pop_batch_size = int(np.ceil(batch_size * 0.1))
            
            pop = self._population_buffer.random_batch(pop_batch_size)
            spec = self._species_buffer.random_batch(species_batch_size)
            
            # Combining two parts of experience
            for key, item in pop.items():
                pop[key] = np.concatenate([pop[key], spec[key]], axis=0)
            return pop
            
        elif self._mode == "population":
            return self._population_buffer.random_batch(batch_size)
            
        elif self._mode == "start":
            return self._init_state_buffer.random_batch(batch_size)
            
        else:
            raise ValueError(f"Unknown mode: {self._mode}")

    def set_mode(self, mode):
        """Set the sampling mode"""
        if mode in ["species", "population", "start"]:
            self._mode = mode
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def reset_species_buffer(self):
 
        self._species_buffer = EnvReplayBuffer(
            env=self._env,
            max_replay_buffer_size=self._max_replay_buffer_size_species
        )
        self._ep_counter = 0
