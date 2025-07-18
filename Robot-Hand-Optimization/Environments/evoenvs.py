from gym import spaces
import numpy as np
from .pybullet_evo.gym_locomotion_envs import AllegroHandBulletEnv
import copy
from utils import BestEpisodesVideoRecorder

class AllegroHandEnv(object):
  
    
    def __init__(self, config={'env': {'render': True, 'record_video': False}}):
        self._config = config
        self._render = self._config['env']['render']
        self._record_video = self._config['env']['record_video']
        self._current_design = [1.0] * 28
        
        # Initialize design parameters
        self._current_design = [1.0] * 28  
        self._config_numpy = np.array(self._current_design)      
        self.design_params_bounds = [(0.8, 1.2)] * 28
        self._env = AllegroHandBulletEnv(render=self._render, design=self._current_design)
        self.init_sim_params = [
            [1.0] * 28, 
            [np.random.uniform(0.8, 1.2) for _ in range(28)],
            [np.random.uniform(0.8, 1.2) for _ in range(28)],
            [np.random.uniform(0.8, 1.2) for _ in range(28)],
            [np.random.uniform(0.8, 1.2) for _ in range(28)]
        ]
        
        self.observation_space = spaces.Box(
            -np.inf, np.inf, 
            shape=[self._env.observation_space.shape[0] + 28],
            dtype=np.float32
        )
        self.action_space = self._env.action_space
        if self._record_video:
            self._video_recorder = BestEpisodesVideoRecorder(
                path=config['data_folder_experiment'],
                max_videos=5
            )

        self._design_dims = list(range(
            self.observation_space.shape[0] - len(self._current_design),
            self.observation_space.shape[0]
        ))

    def step(self, a):

        a = a.astype(np.float32) if isinstance(a, np.ndarray) else a        
        state, reward, done, info = self._env.step(a) 
        state = state.astype(np.float32)
        state = np.append(state, self._config_numpy.astype(np.float32))
        
        if self._record_video:
            self._video_recorder.step(
                env=self._env,
                state=state,
                reward=reward,
                done=done
            )
        
        return state, reward, done, info

    def reset(self):

        state = self._env.reset()
        state = np.append(state, self._config_numpy)
        
        if self._record_video:
            self._video_recorder.reset(
                env=self._env,
                state=state,
                reward=0,
                done=False
            )
            
        return state
        
    def set_new_design(self, design):

        self._env.reset_design(design)
        self._current_design = design
        self._config_numpy = np.array(design)
        if self._record_video:
            self._video_recorder.increase_folder_counter()


    def get_random_design(self):
 
        return np.random.uniform(0.8, 1.2, size=28)


    def get_current_design(self):
  
        return copy.copy(self._current_design)

    def get_design_dimensions(self):
   
        return copy.copy(self._design_dims)
