import sys
import os
import ray
import time
import random
from collections import deque

import logging
from vizdoom import DoomGame, Button, ScreenFormat, GameVariable
from PIL import Image
import gym
import numpy as np
from gym.spaces import Box, Dict
from ray.rllib.utils.annotations import override

default_reward_values = {
    'BASE_REWARD': 0.,
    'DISTANCE': 0.0005,
    'KILL': 5.,
    'DEATH': -5.,
    'SUICIDE': -5.,
    'MEDIKIT': 1.,
    'ARMOR': 1.,
    'INJURED': -1.,
    'WEAPON': 1.,
    'AMMO': 1.,
    'USE_AMMO': -0.2,
    'LIVING': 0.05,
    'STANDSTILL': -0.15,
}

game_variables = {

    'frag_count': GameVariable.FRAGCOUNT,
    'score': GameVariable.USER2,
    'visible': GameVariable.USER1,
    'health': GameVariable.HEALTH,
    'armor': GameVariable.ARMOR,
    'sel_ammo': GameVariable.SELECTED_WEAPON_AMMO,
    'position_x': GameVariable.POSITION_X,
    'position_y': GameVariable.POSITION_Y,
    'position_z': GameVariable.POSITION_Z

}


class VizDoomGym(gym.Env):
    """
    Wraps a VizDoom environment
    """

    def __init__(self, mission_file, scaled_resolution):

        self._init(mission_file, scaled_resolution)
        self.reset()

    def _init(self, mission_file, scaled_resolution):
        """
        :param mission_file: name of the mission (.cfg) to run,
        :param scaled_resolution: resolution (height, width) of the video frames
                                  to run training on
        """

        super(VizDoomGym, self).__init__()
        self.mission_file = mission_file
        self._logger = logging.getLogger(__name__)
        self._logger.info("Creating environment: VizDoom (%s)", self.mission_file)

        self.deathmatch = True
        # distance we need the agent to travel per time-step, otherwise we penalise
        self.distance_threshold = 15

        assert not ((self.curriculum is True) and (self.increase_diff is True)), "Don't have both true!"

        self.prev_properties = None
        self.properties = None

        self.cum_kills = np.array([0])

        # Create an instace on VizDoom game, initalise it from a scenario config file
        self.env = DoomGame()
        self.env.load_config(self.mission_file)
        self.env.set_window_visible(False)
        self.env.set_screen_format(ScreenFormat.RGB24)
        if self.deathmatch:
            self.env.add_game_args("-deathmatch")
        self.curr_skill = 1
        self.map_level = 1
        if self.increase_diff:
            self.env.set_doom_skill(self.curr_skill)
        else:
            self.env.set_doom_skill(4)
        self.env.init()

        # Perform config validation:
        # Only RGB format with a seperate channel per colour is supported
        # print(self.env.get_screen_format())
        assert self.env.get_screen_format() == ScreenFormat.RGB24
        # Only discrete actions are supported (no delta actions)
        self.available_actions = self.env.get_available_buttons()
        not_supported_actions = [Button.LOOK_UP_DOWN_DELTA, Button.TURN_LEFT_RIGHT_DELTA,
                                 Button.MOVE_LEFT_RIGHT_DELTA, Button.MOVE_UP_DOWN_DELTA,
                                 Button.MOVE_FORWARD_BACKWARD_DELTA]
        # print(available_actions)
        assert len((set(self.available_actions) - set(not_supported_actions))) == len(self.available_actions)

        self.metadata['render_modes'] = ['rgb_array']

        # Allow only one button to be pressed at a given step
        self.action_space = gym.spaces.Discrete(self.env.get_available_buttons_size() - 1)

        self.rows = scaled_resolution[0]
        self.columns = scaled_resolution[1]
        self.observation_space = gym.spaces.Box(low=0.0,
                                                high=1.0,
                                                shape=(self.rows, self.columns, 3),
                                                dtype=np.float32)

        self._rgb_array = None
        self.steps = 0
        self.global_steps = 0
        self.time_start = time.monotonic()
        self._action_frame_repeat = action_frame_repeat

    def _process_image(self, img):
        # PIL resize has indexing opposite to numpy array
        img = np.array(Image.fromarray(img).resize((self.columns, self.rows)))
        img = img.astype(np.float32)
        img = img / 255.0
        return img

    def update_game_variables(self):
        """
        Check and update game variables.
        """
        # read game variables
        new_v = {k: self.env.get_game_variable(v) for k, v in game_variables.items()}
        assert all(v.is_integer() or k[-2:] in ['_x', '_y', '_z'] for k, v in new_v.items())
        new_v = {k: (int(v) if v.is_integer() else float(v)) for k, v in new_v.items()}
        health = new_v['health']
        armor = new_v['armor']

        # check game variables
        assert 0 <= health <= 200 or health < 0 and self.env.is_player_dead()
        assert 0 <= armor <= 200, (health, armor)

        # update actor properties
        self.prev_properties = self.properties
        self.properties = new_v

    def update_reward(self):
        """
        Update reward.
        """

        # we need to know the current and previous properties
        assert self.prev_properties is not None and self.properties is not None

        reward = 0

        # kill
        d = self.properties['score'] - self.prev_properties['score']
        if d > 0:
            self.cum_kills += d
            reward += d * default_reward_values['KILL']

        # death
        if self.env.is_player_dead():
            reward += default_reward_values['DEATH']

        # suicide
        if self.properties['frag_count'] < self.prev_properties['frag_count']:
            reward += default_reward_values['SUICIDE']

        # found / lost health
        d = self.properties['health'] - self.prev_properties['health']
        if d != 0:
            if d > 0:
                reward += default_reward_values['MEDIKIT']
            else:
                reward += default_reward_values['INJURED']

        # found / lost armor
        d = self.properties['armor'] - self.prev_properties['armor']
        if d != 0:
            if d > 0:
                reward += default_reward_values['ARMOR']

        # found / lost ammo
        d = self.properties['sel_ammo'] - self.prev_properties['sel_ammo']
        if d != 0:
            if d > 0:
                reward += default_reward_values['AMMO']
            else:
                reward += default_reward_values['USE_AMMO']

        # distance
        # turn_left = (Button.TURN_LEFT == self.available_actions[action])
        # turn_right = (Button.TURN_RIGHT == self.available_actions[action])
        # if not (turn_left or turn_right):
        diff_x = self.properties['position_x'] - self.prev_properties['position_x']
        diff_y = self.properties['position_y'] - self.prev_properties['position_y']
        distance = np.sqrt(diff_x ** 2 + diff_y ** 2)
        if distance > self.distance_threshold:
            reward += default_reward_values['DISTANCE'] * distance
        else:
            reward += default_reward_values['STANDSTILL']

        # living
        reward += default_reward_values['LIVING']

        return reward

    def increase_difficulty(self):
        self.curr_skill += 1
        self.env.close()
        self.env.set_doom_skill(self.curr_skill)
        self.env.init()
        print('changing skill to', self.curr_skill)

    def update_map(self):
        self.map_level += 1
        map_str = 'map0' + str(self.map_level)
        # go with initial wad file if there's still maps on it
        self.env.close()
        self.env.set_doom_map(map_str)
        self.env.init()

    def _reset(self):
        """Reset environment"""
        # if (self.global_steps % 100000 == 0) & (self.global_steps != 0):
        if self.increase_diff or self.curriculum:
            curr_time = time.monotonic()
            if curr_time - self.time_start > 3600:
                self.time_start = curr_time
                if (self.curr_skill < 4) & self.increase_diff:
                    self.increase_difficulty()
                elif (self.map_level < 2) & self.curriculum:
                    self.update_map()
        self.steps = 0
        self.cum_kills = np.array([0])
        self.prev_properties = None
        self.properties = None
        self.env.new_episode()
        self._rgb_array = self.env.get_state().screen_buffer
        observation = self._process_image(self._rgb_array)
        return observation

    def reset(self):
        observation = self._reset()
        return observation

    def _step(self, action):
        """Take step"""
        one_hot_action = np.zeros(self.action_space.n, dtype=int)
        one_hot_action[action] = 1

        # ALWAYS SPRINTING
        one_hot_action = np.append(one_hot_action, [1])
        assert len(one_hot_action) == len(self.env.get_available_buttons())

        _ = self.env.make_action(list(one_hot_action), self._action_frame_repeat)

        self.update_game_variables()

        if self.steps > 1:
            reward = self.update_reward()
            # print("skill from DoomGame is", self.properties['skill'])
        else:
            reward = 0

        self.steps += 1
        self.global_steps += 1
        done = self.env.is_episode_finished()
        # state is available only if the episode is still running
        if not done:
            self._rgb_array = self.env.get_state().screen_buffer
        observation = self._process_image(self._rgb_array)
        return observation, reward, done

    def step(self, action):
        observation, reward, done = self._step(action)
        return observation, reward, done, {}

    def close(self):
        """Close environment"""
        self.env.close()

    def seed(self, seed=None):
        """Seed"""
        raise NotImplementedError

    def render(self, mode='human'):
        """Render frame"""
        if mode == 'rgb_array':
            return self._rgb_array
        raise NotImplementedError


class VizDoomGymFeatStackVar(VizDoomGym):

    @override(VizDoomGym)
    def __init__(self, mission_file, scaled_resolution, action_frame_repeat, curriculum=False, increase_diff=False,
                 deathmatch=True):
        super(VizDoomGymFeatStackVar, self)._init(mission_file, scaled_resolution, action_frame_repeat, curriculum,
                                                 increase_diff, deathmatch)

        self.stack_len = 4

        self.frames = deque([], maxlen=self.stack_len)

        self.observation_space = Dict({
            "frags": Box(low=0,
                         high=100,
                         shape=(1,),
                         dtype=np.int64),
            "frames": gym.spaces.Box(low=0.0,
                                     high=1.0,
                                     shape=(self.rows, self.columns, 3 * self.stack_len),
                                     dtype=np.float32),
            # start with just enemy visible or not (then we can add stuff pertaining to health, target inline, etc.
            "game_features": Box(low=0,
                                 high=1,
                                 shape=(1,),
                                 dtype=np.int64),
            # health and ammo
            "game_variables": Box(low=-10,
                                  high=10,
                                  shape=(2,),
                                  dtype=np.float32)
        })

        self.reset()

    def get_game_variables(self):
        return np.array([self.properties['health'] / 100., self.properties['sel_ammo'] / 15.])

    @override(VizDoomGym)
    def step(self, action):
        observation, reward, done = self._step(action)
        self.frames.append(observation)
        assert len(self.frames) == self.stack_len
        observation = np.concatenate(self.frames, axis=2)
        game_features = self.get_game_features()
        game_vars = self.get_game_variables()
        observation = {
            "frags": self.cum_kills,
            "frames": observation,
            "game_features": game_features,
            "game_variables": game_vars
        }
        return observation, reward, done, {}

    # We will need to modify this if we want to indicate more visible stuff
    def get_game_features(self):
        if int(self.properties['visible']) == 1:
            return np.array([1])
        else:
            return np.array([0])

    @override(VizDoomGym)
    def reset(self):
        observation = self._reset()
        for _ in range(self.stack_len):
            self.frames.append(observation)
        observation = np.concatenate(self.frames, axis=2)
        observation = {
            "frags": np.array([0]),
            "frames": observation,
            "game_features": np.array([0]),
            "game_variables": np.array([1.0, 1.0])
        }
        return observation