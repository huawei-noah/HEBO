# 2023.02.14-Changed for RLLG
#            Huawei Technologies Co., Ltd. <paul.daoudi1@huawei.com>

# Copyright (c) Davidobot (https://github.com/davidobot/adversarial_hrl).

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class PointFallEnv(gym.Env):
    """
    This enviroment is taken from https://github.com/davidobot/adversarial_hrl.

    Description:
        Simple and fast-to-run implementation of PointFall, based on AntFall.

    Source:
        Inspired by https://github.com/tensorflow/models/tree/master/research/efficient-hrl/environments

    Observation:
        Type: Box(3)
        Num     Observation               Min                     Max
        0       Point X co-ordinate       -1.5                    2.5
        1       Point Y co-ordinate       -1.5                    4.5
        2       Point Orientation         0.0                     1.0 (equivalent to 2pi)
        3       Time                      0.0                     max_steps / 10. (default: 50.0)

    Actions:
        Type: Box(2)
        Num   Action    Min         Max       
        0     Move      -1./scale   1./scale
        1     Rotate    -pi/4       pi/4

    Reward:
        100 if reached goal square; -0.1 otherwise for every timestep

    Starting State:
        Starting state is [U(-0.1, 0.1), U(-0.1, 0.1), (U(-0.1, 0.1) % (2pi)) / (2pi)]
        
        Note: moving with orientation of 0 is going to move the point right.
        The orientation increases anti-clockwise, so if in a state [0, 0, 0.25] (an orientation of pi/2)
        and executing action [0.5, 0] (move forward) will result in a new state [0, 0.5, 0.25]

    Episode Termination:
        Reaching the goal square. Time limit is left up to the user; recommended is 500 for a scaling factor of 4.
        
        The environment (for 500 steps max; scale_factor=4) is considered "solved" if an agent achieves an average reward of 90 or more over the latest 100 episode.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 move_block_only=False,
                 scaling_factor=4,
                 max_steps=1000):
        # 1 is a solid wall; 'r' is the end square;
        # m denotes a movable block
        # defined "right-side up" at the great expense of having to work with two y axis (0 at top, 0 at bottom)
        self.maze = [
            [1, 1, 1, 1],
            [1, 'r', 0, 1],
            [1, 1, 0, 1],
            [1, 0, 'm', 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ]
        
        # scaling factor effects movement
        self.SCALING_FACTOR = scaling_factor
        self.maze_width = len(self.maze[0])
        self.maze_height = len(self.maze)
        
        # x, y, orientation
        self.starting_point = np.array([1.5, 2.0, 0.])
        self.state_normalisation_factor = np.array([1., 1., 2. * math.pi])
        self.max_dist = 1. / self.SCALING_FACTOR
        self.max_turn = math.pi / 4
        
        # movable block
        self.block_offset = np.array([0., 0.])
        self.block_start  = np.array([2, 2])
        self.block_movable_axis = np.array([0, 1]) # 1 indicates movable on given axis
        self.block_axis_restrictions = np.array([0., 1.]) # how far the block can move along each axis (symmetrical)
        self.block_friction = 0.5 # increase to make it more difficult

        # x_lim, y_lim, theta_lim, time_lim
        self.max_steps = max_steps
        high = np.array([self.maze_width - self.starting_point[0],
                         self.maze_height - self.starting_point[1],
                         0.0,
                         1.0,
                         1.0, max_steps / 10.],
                        dtype=np.float32)
        low = np.array([-self.starting_point[0],
                        -self.starting_point[1],
                        0.0,
                        0.0,
                        0.0, 0.0],
                       dtype=np.float32)
        
        # {max movement distance, max turn angle} in a single turn
        action_high = np.array([self.max_dist, self.max_turn], dtype=np.float32)

        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.episode_steps = 0
        self.steps_beyond_done = None
        
        self.reset()

        # move block only or not
        self.move_block_only = move_block_only

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        self.episode_steps += 1
        
        ds = action[0]
        dtheta = action[1]
        
        _x, _y, theta = self.state
        
        # update theta and keep normalised to [0, 2pi] range
        theta = (theta + dtheta) % (2 * math.pi)
        # update position
        x = _x + math.cos(theta) * ds
        y = _y + math.sin(theta) * ds
        
        wall_collision = self.is_colliding_wall(_x, _y, x, y)
        block_collision = self.resolve_block_collision(_x, _y, x, y)
        
        if not (wall_collision or block_collision):
            self.state[0] = x
            self.state[1] = y
            self.state[2] = theta
        
        done = False
        if self.episode_steps + 1 == self.max_steps:
            done = True

        agent_in_goal = self.is_colliding_reward(self.state[0], self.state[1])
        reward = -0.001

        if self.move_block_only:
            if self.block_offset[1] > 0:
                reward = self.block_offset[1]
        else:
            if agent_in_goal: # and self.steps_beyond_done is None:
                # solved the maze!
                reward = 1
                # self.steps_beyond_done = 0

        if self.steps_beyond_done is not None:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1

        return self.normalised_state(), reward, done, {}
    
    def is_colliding_reward(self, x, y):
        x = math.floor(x)
        y = math.floor(y)
        
        if x >= 0 and x < self.maze_width and y >= 0 and y < self.maze_height:
            return self.maze[self.maze_height - 1 - y][x] == 'r'
        
        return False
    
    # https://love2d.org/forums/viewtopic.php?t=76752#p159136
    def liang_barsky(self, l,t,w,h, x1,y1,x2,y2):
        t0, t1 = 0, 1
        dx, dy = x2-x1, y2-y1

        for side in range(1, 5):
            if side == 1:
                p,q = -dx, x1 - l
            elif side == 2:
                p,q =  dx, l + w - x1
            elif side == 3:
                p,q = -dy, y1 - t
            else:
                p,q =  dy, t + h - y1

            if p == 0:
                if q < 0:
                    return None
            else:
                r = q / p
                if p < 0:
                    if r > t1:
                        return None
                    elif r > t0:
                        t0 = r
                else:
                    if r < t0:
                        return None
                    elif r < t1:
                        t1 = r
        return t0, t1
    
    # 1 is a solid wall; 'r' is the end square
    # m denotes a movable block
    def is_colliding_wall(self, _ox, _oy, _x, _y):
        sign = lambda x: math.copysign(1, x)
        x = math.floor(_x)
        y = math.floor(_y)
        ox = math.floor(_ox)
        oy = math.floor(_oy)
        
        # simple check if landing on a impassable square
        if x >= 0 and x < self.maze_width and y >= 0 and y < self.maze_height:
            if self.maze[self.maze_height - 1 - y][x] == 1:
                return True
        
        # perform rudimentary raycasting to prevent clipping through corners
        for my in range(min(oy, y), max(oy, y) + 1):
            for mx in range(min(ox, x), max(ox, x) + 1):
                if self.maze[self.maze_height - 1 - my][mx] == 1:
                    raycast = self.liang_barsky(mx, my, 1, 1, _ox, _oy, _x, _y)
            
                    if raycast is not None:
                        return True
        
        return False
    
    # old x/y, new x/y
    def resolve_block_collision(self, ox, oy, x, y):
        sign = lambda x: math.copysign(1, x)
    
        bx, by = self.block_start + self.block_offset
        
        if self.block_offset[1] == 1.:
            return False # block is fully pushed in, can move across
    
        if x >= bx and x < bx + 1 and y >= by and y < by + 1:
            # colliding with block
            dx = self.block_movable_axis[0] * sign(x - ox) *\
                 (abs(x - ox) - min(abs(bx - ox), abs(ox - bx - 1))) / self.block_friction
            dy = self.block_movable_axis[1] * sign(y - oy) *\
                 (abs(y - oy) - min(abs(by - oy), abs(oy - by - 1))) / self.block_friction
            
            # only move in one axis
            dx *= (ox < bx or ox > bx + 1)
            dy *= (oy < by or oy > by + 1)
            
            # restrict movement of block to one axis
            if np.sum(self.block_movable_axis) == 2:
                if abs(dx) > 0:
                    self.block_movable_axis[1] = 0
                    dy = 0
                elif abs(dy) > 0:
                    self.block_movable_axis[0] = 0
                    dx = 0
            
            self.block_offset[0] += dx
            self.block_offset[1] += dy
            self.block_offset = np.clip(self.block_offset, -self.block_axis_restrictions, self.block_axis_restrictions)
            
            # do not move player if colliding with block
            return True
            
        # raycast to avoid clipping through movable block corners
        raycast = self.liang_barsky(bx, by, 1, 1, ox, oy, x, y)
        if raycast is not None:
            return True
        
        return False
        
    # starting point is always (0, 0); normalise theta to [0, 1]; add time
    def normalised_state(self):
        return np.concatenate([(self.state - self.starting_point) / self.state_normalisation_factor, self.block_offset, [self.episode_steps / 10.]])
    
    def reset(self):
        noise = np.random.uniform(-0.45, 0.45, self.starting_point.shape)
        noise[1] *= 2
        self.state = np.array(self.starting_point + noise, dtype=np.float32)
        self.state[2] += math.pi / 2. # start facing up
        self.state[2] = self.state[2] % (2 * math.pi)
        self.steps_beyond_done = None
        self.block_offset = np.array([0., 0.])
        self.block_movable_axis = np.array([0, 1])
        
        self.episode_steps = 0
        return self.normalised_state()

    def render(self, mode='human'):
        screen_width = 400
        screen_height = 600

        world_width = self.maze_width
        block_size = screen_width/world_width
        
        point_size = block_size / 5.

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            self.point_trans = rendering.Transform()
            self.point_rot_trans = rendering.Transform()
            
            self.movable_trans = rendering.Transform()
            
            for yy in range(self.maze_height):
                y = self.maze_height - 1 - yy
                for x in range(self.maze_width):
                    if self.maze[y][x] == 1 or self.maze[y][x] == 'r':
                        block = rendering.FilledPolygon([(0, 0), (0, block_size), (block_size, block_size), (block_size, 0)])
                        block_trans = rendering.Transform(translation=(x * block_size, yy * block_size))
                        block.add_attr(block_trans)
                        
                        if self.maze[y][x] == 1:
                            block.set_color(0.2, 0.2, 0.2)
                        elif self.maze[y][x] == 'r':
                            block.set_color(0.2, 0.2, 0.8)
                        
                        self.viewer.add_geom(block)
            
            movable = rendering.FilledPolygon([(0, 0), (0, block_size), (block_size, block_size), (block_size, 0)])
            movable.add_attr(rendering.Transform(translation=(self.block_start[0] * block_size, self.block_start[1] * block_size)))
            movable.add_attr(self.movable_trans)
            movable.set_color(0.2, 0.8, 0.2)
            self.viewer.add_geom(movable)
            
            point = rendering.make_circle(point_size / 2.)
            point.add_attr(self.point_trans)
            point.set_color(0.8, 0.2, 0.2)
            self.viewer.add_geom(point)
            
            orientir = rendering.Line((0, 0), (0, 2 * point_size))
            orientir.linewidth.stroke = 5
            orientir.add_attr(self.point_rot_trans)
            orientir.add_attr(self.point_trans)
            orientir.set_color(0.8, 0.2, 0.2)
            self.viewer.add_geom(orientir)          

        if self.state is None:
            return None

        x, y, theta = self.state
        self.point_trans.set_translation(x * block_size, y * block_size)
        self.point_rot_trans.set_rotation(theta - math.pi / 2) # tweak
        
        x, y = self.block_offset
        self.movable_trans.set_translation(x * block_size, y * block_size)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None