from .Agent import Agent
import numpy as np
import matplotlib.pyplot as plt
from sympy import Integer
import heapq
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg

class SwarmRW_Agent(Agent):
    
    def look_around(self):
        detection_radius = self.model.coverage_radius - 1 # detection radius is coverage radius - 1 to make sure the agent can cover edge cells
        # look around the robot and find the obstacles
        neighbors = self.model.grid.iter_neighbors((self.pos[0],self.pos[1]), moore=True, include_center=False, radius=detection_radius)
        for neighbor in neighbors:
            if self.model.is_there_agent(neighbor.pos):
                # record the collision times
                self.delta_t = self.model.steps - self.last_collision_time
                self.last_collision_time = self.model.steps
                self.collision_times.append(self.delta_t)
                # find the angle between the robot and the detected agent/obstacle
                dx = neighbor.pos[0] - self.pos[0]
                dy = neighbor.pos[1] - self.pos[1]
                angle = np.arctan2(dy, dx)
                # opoosite angle
                angle_op = angle + np.pi
                return angle_op # return the opposite angle
        return None

    def destination(self):
        # based on agle and step size, compute the destination
        destination = [self.pos[0] + self.step_size * np.cos(self.angle), self.pos[1] + self.step_size * np.sin(self.angle)]
        destination = [int(destination[0]), int(destination[1])]
        return destination

    def __init__(self, unique_id, model, pos, params, chosen=False):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.model = model

        self.moves = 0
        self.pos = pos
        self.pos_history = [] # history of the positions
        self.pos_history.append(self.pos) # add the initial position to the history

        self.k = params['k']

        # cover initial position
        self.cover_cells(self.pos, support_pm=False)

        # chosen is a flag to indicate that the agent is chosen
        # it is used for plots
        # model randomly chooses an agent and set it as chosen
        self.chosen = chosen
        self.fig = None # figure for the plots

        #initial Step size
        self.max_step_size = np.min([self.model.width, self.model.height])
        self.step_size = self.model.random.uniform(0, self.max_step_size)
        self.angle = self.model.random.uniform(0, 2 * np.pi)

        # collision times array
        self.collision_times = []
        # last collision time
        self.last_collision_time = 0
        # delta_t
        self.delta_t = None

    def t_bar(self):
        if len(self.collision_times) > 0:
            return np.mean(self.collision_times)
        else:
            return 0
    def step(self):
        destination = self.destination()

        # new_pos is one cell towards the destination from the current position
        next_cell = [int(self.pos[0] + np.sign(destination[0] - self.pos[0])), int(self.pos[1] + np.sign(destination[1] - self.pos[1]))]
        # next cell can not be diagonal
        if next_cell[0] != self.pos[0] and next_cell[1] != self.pos[1]:
            # randomly choose one of the two cells
            if self.model.random.uniform(0, 1) < 0.5:
                next_cell = [next_cell[0], self.pos[1]]
            else:
                next_cell = [self.pos[0], next_cell[1]]
        flag_moved = False
        # check if the next_cell is in bounds
        if next_cell[0] >= 0 and next_cell[0] < self.model.width:
            if next_cell[1] >= 0 and next_cell[1] < self.model.height:
                # check if the new position is unoccupied
                if self.model.is_unoccupied(next_cell):
                    self.model.grid.move_agent(self, next_cell)
                    self.moves += 1
                    self.pos = next_cell
                    self.pos_history.append(next_cell)
                    self.cover_cells(next_cell, support_pm=False)
                    flag_moved = True
        # robot tried to move towards the destination
        # if it moved, check if it reached the destination and if so, we need a new move
        # if it did not move, we need a new move anyway

        flag_new_move = False
        if flag_moved:
            if self.pos == destination:
                flag_new_move = True
        else:
            flag_new_move = True

        
        # look around the robot and find any possible collision
        collision_angle = self.look_around()

        if flag_new_move:
            delta_t_up_to_now = self.model.steps - self.last_collision_time
            t_bar = self.t_bar()
            if delta_t_up_to_now >= t_bar:
                # assuimg the robot velocity is 1
                self.step_size = 1 * t_bar + self.k * self.step_size
                self.step_size = min(self.step_size, self.max_step_size)
            else:
                self.step_size = 1 * t_bar - self.k * self.step_size
                self.step_size = max(self.step_size, 1)
            if collision_angle is not None:
                self.angle = collision_angle
                # modify the angle up to +- 45 degrees
                # self.angle += self.model.random.uniform(-np.pi/4, np.pi/4)
            else:
                self.angle = self.model.random.uniform(0, 2 * np.pi)

        # add moves to the model
        self.model.coverage_curve_ttd[-1] += self.moves