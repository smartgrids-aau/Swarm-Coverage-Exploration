from .Agent import Agent
import numpy as np
import matplotlib.pyplot as plt
from sympy import Integer
import heapq
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg

class LF_Agent(Agent):
    
    def levy_flight_step(self,lambda_param, s0=1):
        # Generate random numbers from a uniform distribution
        r = self.model.random.random()
        # Apply the inverse transform sampling formula for the step size
        step_size = s0 * (1 - r) ** (-1 / (lambda_param - 1))
        return step_size

    def generate_path(self):
        self.path = []
        while len(self.path) < 2:
            self.path = []
            # Initialize angle and step size
            self.angle = self.model.random.uniform(0, 2 * np.pi)
            # distribution of step size is P(s) = s^(-lambda)
            self.step_size = self.levy_flight_step(self.lf_lambda)
            # step size should not be larger than map size
            self.step_size = min(self.step_size, self.model.width, self.model.height)
            # compute the new position
            self.destination = [self.pos[0] + self.step_size * np.cos(self.angle), self.pos[1] + self.step_size * np.sin(self.angle)]
            self.destination = [int(self.destination[0]), int(self.destination[1])]

            # generate a path to the destination
            dx = self.destination[0] - self.pos[0]
            dy = self.destination[1] - self.pos[1]
            dir_x = np.sign(dx)
            dir_y = np.sign(dy)
            dx = abs(dx)
            dy = abs(dy)
            for i in range(dx):
                self.path.append([self.pos[0] + dir_x * i, self.pos[1]])
            for i in range(dy):
                self.path.append([self.destination[0], self.pos[1] + dir_y * i])

        # remove the first position from the path
        self.path.pop(0)

    def __init__(self, unique_id, model, pos, params, chosen=False):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.model = model

        self.moves = 0
        self.pos = pos
        self.pos_history = [] # history of the positions
        self.pos_history.append(self.pos) # add the initial position to the history
        
        self.lf_lambda = params['lambda']

        # cover initial position
        self.cover_cells(self.pos, support_pm=False)

        # chosen is a flag to indicate that the agent is chosen
        # it is used for plots
        # model randomly chooses an agent and set it as chosen
        self.chosen = chosen
        self.fig = None # figure for the plots

        # generate initial path
        self.generate_path()


    def step(self):
        if len(self.path) == 0:
            self.generate_path()
        # move along the path
        new_pos = self.path.pop(0)
        # make sure new_pos is integer
        new_pos = [int(new_pos[0]), int(new_pos[1])]
        # try to move to the new position
        flag_moved = False
        # check if the new position is in bounds
        if new_pos[0] >= 0 and new_pos[0] < self.model.width:
            if new_pos[1] >= 0 and new_pos[1] < self.model.height:
                # check if the new position is unoccupied
                if self.model.is_unoccupied(new_pos):
                    self.model.grid.move_agent(self, new_pos)
                    self.moves += 1
                    self.pos = new_pos
                    self.pos_history.append(new_pos)
                    self.cover_cells(new_pos, support_pm=False)
                    flag_moved = True

        # if the agent could not move, generate a new path
        if not flag_moved:
            # if the agent could not move, generate a new path
            self.generate_path()

        # add moves to the model
        self.model.coverage_curve_ttd[-1] += self.moves