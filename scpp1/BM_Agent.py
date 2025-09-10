from .Agent import Agent
import numpy as np
import matplotlib.pyplot as plt
from sympy import Integer
import heapq
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg

class BM_Agent(Agent):
    # Function to round to the velocity to {-1, 0, 1}
    def discretize(self,v):
        return np.clip(np.round(v), -1, 1).astype(int)
   
    def __init__(self, unique_id, model, pos, params, chosen=False):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.model = model

        self.moves = 0
        self.pos = pos
        self.pos_history = [] # history of the positions
        self.pos_history.append(self.pos) # add the initial position to the history
        
        self.sigma = params['sigma']

        # Initialize random velocities v_x and v_y
        self.v_x_t = np.random.uniform(-1, 1)
        self.v_y_t = np.random.uniform(-1, 1)

        # cover initial position
        self.cover_cells(self.pos, support_pm=False)

        # chosen is a flag to indicate that the agent is chosen
        # it is used for plots
        # model randomly chooses an agent and set it as chosen
        self.chosen = chosen
        self.fig = None # figure for the plots


    def step(self):
        new_pos = [self.pos[0] + self.v_x_t, self.pos[1] + self.v_y_t]
        # make sure new_pos is integer
        new_pos = [int(new_pos[0]), int(new_pos[1])]
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

        # Apply Brownian motion
        delta_v_x = np.random.normal(0, self.sigma)
        delta_v_y = np.random.normal(0, self.sigma)
        
        self.v_x_t1 = self.v_x_t + delta_v_x
        self.v_y_t1 = self.v_y_t + delta_v_y
        
        # Round the velocities to {-1, 0, 1}
        self.v_x_t = self.discretize(self.v_x_t1)
        self.v_y_t = self.discretize(self.v_y_t1)

        # one element of v must be zero
        if self.v_x_t != 0 and self.v_y_t != 0:
            if self.model.random.random() < 0.5:
                self.v_x_t = 0
            else:
                self.v_y_t = 0

        # add moves to the model
        self.model.coverage_curve_ttd[-1] += self.moves