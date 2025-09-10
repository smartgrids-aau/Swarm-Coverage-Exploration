from .Agent import Agent
import numpy as np
import matplotlib.pyplot as plt
from sympy import Integer
import heapq
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg

class Astar_node:
    def __init__(self, cell, is_obstacle, g, goal, parent=None):
        self.cell = cell
        self.parent = parent
        if is_obstacle:
            self.g = np.inf
            self.h = np.inf
            self.value = np.inf
        else:
            self.g = g
            # h is the heuristic value equal to the direct distance to the goal
            self.h = np.sqrt((cell[0]-goal[0])**2 + (cell[1]-goal[1])**2)
            self.value = self.g + self.h
    # less than operator is defined for the priority queue
    def __lt__(self, other):
        return self.value < other.value
    # equality is defined based on the cell position
    def __eq__(self, other):
        return self.cell[0] == other.cell[0] and self.cell[1] == other.cell[1]

class scpp1Agent(Agent):
    def __init__(self, unique_id, model, pos, params, chosen=False):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.model = model
        
        self.moves = 0
        self.pos = pos
        self.pos_history = [] # history of the positions
        self.pos_history.append(self.pos) # add the initial position to the history
        self.agents_positions = dict()
        self.vel = [1,1]

        # private visits map [sharable with neighbors]
        self.pm_visits = np.zeros((self.model.width,self.model.height))
        # private obstacles map [sharable with neighbors]
        self.pm_obstacles = np.zeros((self.model.width,self.model.height))
        # private penalty map
        self.pm_penalty = np.zeros((self.model.width,self.model.height))

        # placeholders for the fitness map calculations
        self.alpha = np.zeros((self.model.width,self.model.height))
        self.beta = np.zeros((self.model.width,self.model.height))
        self.gamma = np.zeros((self.model.width,self.model.height))
        # private fitness map
        self.pm_fitness = np.empty((self.model.width,self.model.height))
        
        self.p1 = params['p1']
        self.p2 = params['p2']
        self.p3 = params['p3']

        # used for adjusting parameters based on the number of unvisited cells
        self.copy_p1, self.copy_p2, self.copy_p3 = self.p1, self.p2, self.p3 #TODO: check if this is necessary

        # chosen is a flag to indicate that the agent is chosen
        # it is used for plots
        # model randomly chooses an agent and set it as chosen
        self.chosen = chosen
        self.fig = None # figure for the plots

        # calulate unvisited ratio
        self.unvisited_ratio = 1 #TODO: check if this is necessary
        self.adjust = 1 #TODO: check if this is necessary
        self.randomed = False #TODO: check if this is necessary

        # for the agent to know if it is on a path to a faraway cell (using A* algorithm)
        self.best_pos = self.pos
        self.is_on_path = False
        self.path = []

        # compute available cells to visit
        self.available_cells = self.model.width * self.model.height
        self.total_cells = self.available_cells # constant value during the simulation, only used for ratio calculation, based on the world size

        # cover initial position
        self.cover_cells(self.pos)

    # reports the remaining ratio of the cells to be covered
    def remaining_cells_ratio(self):
        return self.available_cells / self.total_cells

    def information_exchange(self):
        # get current neighbors from the model
        neighbors = self.model.get_neighbors(self)
        # for each neighbor,
        # - update the agent's position in the private dictionary
        # - get the agent's private visits map and obstacles map
        for neighbor in neighbors:
            # update agents positions in the private dictionary
            self.agents_positions[neighbor.unique_id] = neighbor.pos
            # get the agent's private visits map and obstacles map
            self.pm_visits = np.maximum(self.pm_visits, neighbor.pm_visits > 0)
            self.pm_obstacles = np.maximum(self.pm_obstacles, neighbor.pm_obstacles)
            # penalty map is not sharable

    # evaluates the fitness of the grid
    def map_evaluation(self):
        # negative points for a cell: visits + penalties (personal) + if any obstacle is there
        # if obstacle is here, fitness is 0
        self.alpha = self.pm_visits + self.pm_obstacles * 1000000 + self.pm_penalty
        # for plotting
        self.visits_penalties_obstacles = self.alpha
        self.alpha = 1 + 1/(self.alpha+1)
        # apply blur effect with kernel size of self.model.coverage_radius
        # kernel_size = self.model.coverage_radius
        # if kernel_size % 2 == 0:
        #     kernel_size += 1
        # self.alpha = cv2.GaussianBlur(self.alpha, (kernel_size, kernel_size), 0)
        # a positive point for a cell is its distance to other robots
        # calculations are optimized in this part
        K = np.sqrt(self.model.width**2+self.model.height**2)
        # number of agents
        N = len(self.agents_positions)
        # td: total distances (td) to other robots
        td = np.zeros((self.model.width,self.model.height)) + K * N # assume cells are max distance
        distance_to_self = np.zeros((self.model.width,self.model.height)) + K # assume cells are max distance
        # extract positions of other agents from the dictionary
        agent_locations = list(self.agents_positions.values())
        # get coverage radius from the model and compute effecive box half size
        effective_box_hs = self.model.calculation_radius
        # other agents
        # loop in a square around the agent with side effective_box_hs
        for i in range(max(0,self.pos[0]-int(effective_box_hs)), min(self.model.width,self.pos[0]+int(effective_box_hs)+1)):
            for j in range(max(0,self.pos[1]-int(effective_box_hs)), min(self.model.height,self.pos[1]+int(effective_box_hs)+1)):
                for agent_location in agent_locations:
                    # update total distances
                    td[i,j] += np.sqrt((i-agent_location[0])**2 + (j-agent_location[1])**2)
        # this agent
        for i in range(max(0,self.pos[0]-int(effective_box_hs)), min(self.model.width,self.pos[0]+int(effective_box_hs)+1)):
            for j in range(max(0,self.pos[1]-int(effective_box_hs)), min(self.model.height,self.pos[1]+int(effective_box_hs)+1)):
                # update total distances
                distance_to_self[i,j] = np.sqrt((i-self.pos[0])**2 + (j-self.pos[1])**2)
        # for plotting
        self.distance_to_other_robots = td
        self.distance_to_self = distance_to_self

        # normalize total distances if there are more than 1 agent
        # num_agents = len(agent_locations)-1
        # if num_agents > 0:
        #     td = td / num_agents
        
        # make copies of the terms for plotting before puting powers for visualization in timelaps (model code)
        self.alpha_ = self.alpha.copy()
        self.beta_ = td.copy()
        self.gamma_ = distance_to_self.copy()
        
        # combine the fitness metric with the distance metric
        self.alpha = np.power(self.alpha,self.p1)
        self.beta = np.power(td,self.p2) + 1
        self.gamma = np.power(distance_to_self,self.p3) + 1
        self.pm_fitness = self.alpha * self.beta / self.gamma

        # fitness of the current cell is always 0
        self.pm_fitness[self.pos[0],self.pos[1]] = 0.0

    # selects a random option as personal best choice for the robot
    # candidates are grid cells with a fitness equal to the pbv of the agent
    def find_pbest(self):
        self.map_evaluation()
        self.best_val = np.max(self.pm_fitness)
        # candidates
        row_indices, col_indices = np.where(self.pm_fitness == self.best_val)
        num_candidates = len(row_indices)
        random_index = np.random.randint(0, num_candidates)
        selected_row = row_indices[random_index]
        selected_col = col_indices[random_index]
        self.best_pos = np.array([selected_row, selected_col])

        if self.chosen and self.model.steps > 10 and False:
            # Create a named window
            self.window_name = 'Fitness function terms for the chosen agent'
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)  # Allows resizing
            self.fig, self.axs = plt.subplots(2, 2, figsize=(8, 10))
            self.variables = [self.visits_penalties_obstacles, self.distance_to_other_robots, self.distance_to_self, self.pm_fitness]
            # D bar
            a = "D"
            self.titles = [r"$\frac{1}{A}$", r"$\bar{D}$", 'D', 'F']
            for i, (var, title) in enumerate(zip(self.variables, self.titles)):
                var = np.array(var)
                var = var.T
                var = np.flip(var, 0)
                ax = self.axs[i // 2, i % 2]
                im = ax.imshow(var, cmap='jet', interpolation='nearest')
                ax.set_title(title)
                self.fig.colorbar(im, ax=ax)

            plt.tight_layout()

            # Render the plot to a numpy array
            canvas = FigureCanvasAgg(self.fig)
            canvas.draw()
            image_array = np.array(canvas.renderer.buffer_rgba())

            # Convert numpy array to CV2 image
            image_cv2 = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)

            # Show the image in the OpenCV window
            cv2.imshow(self.window_name, image_cv2)
            key = cv2.waitKey(0) & 0xFF

    def step(self):
        # check if current cell needs work
        # else, set velocity to 0, so the agent will stay in the same cell
        # the if condition is always true in the current implementation (Work is not used in the current implementation)
        if not self.model.needs_work(self.pos) or True:
            # before any decision, exchange information with neighbors
            self.information_exchange()
            # check if the updated information reports that the current personal best choice is not valid anymore
            if self.pm_obstacles[self.best_pos[0],self.best_pos[1]] == 1 or self.pm_visits[self.best_pos[0],self.best_pos[1]] == 1:
                self.is_on_path = False # if the personal best choice is not valid, find a new personal best choice and a path to it
            # if the robot is not following a path, find a new path
            if not self.is_on_path:
                # find the personal best choice for the robot
                self.find_pbest()
                # find the path to the personal best choice using A* algorithm
                open_list = []
                closed_list = []
                start_node = Astar_node(self.pos, self.pm_obstacles[self.pos[0],self.pos[1]], 0, self.best_pos)
                heapq.heappush(open_list, start_node)
                while len(open_list) > 0:
                    current_node = heapq.heappop(open_list)
                    if current_node.cell[0] == self.best_pos[0] and current_node.cell[1] == self.best_pos[1]:
                        self.is_on_path = True
                        # put the path to the history
                        self.path = []
                        while current_node is not None:
                            self.path.append(current_node.cell)
                            self.model.toggle_on_path(current_node.cell, True)
                            current_node = current_node.parent
                        self.path.reverse()
                        # omit the first cell, since the robot is already there
                        self.path.pop(0)
                        # if self.chosen:
                        #     print('I am in position', self.pos, 'and I am going to', self.best_pos, 'with a path of', self.path)
                        break
                    closed_list.append(current_node)
                    neighbors = self.model.grid.iter_neighbors((current_node.cell[0],current_node.cell[1]), moore=False, include_center=False, radius=1)
                    for neighbor in neighbors:
                        neighbor_node = Astar_node(neighbor.pos, self.pm_obstacles[neighbor.pos[0],neighbor.pos[1]], current_node.g+1, self.best_pos, current_node)
                        if neighbor_node in closed_list:
                            continue
                        if neighbor_node in open_list:
                            continue
                        heapq.heappush(open_list, neighbor_node)
                # must not happen that a path is not found
                # raise an exception if this happens
                if not self.is_on_path:
                    raise Exception("Path not found")
            
            # robot is on path
            next_cell = self.path.pop(0)
            if len(self.path) == 0:
                self.is_on_path = False
            self.vel = [next_cell[0]-self.pos[0],next_cell[1]-self.pos[1]]
            mv = np.sqrt(self.vel[0]**2+self.vel[1]**2)
            self.vel =  1000*self.vel / mv
            v = self.vel.astype(int)*1
            if v[0] < 0:
                v[0] = -1
            if v[0] > 0:
                v[0] = 1
            if v[1] < 0:
                v[1] = -1
            if v[1] > 0:
                v[1] = 1
            
            if [v[0],v[1]] == [1,-1]:
                v = self.model.random.choice(([0,-1],[1,0]))
            elif [v[0],v[1]] == [1,1]:
                v = self.model.random.choice(([0,1],[1,0]))
            elif [v[0],v[1]] == [-1,1]:
                v = self.model.random.choice(([0,1],[-1,0]))
            elif [v[0],v[1]] == [-1,-1]:
                v = self.model.random.choice(([0,-1],[-1,0]))
            if self.randomed:
                self.randomed = False
                # randomly change 1 element of the velocity
                if self.model.random.random() < 0.5:
                    v[0] = -v[0]
                else:
                    v[1] = -v[1]
        else:
            # this only happens if the robot is in a cell that needs work
            # in the current implementation, this is not possible [TODO: rename Work to Trap]
            v = [0,0]

        # try to move the robot to the new position based on the velocity
        new_pos = [self.pos[0]+v[0],self.pos[1]+v[1]]
        # check if new position is within bounds or not
        successful_move = False
        if new_pos[0] >= 0 and new_pos[0] < self.model.width:
            if new_pos[1] >= 0 and new_pos[1] < self.model.height:
                # check if the new position is unoccupied
                if self.model.is_unoccupied(new_pos):
                        # now that the new position is in bounds and unoccupied, move the robot
                        self.model.toggle_on_path(self.pos, False) #TODO: check if this is necessary
                        self.model.grid.move_agent(self, new_pos)
                        # add the new position to the history
                        self.pos_history.append(new_pos)

                        # increase the number of moves for this agent
                        self.moves += 1

                        # on a successful movement, put a discount on previous penalties
                        self.pm_penalty = self.pm_penalty * 0.5

                        # mark the new position and the cells around it as covered (limited by the coverage radius)
                        self.cover_cells(new_pos)

                        # mark the move as successful
                        successful_move = True
                else:
                    # path is blocked, find a new path
                    self.is_on_path = False
                    
                    # check if an obstacle is in the target cell
                    if self.model.is_obstacle(new_pos):
                        # mark the obstacle in the private map
                        if self.pm_obstacles[new_pos[0],new_pos[1]] == 0:
                            # if the cell is not already marked as an obstacle
                            # mark it as an obstacle
                            # decrease the number of available cells
                            self.available_cells -= 1
                            self.pm_obstacles[new_pos[0],new_pos[1]] = 1
                        
                    # if the cell is occupied by another agent, put penalty on the cell that caused the collision
                    else:
                        self.pm_penalty[self.best_pos[0],self.best_pos[1]] += 1


        # look around the robot and find the obstacles
        neighbors = self.model.grid.iter_neighbors((self.pos[0],self.pos[1]), moore=True, include_center=False, radius=1)
        for neighbor in neighbors:
            if self.model.is_obstacle(neighbor.pos):
                # mark the obstacle in the private map
                if self.pm_obstacles[neighbor.pos[0],neighbor.pos[1]] == 0:
                    self.available_cells -= 1
                    self.pm_obstacles[neighbor.pos[0],neighbor.pos[1]] = 1

        # if self.model.with_obstacle:
        #     # if map is None, there is no obstacle, so this check is not required
        #     self.available_cells -= self.find_unreachables()
        
        # based on ratio of unvisited cells, adjust the parameters
        # increase p1, decrease p3, decrease p2
        rem = self.remaining_cells_ratio()
        if rem < 0.1:
            p2_cut = 0.5 * self.p2
            if self.p2 - p2_cut >= 1:
                self.p2 -= p2_cut
                self.p1 += p2_cut
            p3_cut = 0.5 * self.p3
            if self.p3 - p3_cut >= 1:
                self.p3 -= p3_cut
                self.p1 += p3_cut

        # add moves to the model
        self.model.coverage_curve_ttd[-1] += self.moves

    # finds the unreachable cells in the grid
    # especially useful for the inner cells of the obstacles
    # otherwize, the robot will be stuck in the obstacle
    # since there is no way to check the inner cells of the obstacle
    # and mark them as obstacles or visited
    def find_unreachables(self):
        mmm = self.pm_obstacles.copy()
        for pos in self.agents_positions.values():
            # 0xAB is a number that is not obstacle and not free
            # in obstacle map, 1 is obstacle, 0 is free
            mmm[pos[0],pos[1]] = 0xAB
        mmm[self.pos[0],self.pos[1]] = 0xAB

        n = 1
        while np.min(mmm) == 0 and n > 0:
            n = 0
            for a in list(zip(*np.where(mmm==0xAB))):
                if a[1] > 0: 
                    if mmm[a[0],a[1]-1] == 0:
                        mmm[a[0],a[1]-1] = 0xAB
                        n = n+1
                if a[1] < self.model.height - 1:  
                    if mmm[a[0],a[1]+1] == 0:
                        mmm[a[0],a[1]+1] = 0xAB
                        n = n+1
                if a[0] > 0:  
                    if mmm[a[0]-1,a[1]] == 0:
                        mmm[a[0]-1,a[1]] = 0xAB
                        n = n+1
                if a[0] < self.model.width - 1:  
                    if mmm[a[0]+1,a[1]] == 0:
                        mmm[a[0]+1,a[1]] = 0xAB 
                        n = n+1
        # map2 = mmm
        # map2 = map2.T
        # map2 = np.flip(map2,0)
        # plt.subplot(1, 2, 2)
        # plt.imshow(map2, cmap='jet', interpolation='nearest')
        # plt.pause(0.01)

        current_num_obstacles = np.sum(self.pm_obstacles)

        self.pm_obstacles = (mmm != 0xAB) * 1

        new_num_obstacles = np.sum(self.pm_obstacles)

        return new_num_obstacles - current_num_obstacles

        