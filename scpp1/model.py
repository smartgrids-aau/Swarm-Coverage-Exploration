import mesa
from .scppAgent import scpp1Agent
from .BM_Agent import BM_Agent
from .LF_Agent import LF_Agent
from .SwarmRW_Agent import SwarmRW_Agent
global Agent
from .Cell import Cell
import numpy as np
import matplotlib.pyplot as plt
import cv2
import distinctipy
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os

class model(mesa.Model):
    def coverage_percentage(self):
        visited_cells = 0

        for cell in self.schedule_cells.agents:
            if cell.visits > 0 and cell.obstacle == False:
                visited_cells += 1
        return visited_cells / self.available_cells * 100
        
    def energy_consumption(self):
        return sum([agent.moves for agent in self.schedule_agents.agents])
    
    def cells_collector(self):
        self.visits = 0
        self.ns_visits = 0
        self.overlapping_visits = 0
        for cell in self.schedule_cells.agents:
            self.visits += cell.visits
            self.ns_visits += cell.ns_visits
            if cell.ns_visits > 1:
                self.overlapping_visits += cell.ns_visits - 1

    def __init__(self, batch_run_mode,algorithm, N, world_size, map, work, stations, obstacle_rate, recovery_window, com_er, com_range, coverage_radius, w, C1, p1, p2, p3, sigma, lf_lambda,k,calculation_radius):
        super().__init__()

        self.batch_run_mode = batch_run_mode
        # termination condition: every cell has been visited at least once
        # this flag is set to True in every step and sets to False by any of the unvisited cells
        self.visits_done = False

        # Algorithm selection
        global Agent # to make sure that the Agent class is used from the selected algorithm
        # Options for the algorithm parameter:
        # SCPP1: Agent.py
        # BM RW: BM_Agent.py
        # LF RW: LF_Agent.py
        # Swarm RW: SwarmRW_Agent.py
        if algorithm == "SCPP1":
            Agent = scpp1Agent
        elif algorithm == "BM RW":
            Agent = BM_Agent
        elif algorithm == "LF RW":
            Agent = LF_Agent
        elif algorithm == "Swarm RW":
            Agent = SwarmRW_Agent

        self.num_agents = N
        self.schedule_agents = mesa.time.RandomActivation(self)
        self.schedule_cells = mesa.time.RandomActivation(self)
        self.steps = 0
        width,height = world_size[0],world_size[1]
        if map is None:
            # no map provided, create a obstacle-free map with the given dimensions
            self.width = width
            self.height = height
            # create a map with no obstacles
            self.map = np.zeros((width, height))
            self.with_obstacle = False
        else:
            # load the map from the file
            self.map = cv2.imread(map)
            # preprocess the map
            # set obstacles to 1 and free cells to 0
            self.map = cv2.cvtColor(self.map, cv2.COLOR_BGR2GRAY)
            self.map = np.flip(self.map,0)
            self.map = self.map < 1
            self.map = self.map.T
            # get the dimensions of the map and set the width and height
            self.width, self.height = np.shape(self.map)
            self.with_obstacle = True

        # create the grid
        self.grid = mesa.space.MultiGrid(self.width, self.height, False)
        
        agent_id = 0

        self.obstacle_positions = []
        for i in range(self.width):
            for j in range(self.height):
                if work > 0 and self.random.random() < stations:
                    w = self.random.randrange(work)
                else:
                    w = 0
                c = Cell(agent_id, self, [i,j],self.map[i,j], work=w)
                if self.random.random() < obstacle_rate:
                    c.obstacle = True

                    # check if any other obstacle is already placed in the neighborhood
                    for neighbor in self.grid.iter_neighbors((i,j), moore=True, include_center=False, radius=1):
                        for agent in self.grid.get_cell_list_contents(neighbor.pos):
                            if isinstance(agent, Cell):
                                if agent.obstacle:
                                    c.obstacle = False
                                    break
                if c.obstacle:
                    self.obstacle_positions.append([i,j])
                self.schedule_cells.add(c)
                self.grid.place_agent(c, [i,j])
                agent_id += 1
        
        self.communication_range = com_range
        self.com_er = com_er
        self.coverage_radius = coverage_radius 
        self.scpp1_params = [w, C1, p1, p2, p3]
        self.calculation_radius = calculation_radius
        agents_to_place = self.num_agents
        # mark one of the agents as chosen
        chosen = True
        while agents_to_place > 0:
            # pos = [self.random.randrange(self.width),self.random.randrange(self.height)]
            box_side = int(np.sqrt(self.num_agents)*4)
            pos = [int(self.random.randrange(box_side)+box_side/2-box_side/2),int(self.random.randrange(box_side)+box_side/2-box_side/2)]
            if self.is_unoccupied(pos):
                # put parameters in a dictionary
                params = {}
                if algorithm == "SCPP1":
                    params['w'] = w
                    params['C1'] = C1
                    params['p1'] = p1
                    params['p2'] = p2
                    params['p3'] = p3
                elif algorithm == "BM RW":
                    params['sigma'] = 1
                elif algorithm == "LF RW":
                    params['lambda'] = lf_lambda
                elif algorithm == "Swarm RW":
                    params['k'] = k

                a = Agent(agent_id, self, pos, params, chosen)
                self.schedule_agents.add(a)
                self.grid.place_agent(a, a.pos)
                agent_id += 1
                agents_to_place -= 1
                chosen = False

        self.running = True
        
        # prepare data for data collection
        # number of obstacles
        self.num_obstacles = 0
        for cell in self.schedule_cells.agents:
            if cell.obstacle:
               self.num_obstacles += 1
        # number of coverable cells
        self.available_cells = self.width * self.height - self.num_obstacles
        # total work required for all cells
        self.total_work = 0
        for cell in self.schedule_cells.agents:
            if cell.obstacle:
                continue
            if cell.work_copy == 0:
               self.total_work += 1
            else:
               self.total_work += cell.work_copy
        self.visits = 0
        self.ns_visits = 0
        self.overlapping_visits = 0
        self.coverage_curve_ttd = [0]
        self.coverage_curve_cvg = [0]
        self.datacollector = mesa.DataCollector(
            {"world_height": lambda self: self.height,
             "world_width": lambda self: self.width,
             "num_obstacles": lambda self: self.num_obstacles,
             "available_cells": lambda self: self.available_cells,
             "total_work": lambda self: self.total_work,
             "visits": lambda self: self.visits,
             "ns_visits": lambda self: self.ns_visits,
             "overlapping_visits": lambda self: self.overlapping_visits,
             "energy_consumption": self.energy_consumption,
             "ttd": lambda self: self.coverage_curve_ttd,
             "cvg": lambda self: self.coverage_curve_cvg,
             'percentage': self.coverage_percentage}
        )
        if self.batch_run_mode:
            self.schedule = mesa.time.RandomActivation(self) # just for data collection in bachrun mode
        self.schedule_cells.step() # mark the initial visits
        self.coverage_curve_cvg[-1] /= self.available_cells / 100

        # agents_ is a list of agents in the order of their unique_id
        # to keep the order for color consistency
        self.agents_ = list(self.schedule_agents.agents)
        # fix colors for agents
        self.colors = distinctipy.get_colors(self.num_agents)
        # clear time_lapse folder
        # for f in os.listdir('time_lapse'):
        #     os.remove(os.path.join('time_lapse', f))
        self.chosen_agent = None
        for agent in self.agents_:
            if agent.chosen:
                self.chosen_agent = agent

        # if not self.batch_run_mode:
        #     # Create a named window
        #     self.sample_agent_window = 'Sample agent'
        #     cv2.namedWindow(self.sample_agent_window, cv2.WINDOW_NORMAL)  # Allows resizing
        #     # key = cv2.waitKey(0) & 0xFF # wait for window resizing and replacement
        #     self.fitness_maps_window = 'Fitness maps'
        #     cv2.namedWindow(self.fitness_maps_window, cv2.WINDOW_NORMAL)  # Allows resizing
        #     # key = cv2.waitKey(0) & 0xFF # wait for window resizing and replacement
        #     # create a fixed order of agents
        #     self.agents_ = list(self.schedule_agents.agents)
        #     self.do_plots()
    def do_plots(self):
        # create subplots for fitness maps for all agents
        rows = int(np.ceil(np.sqrt(self.num_agents)))
        cols = int(np.ceil(self.num_agents / rows))
        fig, axs = plt.subplots(rows, cols, figsize=(8, 10))
        # no axis
        for ax in axs.flat:
            ax.axis('off')

        # for the chosen agent, plot lambda, gamma and phi
        index = 0
        self.chosen_agent = None
        for agent in self.agents_:
            if agent.chosen:
                self.chosen_agent = agent
        
            if not agent.is_on_path:
                agent.map_evaluation()
            
            var = agent.pm_fitness
            var = np.array(var)
            var = var.T
            var = np.flip(var, 0)
            if cols == 1:
                ax = axs[index]
            else:
                ax = axs[index // cols, index % cols]
            im = ax.imshow(var, cmap='jet', interpolation='nearest')
            ax.set_title(f'Agent {agent.unique_id}')
            index += 1

        plt.tight_layout()

        # Render the plot to a numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        image_array = np.array(canvas.renderer.buffer_rgba())

        # Convert numpy array to CV2 image
        image_cv2 = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)

        # Show the image in the OpenCV window
        cv2.imshow(self.fitness_maps_window, image_cv2)

        fig, axs = plt.subplots(2, 2, figsize=(8, 10))
        variables = [self.chosen_agent.alpha, self.chosen_agent.beta, 1/self.chosen_agent.gamma, self.chosen_agent.pm_fitness]
        titles = ['lambda', 'gamma', '1/phi', 'fitness map']
        for i, (var, title) in enumerate(zip(variables, titles)):
            var = np.array(var)
            var = var.T
            var = np.flip(var, 0)
            ax = axs[i // 2, i % 2]
            ax.axis('off')
            im = ax.imshow(var, cmap='jet', interpolation='nearest')
            ax.set_title(title)

        plt.tight_layout()

        # Render the plot to a numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        image_array = np.array(canvas.renderer.buffer_rgba())

        # Convert numpy array to CV2 image
        image_cv2 = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)

        # Show the image in the OpenCV window
        cv2.imshow(self.sample_agent_window, image_cv2)
        key = cv2.waitKey(1) & 0xFF

    def step(self):
        if not self.visits_done:            
            # place a 0 so that robots add their total moves to it
            self.coverage_curve_ttd.append(0)
            # place a 0 so that cells can mark it if they are visited, at the end it can be used to compute the coverage percentage
            self.coverage_curve_cvg.append(0)

            self.schedule_agents.step()

            # initially set termination condition to True
            # any cell with visits = 0 (unvisited) will set this to False
            self.visits_done = True
            self.schedule_cells.step()

            # compute the coverage percentage by deviding the last element of the coverage_curve_cvg by the number of available cells
            self.coverage_curve_cvg[-1] /= self.available_cells / 100

            self.steps += 1
            if self.batch_run_mode:
                self.schedule.step()
                self.cells_collector()
                self.datacollector.collect(self)
        else: 
        # coverage is completed
            ## collect data
            self.cells_collector()
            # data collection, plot generation and other final steps
            if self.batch_run_mode:
                self.datacollector.collect(self)
            
            else: # GUI mode
                ## text report
                coverage_time = self.steps/self.available_cells*100
                print(f"Simulation completed in {self.steps} steps that is {coverage_time:.2f} % compared the optimal single agent coverage time. ({100-coverage_time:.2f} % saved!)")
                energy_consumption = self.energy_consumption() / (self.available_cells - self.num_agents)*100
                print(f"Energy consumption was {energy_consumption} % compared to the optimal single agent coverage energy consumption. ({energy_consumption-100:.2f} % extra energy consumed!)")
                overlaps = self.overlapping_visits / (self.width * self.height) * 100
                print(f"Overlapping visits on {overlaps:.2f} % of the area.")
                ## plot visits as a heatmap

                # visits = np.zeros((self.height, self.width))
                # for cell in self.schedule_cells.agents:
                #     visits[cell.col, cell.row] = cell.ns_visits
                # # flip the map vertically
                # visits = np.flip(visits,0)
                # plt.imshow(visits, cmap='hot', interpolation='nearest')
                # plt.waitforbuttonpress()


                # save the coverage path as an image
                plt.gca().set_aspect('equal', adjustable='box')
                for pos in self.obstacle_positions:
                    plt.scatter(pos[0], pos[1], color='black', marker='s', s=70)
                colors = distinctipy.get_colors(len(self.schedule_agents.agents))
                bot_index = 0
                for agent in self.schedule_agents.agents:
                    pos_history = np.array(agent.pos_history)
                    plt.plot(pos_history[:,0],pos_history[:,1], color=colors[bot_index], linestyle='dashed')
                    bot_index += 1
                plt.savefig("coverage_path.png", dpi=300)
                plt.clf()
            self.running = False

        # not in batch run mode
        # if not self.batch_run_mode:
        #     # every 5 steps save the coverage path as an image to the time_lapse folder
        #     if self.steps % 10 == 0 or self.visits_done:

        #         plt.gca().set_aspect('equal', adjustable='box')
        #         fig, axs = plt.subplots(figsize=(10, 10))
        #         for pos in self.obstacle_positions:
        #             plt.scatter(pos[0], pos[1], color='black', marker='s', s=200)
        #         bot_index = 0
        #         for agent in self.agents_:
        #             # it it's finished, ask agents to update their fitness maps   
        #             if self.visits_done:
        #                 agent.step()
        #             pos_history = np.array(agent.pos_history)
        #             plt.plot(pos_history[:,0],pos_history[:,1], color=self.colors[bot_index], linestyle='dashed', linewidth=4)
        #             # mark the last position of the agent with a box
        #             plt.scatter(pos_history[-1,0], pos_history[-1,1], color=self.colors[bot_index], marker='s', s=20)
        #             bot_index += 1
        #         # set x and y axis limits to the map dimensions
        #         plt.xlim(-1,self.width)
        #         plt.ylim(-1,self.height)
        #         plt.tight_layout()
        #         plt.savefig(f"time_lapse/coverage_path{self.steps}.png", dpi=300)
        #         plt.clf()
                
        #         # create subplots for fitness maps a chosen agent
        #         plt.gca().set_aspect('equal', adjustable='box')
        #         fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        #         # font size
        #         plt.rcParams.update({'font.size': 22})
        #         variables = [self.chosen_agent.alpha_, self.chosen_agent.beta_, self.chosen_agent.gamma_, self.chosen_agent.pm_fitness]
        #         titles = [r"$A$", r"$\bar{D}$", 'D', 'F']
        #         for i, (var, title) in enumerate(zip(variables, titles)):
        #             var = np.array(var)
        #             var = var.T
        #             var = np.flip(var, 0)
        #             ax = axs[i // 2, i % 2]
                    
        #             ax.axis('off')
        #             im = ax.imshow(var, cmap='jet', interpolation='nearest')
        #             ax.set_title(title)

        #         # add one colorbar to the right of all subplots
        #         cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=.1)
        #         cbar.ax.yaxis.set_ticks_position('right')
        #         # no ticks
        #         cbar.ax.set_yticklabels([])
        #         # plt.tight_layout()

        #         plt.savefig(f"time_lapse/fitness{self.steps}.png", dpi=300)
        #         plt.clf()

        #         # release the memory
        #         plt.close('all')
        #         # clean up any plt memory
        #         plt.cla()
        #         plt.clf()

    def is_obstacle (self, pos):
        for agent in self.grid.get_cell_list_contents(pos):
            if isinstance(agent, Cell):
                return agent.obstacle
        return False

    # TODO: check if this is used
    def toggle_on_path (self, pos, state):
        for agent in self.grid.get_cell_list_contents(pos):
            if isinstance(agent, Cell):
                agent.is_on_path = state

    def is_unoccupied (self, pos):
        global Agent
        for agent in self.grid.get_cell_list_contents(pos):
            if isinstance(agent, Agent):
                return False
            if isinstance(agent, Cell):
                if agent.obstacle:
                    return False
        return True
    
    def is_there_agent (self, pos):
        for agent in self.grid.get_cell_list_contents(pos):
            if isinstance(agent, Agent):
                return True
        return False
        
    def needs_work(self, pos):
        for agent in self.grid.get_cell_list_contents(pos):
            if isinstance(agent, Cell):
                return agent.work > 0
        return False

    def get_neighbors(self, caller_agent):
        neighbors = []
        # iterate over agents
        for agent in self.schedule_agents.agents:
            if agent.unique_id != caller_agent.unique_id:
                # direct distance
                distance = np.sqrt((agent.pos[0] - caller_agent.pos[0])**2 + (agent.pos[1] - caller_agent.pos[1])**2)
                if distance <= self.communication_range or self.communication_range == 0:
                    neighbors.append(agent)
        return neighbors