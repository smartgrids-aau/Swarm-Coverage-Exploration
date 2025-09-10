import mesa
import numpy as np
class Agent(mesa.Agent):
    # cover cells around the agent
    def cover_cells(self, pos, support_pm=True):
        covered_cells = self.model.grid.iter_neighbors((pos[0],pos[1]), moore=True, include_center=True, radius=int(np.ceil(self.model.coverage_radius)))
        for cell in covered_cells:
            # check if it's an agent, skip the agent
            if isinstance(cell, self.__class__):
                continue
            # check if direct distance is less than the coverage radius
            if np.sqrt((cell.pos[0]-pos[0])**2 + (cell.pos[1]-pos[1])**2) > self.model.coverage_radius:
                continue # if not, skip the cell

            # only for SCPP1 agents support private map
            if support_pm:
                # if the cell is an obstacle, mark it as an obstacle in the private map
                if cell.obstacle:
                    if self.pm_obstacles[cell.pos[0],cell.pos[1]] == 0:
                        self.available_cells -= 1
                        self.pm_obstacles[cell.pos[0],cell.pos[1]] = 1
                # for private map to know that it is covered
                if self.pm_visits[cell.pos[0],cell.pos[1]] == 0:
                    self.available_cells -= 1
                    self.pm_visits[cell.pos[0],cell.pos[1]] = 1
            
            cell.covered = True # for cell to know that it is covered
            cell.covered_by = self # for cell to know which agent covered it
