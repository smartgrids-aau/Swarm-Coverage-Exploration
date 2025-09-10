import mesa
class Cell(mesa.Agent):
    def __init__(self, unique_id, model, pos, obstacle, work):
        super().__init__(unique_id, model)
        self.is_on_path = False
        self.row = pos[0]
        self.col = pos[1]
        self.obstacle = obstacle == 1
        # work required to move through this cell
        self.work = work
        self.work_copy = work
        self.last_agent = None
        self.ns_visits = 0 # when robot remains in the cell, this is not increases
        self.visits = 0 #whether the robot remained in the cell or a new robot came, this is increases
        self.covered = False # robots change this to True when they cover the cell
        self.covered_by = None
    def step(self):
        if self.covered:
            self.visits += 1
        self.covered = False

        # if self.work > 0:
        #     for agent in self.model.grid.get_cell_list_contents([self.pos]):
        #         if isinstance(agent, Agent):
        #             self.work -= 1
        # if self.work > 0:
        #     return # if work is not done, visits will not be counted

        # for ns_visits, only increment if the agent is different from the last one
        if self.covered_by is not None:
            if self.last_agent is None:
                self.last_agent = self.covered_by
                self.ns_visits += 1
            else:
                if self.last_agent.unique_id != self.covered_by.unique_id:
                    self.ns_visits += 1
                    self.last_agent = self.covered_by
        else:
            self.last_agent = None

        # if unvisited, set model's termination flag to False
        if self.visits == 0 and not self.obstacle:
            self.model.visits_done = False
        else:
            # if visited, increment the coverage count
            self.model.coverage_curve_cvg[-1] += 1
        