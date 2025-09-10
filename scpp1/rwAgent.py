import mesa

class rwAgent(mesa.Agent):
    def __init__(self, unique_id, model, pos, w=0.5, c1=2, c2=0, proximity=1):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.model = model

        self.moves = 0
        self.pos_history = []
        self.pos = pos
        self.vel = [1,1]

    def evaluate(self):
        self.fitness = 0

    def step(self):
        self.pos_history.append(self.pos)
        new_pos = [self.pos[0]+self.vel[0],self.pos[1]+self.vel[1]]
        # check if new position is within bounds or not
        if new_pos[0] >= 0 and new_pos[0] < self.model.width:
            if new_pos[1] >= 0 and new_pos[1] < self.model.height:
                if self.model.is_unoccupied(new_pos):
                        self.model.grid.move_agent(self, new_pos)
                        self.moves += 1
        self.vel = self.model.random.choice([[1,0],[0,1],[-1,0],[0,-1]])