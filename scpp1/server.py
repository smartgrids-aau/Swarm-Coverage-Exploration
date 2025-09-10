import mesa
from .model import model
from .Cell import Cell
import cv2
import numpy as np

# -----------Map-------------------
# map = 'scpp1/obstacle_maps/sensors_paper/map5.png'
# map = 'scpp1\obstacle_maps\map1.png'
# map = 'scpp1\obstacle_maps\map2.png'
# map = 'scpp1\obstacle_maps\map3.png'
# map = 'scpp1\obstacle_maps\map4.png'
map = None
# ----------------------------------

if map is not None:
    # load the map from the file
    map_data = cv2.imread(map)
    # get the dimensions of the map
    grid_cols, grid_rows, _ = np.shape(map_data)
else:
    grid_rows = 50
    grid_cols = 50

# set cell size adaptively
cell_size = 700 // max(grid_rows, grid_cols)

canvas_width = grid_rows * cell_size
canvas_height = grid_cols * cell_size

def agent_portrayal(agent):
    if agent is None:
        raise AssertionError
    if isinstance(agent, Cell):
        portrayal = {"Shape": "rect", "w": 1, "h": 1, "Filled": "true", "Layer": 0}
        portrayal["x"] = agent.row
        portrayal["y"] = agent.col
        if agent.is_on_path and False:
            if agent.obstacle:
                portrayal["Color"] = "#550000"
            else:
                portrayal["Color"] = "#FF0000"
        else:
            if agent.obstacle:
                portrayal["Color"] = "#000000"
            else:
                if agent.visits == 0:
                    if agent.work == 0:
                        portrayal["Color"] = "#FFFFFF"
                    elif agent.work <= 5:
                        # light orange
                        portrayal["Color"] = "#FFA500"
                    else:
                        # red
                        portrayal["Color"] = "#FF0000"
                else:
                    portrayal["Color"] = "#A897A8"
        return portrayal
    else:
        portrayal = {"Shape": "circle", "r": 0.5,"Filled": "true", "Layer": 1}
        portrayal["Color"] = "#A60325"
        return portrayal
canvas_element = mesa.visualization.CanvasGrid(
    agent_portrayal, grid_rows, grid_cols, canvas_width, canvas_height
)

model_params = {
    "batch_run_mode": False, # if True, the model will skip the plotting and text reporting,
    "algorithm": mesa.visualization.Choice(
        name="Algorithm",
        choices=["SCPP1", "BM RW", "LF RW", "Swarm RW"], # Swarm RW is based on the paper https://doi.org/10.1155/2019/6914212
        value="SCPP1"
    ),
    "N": mesa.visualization.Slider(
        name="Number of agents", value=4, min_value=2, max_value=100, step=1
    ),
    "world_size": [grid_rows,grid_cols],
    "map":map,
    "stations": mesa.visualization.Slider(
        name="Stations", value=0.01, min_value=0, max_value=0.25, step=0.001
    ),
    "work": mesa.visualization.Slider(
        name="Work steps", value=0, min_value=0, max_value=10, step=1
    ),
    "obstacle_rate": mesa.visualization.Slider(
        name="Obstacle chance", value=0, min_value=0, max_value=0.3, step=0.01
    ),
    "recovery_window": mesa.visualization.Slider(
        name="Recovery window", value=1, min_value=1, max_value=100, step=1
    ),
    "com_er": mesa.visualization.Slider(
        name="Communication error rate", value=0, min_value=0, max_value=1, step=0.01
    ),
    "com_range": mesa.visualization.Slider(
        name="Communication range", value=0, min_value=0, max_value=100, step=1
    ),
    "coverage_radius": mesa.visualization.Slider(
        name="Coverage radius", value=2, min_value=1, max_value=10, step=0.1
    ),
    "w": mesa.visualization.Slider(
        name="inertia (w)", value=0.5, min_value=0, max_value=0.5, step=0.00005
    ),
    "C1": mesa.visualization.Slider(
        name="C1", value=2, min_value=0, max_value=20, step=0.5
    ),
    "p1": mesa.visualization.Slider(
        name="p1", value=60, min_value=5, max_value=100, step=5
    ),
    "p2": mesa.visualization.Slider(
        name="p2", value=2, min_value=0, max_value=3, step=0.1
    ),
    "p3": mesa.visualization.Slider(
        name="p3", value=2, min_value=0, max_value=3, step=0.1
    ),
    "sigma": mesa.visualization.Slider(
        name="BM_sigma", value=2, min_value=0, max_value=3, step=0.1
    ),
    "lf_lambda": mesa.visualization.Slider(
        name="LF_lambda", value=1.5, min_value=1, max_value=3, step=0.1
    ),
    "k": mesa.visualization.Slider(
        name="SwarmRW_k", value=0.5, min_value=0, max_value=1, step=0.1
    ),
    "calculation_radius": mesa.visualization.Slider(
        name="calculation radius", value=10, min_value=4, max_value=100, step=2
    ),
    
}
server = mesa.visualization.ModularServer(
    model_cls = model,
    model_params = model_params,
    visualization_elements = [canvas_element],
    name = "Swarm-based Coverage Path Planning (SCPP1)",
)
