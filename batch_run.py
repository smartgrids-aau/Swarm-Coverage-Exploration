from mesa.batchrunner import batch_run
from multiprocessing import freeze_support
from scpp1.model import model
import pandas as pd
import warnings
import numpy as np
import os

warnings.simplefilter("ignore", category=FutureWarning)


expriment_name = "paper_R_c_SS"
model_params = {
    "batch_run_mode": True,
    "algorithm": ["SCPP1"],
    "N": [10],
    "world_size": [[20,20]],
    "map": None,
    "recovery_window": 10,
    "com_er": 0,
    "com_range": 0,
    "stations": 0,
    "work": 0,
    "obstacle_rate": 0,
    "coverage_radius": 3,
    "w": 0.5,
    "C1": 2,
    "p1": 60,
    "p2": 2,
    "p3": 2,
    "sigma": 1,
    "lf_lambda": 1.5,
    "k": 0.75,
}

if __name__ == '__main__':
    freeze_support()
    results = batch_run(
        model,
        parameters=model_params,
        iterations=50,
        max_steps=3000,
        number_processes=None,
        display_progress=True,
        )
    
    results_df = pd.DataFrame(results)
    # save to csv file under scpp1/results with a unique name based on the date and time

    # Retrieve the path from the environment variable
    results_path = os.environ.get("RESULTS_PATH")

    # result file name
    csv_file_name = f'{expriment_name}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
    
    csv_file = os.path.join(results_path, csv_file_name)
    
    results_df.to_csv(csv_file)