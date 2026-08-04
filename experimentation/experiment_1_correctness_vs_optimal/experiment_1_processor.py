"""Process experiment 1 JSON results into analysis tables."""


import json
from pathlib import Path
from typing import Dict

import pandas as pd
def load_experiments(config: Dict, experiment_result_path: Path, base_file_name: str):
    experiment_info = []
    experiments_loaded = {}
    for selection in config["selections"]:
        selection_file_name = base_file_name + f'd{selection["dits"]}_k{selection["k"]}.json'
       
        if selection_file_name in experiments_loaded:
            experiment = experiments_loaded[selection_file_name]
        else:
            experiment_selection_path = experiment_result_path / selection_file_name
            if experiment_selection_path.exists():
                with experiment_selection_path.open(encoding="utf-8") as f:
                    experiment = json.load(f)
                experiments_loaded[selection_file_name] = experiment
            else:
                raise ValueError(f"No se ha encontrado el experimento {experiment_selection_path}")
        
        n_variables = selection["n_variables"] 
        summary_entry_matrix = next(
            item for item in experiment["matrix_method_summary"]
            if item["n_variables"] == n_variables
        )
        summary_entry_heuristic = next(
            item for item in experiment["heuristic_summary"]
            if item["n_variables"] == n_variables
        )
        experiment_info.append({
            "method": "matrix_method",
            "dits": selection["dits"],
            "k": selection["k"],
            **summary_entry_matrix,
        })
        experiment_info.append({
            "method": "heuristic_method",
            "dits": selection["dits"],
            "k": selection["k"],
            **summary_entry_heuristic,
        })
    return experiment_info
 
def experiments_dataframe(experiments):
    return pd.DataFrame(experiments)
def main() -> None:

    CONFIG_PATH = Path(__file__).resolve().parent / "emperiment_configuration.json"
    EXPERIMENT_RESULTS_PATH = Path(__file__).resolve().parent / "results"
    BASE_EXPERIMENT_FILE_NAME = "experiment_1_params_"

    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    
    experiments = load_experiments(config, EXPERIMENT_RESULTS_PATH, BASE_EXPERIMENT_FILE_NAME)
    experiments = experiments_dataframe(experiments)
    print(experiments)
if __name__ == "__main__":
    main()
