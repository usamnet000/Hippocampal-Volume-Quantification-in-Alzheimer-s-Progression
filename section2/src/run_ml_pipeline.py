"""
This file contains code that will kick off training and testing processes
"""
import os
import json
import random

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = r"../data"
        self.n_epochs = 1
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = "test_results"

if __name__ == "__main__":
    # Get configuration
    c = Config()

    # Load data
    print("Loading data...")

    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)

    # create three keys in the dictionary ("train", "val" and "test") with each key storing the array with indices 
    split = dict()
    n = len(data)
    idx_list = list(range(n))
    random.shuffle(idx_list)
    split['train'] = idx_list[ :int(n * 0.7)]
    split['val'] = idx_list[int(n * 0.7) : int(n * 0.85)]
    split['test'] = idx_list[int(n * 0.85):]  

    # Set up and run experiment
    exp = UNetExperiment(c, split, data)

    # run training
    exp.run()

    # prep and run testing
    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))

