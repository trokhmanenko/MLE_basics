import numpy as np
import pandas as pd
import logging
import os
import sys
import json
from sklearn.datasets import load_iris
from utils import singleton, get_project_dir, configure_logging
from dotenv import load_dotenv

load_dotenv()

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the path to the project root directory
PROJECT_ROOT_DIR = get_project_dir('')

# Load configuration settings from JSON
CONF_FILE = os.getenv('CONF_PATH')
CONF_PATH = os.path.join(PROJECT_ROOT_DIR, CONF_FILE)
logger.info("Loading configuration settings from JSON...")
with open(CONF_PATH, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
DATA_DIR = get_project_dir(conf['general']['data_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])


@singleton
class IrisDataProcessor:
    def __init__(self):
        self.data = None
        self.target = None

    def load_data(self):
        logger.info("Loading Iris dataset...")
        iris = load_iris()
        self.data = iris.data
        self.target = iris.target

    def split_data(self):
        logger.info("Splitting data into training and inference sets...")
        inference_size = int(0.1 * len(self.data))
        indices = np.arange(len(self.data))
        np.random.shuffle(indices)

        inference_indices = indices[:inference_size]
        train_indices = indices[inference_size:]

        return (self.data[train_indices], self.target[train_indices]), (
            self.data[inference_indices], self.target[inference_indices])

    @staticmethod
    def save_data(data, target, path):
        logger.info(f"Saving data to {path}...")
        df = pd.DataFrame(data, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
        df['target'] = target
        df.to_csv(path, index=False)


# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting script...")
    iris_processor = IrisDataProcessor()
    iris_processor.load_data()
    train_set, inference_set = iris_processor.split_data()
    iris_processor.save_data(*train_set, TRAIN_PATH)
    iris_processor.save_data(*inference_set, INFERENCE_PATH)
    logger.info("Script completed successfully.")
