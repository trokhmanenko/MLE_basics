"""
Script loads the latest trained model, data for inference and predicts results.
Imports necessary packages and modules.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
from utils import get_project_dir, configure_logging
from dotenv import load_dotenv

load_dotenv()

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Define the path to the project root directory
PROJECT_ROOT_DIR = get_project_dir('')

# Load configuration settings from JSON
CONF_FILE = os.getenv('CONF_PATH')
CONF_PATH = os.path.join(PROJECT_ROOT_DIR, CONF_FILE)
with open(CONF_PATH, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file",
                    help="Specify inference data file",
                    default=conf['inference']['inp_table_name'])
parser.add_argument("--out_path",
                    help="Specify the path to the output table")


def get_latest_model_path() -> str:
    """Gets the path of the latest saved model"""
    latest = None
    for (dirpath, dirnames, filenames) in os.walk(MODEL_DIR):
        for filename in filenames:
            if not latest or datetime.strptime(latest, conf['general']['datetime_format'] + '.pth') < \
                    datetime.strptime(filename, conf['general']['datetime_format'] + '.pth'):
                latest = filename
    return os.path.join(MODEL_DIR, latest)


class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def get_model_by_path(path: str) -> nn.Module:
    """Loads and returns the specified PyTorch model"""
    model = SimpleNet(input_size=4, hidden_size=10, num_classes=3)
    try:
        model.load_state_dict(torch.load(path))
        model.eval()
        logging.info(f'Path of the model: {path}')
        return model
    except Exception as e:
        logging.error(f'An error occurred while loading the model: {e}')
        sys.exit(1)


def get_inference_data(path: str) -> pd.DataFrame:
    """loads and returns data for inference from the specified csv file"""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f"An error occurred while loading inference data: {e}")
        sys.exit(1)


def predict_results(model: nn.Module, infer_data: pd.DataFrame) -> pd.DataFrame:
    """Predict the results using a PyTorch model"""
    features = infer_data[['sepal length', 'sepal width', 'petal length', 'petal width']].values
    features = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        results = model(features)
        _, predicted = torch.max(results.data, 1)
    infer_data['results'] = predicted.numpy()
    return infer_data


def store_results(results: pd.DataFrame, path: str = None) -> None:
    """Store the prediction results in 'results' directory with current datetime as a filename"""
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        path = datetime.now().strftime(conf['general']['datetime_format']) + '.csv'
        path = os.path.join(RESULTS_DIR, path)
    results.to_csv(path, index=False)
    logging.info(f'Results saved to {path}')


def main():
    """Main function"""
    configure_logging()
    args = parser.parse_args()

    model = get_model_by_path(get_latest_model_path())
    infer_file = args.infer_file
    infer_data = get_inference_data(os.path.join(DATA_DIR, infer_file))
    results = predict_results(model, infer_data)
    store_results(results, args.out_path)

    logging.info(f'Prediction results: {results}')


if __name__ == "__main__":
    main()
