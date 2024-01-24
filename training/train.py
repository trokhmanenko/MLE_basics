"""
This script prepares the data, runs the training, and saves the model.
"""

import argparse
import os
import sys
import json
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import get_project_dir, configure_logging
import mlflow
from dotenv import load_dotenv
import datetime

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
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file",
                    help="Specify inference data file",
                    default=conf['train']['table_name'])
parser.add_argument("--model_path",
                    help="Specify the path for the output model")


class DataProcessor:
    def __init__(self) -> None:
        pass

    @staticmethod
    def prepare_data(path: str, max_rows: int = None) -> pd.DataFrame:
        logging.info("Preparing data for training...")
        df = pd.read_csv(path)
        if max_rows is not None:
            df = df.sample(n=min(max_rows, len(df)), random_state=conf['general']['random_state'])
        df = DataProcessor.normalize_data(df)
        return df

    @staticmethod
    def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = ['sepal length', 'sepal width', 'petal length', 'petal width']  # Захардкоденные названия колонок
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        return df


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


class Training:
    def __init__(self) -> None:
        self.model = SimpleNet(input_size=4, hidden_size=10, num_classes=3)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)

    def run_training(self, df: pd.DataFrame, test_size: float = 0.33, epochs: int = 10) -> None:
        logging.info("Running training...")
        x_train, x_test, y_train, y_test = self.data_split(df, test_size)
        train_loader = DataLoader(TensorDataset(torch.tensor(x_train.values, dtype=torch.float32),
                                                torch.tensor(y_train.values, dtype=torch.long)),
                                  batch_size=64, shuffle=True)

        for epoch in range(epochs):
            self.train_epoch(train_loader)
            logging.info(f'Epoch [{epoch + 1}/{epochs}] completed.')

        self.test(torch.tensor(x_test.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.long))

    @staticmethod
    def data_split(df: pd.DataFrame, test_size: float) -> tuple:
        feature_cols = ['sepal length', 'sepal width', 'petal length', 'petal width']
        x = df[feature_cols]
        y = df['target']
        return train_test_split(x, y, test_size=test_size, random_state=conf['general']['random_state'])

    def train_epoch(self, train_loader):
        self.model.train()
        for inputs, labels in train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def test(self, x_test: torch.Tensor, y_test: torch.Tensor) -> None:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_test)
            _, predicted = torch.max(outputs.data, 1)
            f1 = f1_score(y_test.numpy(), predicted.numpy(), average='weighted')
            logging.info(f"f1_score: {f1}")

    def save(self, path: str = None) -> None:
        logging.info("Saving the model...")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        filename = datetime.datetime.now().strftime(conf['general']['datetime_format']) + '.pth'
        save_path = os.path.join(MODEL_DIR, filename)

        if path:
            save_path = os.path.join(MODEL_DIR, path + "_" + filename)

        torch.save(self.model.state_dict(), save_path)


def main():
    configure_logging()

    mlflow_tracking_uri = 'file://' + os.path.abspath(os.path.join(ROOT_DIR, '..', 'mlruns'))
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    mlflow.autolog()

    data_proc = DataProcessor()
    df = data_proc.prepare_data(TRAIN_PATH, max_rows=conf['train']['data_sample'])

    tr = Training()
    tr.run_training(df, test_size=conf['train']['test_size'], epochs=conf['train']['epochs'])
    tr.save()


if __name__ == "__main__":
    main()
