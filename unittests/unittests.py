import unittest
import pandas as pd
import os
import sys
import json
from training.train import DataProcessor, Training, SimpleNet
from dotenv import load_dotenv
from utils import get_project_dir

load_dotenv()

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Define the path to the project root directory
PROJECT_ROOT_DIR = get_project_dir('')

# Load configuration settings from JSON
CONF_FILE = os.getenv('CONF_PATH')
CONF_PATH = os.path.join(PROJECT_ROOT_DIR, CONF_FILE)
with open(CONF_PATH, "r") as fi:
    c = json.load(fi)
MODEL_DIR = get_project_dir(c['general']['models_dir'])


class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONF_PATH, "r") as file:
            conf = json.load(file)
        cls.train_path = os.path.join(conf['general']['data_dir'], conf['train']['table_name'])

    def test_prepare_data(self):
        dp = DataProcessor()
        df = dp.prepare_data(self.train_path, max_rows=100)
        self.assertEqual(df.shape[0], 100)
        self.assertIn('sepal length', df.columns)
        self.assertIn('target', df.columns)

    def test_normalize_data(self):
        dp = DataProcessor()
        df = pd.DataFrame({
            'sepal length': [5.1, 4.9, 4.7],
            'sepal width': [3.5, 3.0, 3.2],
            'petal length': [1.4, 1.4, 1.3],
            'petal width': [0.2, 0.2, 0.2]
        })
        normalized_df = dp.normalize_data(df)
        self.assertAlmostEqual(normalized_df.mean().sum(), 0, places=5)
        self.assertAlmostEqual(normalized_df.std().sum(), len(df.columns), places=5)


class TestTraining(unittest.TestCase):
    def test_data_split(self):
        tr = Training()
        df = pd.DataFrame({
            'sepal length': [5.1, 4.9, 4.7, 5.0],
            'sepal width': [3.5, 3.0, 3.2, 3.6],
            'petal length': [1.4, 1.4, 1.3, 1.5],
            'petal width': [0.2, 0.2, 0.2, 0.4],
            'target': [0, 0, 1, 1]
        })
        x_train, x_test, y_train, y_test = tr.data_split(df, test_size=0.5)
        self.assertEqual(len(x_train), 2)
        self.assertEqual(len(x_test), 2)

    def test_run_training(self):
        tr = Training()
        df = pd.DataFrame({
            'sepal length': [5.1, 4.9, 4.7, 5.0],
            'sepal width': [3.5, 3.0, 3.2, 3.6],
            'petal length': [1.4, 1.4, 1.3, 1.5],
            'petal width': [0.2, 0.2, 0.2, 0.4],
            'target': [0, 0, 1, 1]
        })
        tr.run_training(df, test_size=0.5, epochs=5)
        self.assertTrue(any(p.numel() > 0 for p in tr.model.parameters()))

    def test_save_model(self):
        tr = Training()
        tr.model = SimpleNet(input_size=4, hidden_size=10, num_classes=3)
        tr.save()
        saved_models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')]
        self.assertTrue(len(saved_models) > 0)


if __name__ == '__main__':
    unittest.main()
