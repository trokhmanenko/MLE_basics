# MLE_basics
Welcome to the `MLE_basics` project! This repository is dedicated to processing and analyzing the famous Iris flower dataset. The Iris dataset is a multivariate dataset introduced by the British statistician and biologist Ronald Fisher in 1936, which has since become a staple in pattern recognition and machine learning demonstrations.

## Project structure:

This project has a modular structure, where each folder has a specific duty.

```
MLE_basics
├── data                      # Contains datasets for training and inference
│   ├── iris_inference_data.csv  # Data to be used for inference
│   └── iris_train_data.csv      # Data to be used for training the model
├── data_process              # Scripts for data preprocessing and augmentation
│   ├── data_generation.py    # Script to generate and preprocess the data
│   └── __init__.py           # Marks the directory as a Python package
├── inference                 # Inference-related scripts and Docker environment files
│   ├── Dockerfile            # Dockerfile to create a container for inference
│   ├── run.py                # Script to run the inference process
│   └── __init__.py           # Marks the directory as a Python package
├── models                    # Directory where the trained model files are stored
│   └── [Model files will be stored here]
├── results                   # Directory for storing results from inference
├── training                  # Contains training scripts and Docker environment files
│   ├── Dockerfile            # Dockerfile to create a container for training
│   ├── train.py              # Script to train the machine learning model
│   └── __init__.py           # Marks the directory as a Python package
├── unittests                 # Contains unit tests for the project's modules
│   └── unittests.py          # Script containing unit tests
├── .env                      # Environment variables for the project
├── .gitignore                # Specifies files to be ignored by version control
├── __init__.py               # Marks the root directory as a Python package
├── README.md                 # The project's README file with detailed information
├── requirements.txt          # Lists all the dependencies to be installed
├── settings.json             # Stores configurable parameters and settings for the project
└── utils.py                  # Utility functions and classes used throughout the project
```

## Data generation:
To prepare the Iris dataset, execute the script:
`python data_process/data_generation.py`

This will generate:
- `data/iris_train_data.csv` for training
- `data/iris_inference_data.csv` for inference

Modify `settings.json` for any custom configurations.

## Data:
Data is the cornerstone of any Machine Learning project. For generating the data, use the script located at `data_process/data_generation.py`. The generated data is used to train the model and to test the inference. Following the approach of separating concerns, the responsibility of data generation lies with this script.

## Training:
The training phase of the ML pipeline includes preprocessing of data, the actual training of the model, and the evaluation and validation of the model's performance. All of these steps are performed by the script `training/train.py`.

1. To train the model using Docker: 

- Build the training Docker image. If the built is successfully done, it will automatically train the model:
```bash
docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t training_image .
```
- You may run the container with the following parameters to ensure that the trained model is here:
```bash
docker run -it training_image /bin/bash
```
Then, move the trained model from the directory inside the Docker container `/app/models` to the local machine using:
```bash
docker cp <container_id>:/app/models/<model_name>.pickle ./models
```
Replace `<container_id>` with your running Docker container ID and `<model_name>.pickle` with your model's name.

1. Alternatively, the `train.py` script can also be run locally as follows:

```bash
python3 training/train.py
```

## Inference:
Once a model has been trained, it can be used to make predictions on new data in the inference stage. The inference stage is implemented in `inference/run.py`.

1. To run the inference using Docker, use the following commands:

- Build the inference Docker image:
```bash
docker build -f ./inference/Dockerfile --build-arg model_name=<model_name>.pickle --build-arg settings_name=settings.json -t inference_image .
```
- Run the inference Docker container:
```bash
docker run -v /path_to_your_local_model_directory:/app/models -v /path_to_your_input_folder:/app/input -v /path_to_your_output_folder:/app/output inference_image
```
- Or you may run it with the attached terminal using the following command:
```bash
docker run -it inference_image /bin/bash  
```
After that ensure that you have your results in the `results` directory in your inference container.

2. Alternatively, you can also run the inference script locally:

```bash
python inference/run.py
```

Replace `/path_to_your_local_model_directory`, `/path_to_your_input_folder`, and `/path_to_your_output_folder` with actual paths on your local machine or network where your models, input, and output are stored.
