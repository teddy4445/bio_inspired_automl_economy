import os

RESULTS_FOLDER_NAME = "results"
DATA_FOLDER_NAME = "data"
RESULTS_FOLDER_PATH = os.path.join(os.path.dirname(__file__), RESULTS_FOLDER_NAME)
DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), DATA_FOLDER_NAME)
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), DATA_FOLDER_NAME, "config.csv")

RANDOM_STATE = 73  # Shaldon number is fund
META_DATASET_FIT = "meta_dataset_fit.csv"
META_DATASET_TRAIN = "meta_dataset_train.csv"
META_DATASET_TEST = "meta_dataset_test.csv"
