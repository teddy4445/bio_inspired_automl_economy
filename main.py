# library imports
import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, LogisticRegression # linear model for regression and classification
from dl85 import DL85Classifier, DL85Predictor  # Optimal decision tree
import autosklearn.classification # AutoML baseline - https://automl.github.io/auto-sklearn/master/#manual  [classification]
import autosklearn.regression # AutoML baseline - https://automl.github.io/auto-sklearn/master/#manual [regression]

from tpot import TPOTClassifier, TPOTRegressor # bio-inspired autoML
from gplearn.genetic import SymbolicRegressor # bio-inspired autoML #2
from scimed import SciMed # bio-inspired autoML #3
# TODO: add later # bio-inspired autoML #4


# project imports
from consts import *


# define the model's score metric for each case so it will be easy to replace later #
def metric(y_pred, y):
    if len(set(y)) / len(y) > 0.1:
        return regression_metric(y_pred=y_pred,
                                 y=y)
    return classification_metric(y_pred=y_pred,
                                 y=y)


def classification_metric(y_pred, y):
    # TODO: david - add acc
    pass


def regression_metric(y_pred, y):
    # TODO: david - add mae
    pass


class Main:
    """
    A class to run the main and test the components of model
    """

    def __init__(self):
        pass

    @staticmethod
    def run():
        Main.prepare_io()
        Main.prepare_meta_datasets()
        Main.analyze_meta_datasets()

    @staticmethod
    def prepare_io():
        create_paths = [os.path.join(os.path.dirname(__file__), RESULTS_FOLDER_PATH),
                        os.path.join(os.path.dirname(__file__), DATA_FOLDER_PATH)]
        for path in create_paths:
            try:
                os.mkdir(path)
            except Exception as error:
                pass

    @staticmethod
    def prepare_meta_datasets():
        columns = ["LR", "ODT", "DE", "AutoSKlearn", "TPOT", "GPlearn", "SciMed", "Glearn"]

        if all([os.path.exists(os.path.join(RESULTS_FOLDER_PATH, file_name)) for file_name in ["meta_dataset_fit.csv",
                                                                                               "meta_dataset_train.csv",
                                                                                               "meta_dataset_test.csv"]]):
            # TODO: david add logic to load the lists we already have
            pass
        else:
            index = []
            fit_performance = []
            train_performance = []
            test_performance = []

        # run over all the datasets
        for data_path in glob(os.path.join(DATA_FOLDER_PATH, "*.csv")):

            # if we have this dataset, just skip it
            if os.path.basename(data_path.split(".csv")[0]) in index:
                continue

            # load dataset
            df = pd.read_csv(data_path)
            # make sure everything in the dataset is legit
            df.dropna(inplace=True)
            for col in list(df):
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.reset_index(inplace=True, drop=True)

            # prepare where to save results
            row_fit = []
            row_train = []
            row_test = []

            # split to x and y
            x = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            # split to train and test for later
            x_train, x_test, y_train, y_test = train_test_split(x,
                                                                y,
                                                                test_size=0.2,
                                                                random_state=RANDOM_STATE)
            # train models #
            # TODO: david - add logic for all the models here
            for model in [LinearRegression()]
                fit_model = model
                fit_model.fit(x,
                              y)
                y_pred = fit_model.predict(x)
                row_fit.append(metric(y_pred=y_pred,
                                      y=y))

                real_model = model
                real_model.fit(x_train,
                               y_train)
                y_pred_train = fit_model.predict(x_train)
                row_train.append(metric(y_pred=y_pred_train,
                                        y=y_train))
                y_pred_test = fit_model.predict(x_test)
                row_test.append(metric(y_pred=y_pred_test,
                                       y=y_test))


            # store results for this dataset
            fit_performance.append(row_fit)
            train_performance.append(row_train)
            test_performance.append(row_test)

            # recall the file name so we will use it later, making sure the order is not important
            index.append(os.path.basename(data_path.split(".csv")[0]))

            # save all three datasets of the answers - this is not optimal but helps if something clashes
            pd.DataFrame(data=fit_performance,
                         index=index,
                         columns=columns).to_csv(os.path.join(RESULTS_FOLDER_PATH, "meta_dataset_fit.csv"),
                                                 index=False)
            pd.DataFrame(data=train_performance,
                         index=index,
                         columns=columns).to_csv(os.path.join(RESULTS_FOLDER_PATH, "meta_dataset_train.csv"),
                                                 index=False)
            pd.DataFrame(data=test_performance,
                         index=index,
                         columns=columns).to_csv(os.path.join(RESULTS_FOLDER_PATH, "meta_dataset_test.csv"),
                                                 index=False)

    @staticmethod
    def analyze_meta_datasets():
        # TODO: add later
        pass


if __name__ == '__main__':
    Main.run()
