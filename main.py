# library imports
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error,mean_squared_error,r2_score

from sklearn.linear_model import LinearRegression  # linear model for regression and classification
try:
    #from dl85 import DL85Predictor  # Optimal decision tree
    #from model.gosdt import GOSDT
    #import autosklearn.regression  # AutoML baseline - https://automl.github.io/auto-sklearn/master/#manual [regression]

    from tpot import TPOTClassifier, TPOTRegressor  # bio-inspired autoML
    from gplearn.genetic import SymbolicRegressor  # bio-inspired autoML #2
    #from autogluon.tabular import TabularDataset, TabularPredictor  # bio-inspired autoML #3
    
    # TODO: add later # bio-inspired autoML #4
except Exception as error:
    print("Error in loading libraries - try again! Error = {}".format(error))

# project imports
from consts import *
from model_eval_func import model_eval_func, model_eval_func_titles


# define the model's score metric for each case so it will be easy to replace later #
def metric(y_pred, y):
    print(y)
    print(y_pred)
    metricList = []
    metricList.append(mean_absolute_error(y_true=y,y_pred=y_pred))
    metricList.append(mean_squared_error(y_true=y,y_pred=y_pred))
    metricList.append(r2_score(y_true=y,y_pred=y_pred))

    return metricList



class Main:
    """
    A class to run the main and test the components of model
    """

    def __init__(self):
        pass

    @staticmethod
    def run(prepare_meta_dataset: bool, analyze_meta_dataset: bool):
        Main.prepare_io()
        if prepare_meta_dataset:
            Main.prepare_meta_datasets()
        if analyze_meta_dataset:
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
        columns = ["LR"]#, "ODT", "DE", "AutoSKlearn", "TPOT", "GPlearn", "SciMed", "Glearn"]

        # if all([os.path.exists(os.path.join(RESULTS_FOLDER_PATH, file_name)) for file_name in [META_DATASET_FIT,
        #                                                                                        META_DATASET_TRAIN,
        #                                                                                        META_DATASET_TEST]]):
        #     fit_df = pd.read_csv(os.path.exists(os.path.join(RESULTS_FOLDER_PATH, META_DATASET_FIT)),
        #                          index_col="index")
        #     index = list(fit_df.index)
        #     fit_performance = fit_df.values.tolist()
        #     train_performance = pd.read_csv(os.path.exists(os.path.join(RESULTS_FOLDER_PATH, META_DATASET_TRAIN)),
        #                                     index_col="index").values.tolist()
        #     test_performance = pd.read_csv(os.path.exists(os.path.join(RESULTS_FOLDER_PATH, META_DATASET_TEST)),
        #                                    index_col="index").values.tolist()
        # else:
        index = []
        fit_performance = []
        train_performance = []
        test_performance = []

        # load config file #
        # it is assumed to be file_name,task_type - c/r,is_time_series - 0/1,window - 1 to infinity,y_target_col
        config_df = {row["file_name"]: {"is_time_series": int(row["is_time_series"]),
                                        "window": int(row["window"]),
                                        "y_target_col": row["y_target_col"]}
                     for row_index, row in pd.read_csv(CONFIG_FILE_PATH).iterrows()}

        # run over all the datasets
        for data_path in config_df: #glob(os.path.join(DATA_FOLDER_PATH, "*.csv")):

            # if we have this dataset, just skip it
            root_path = "data/"
            file_ext = ".csv"
            file_path = root_path + data_path + file_ext
            file_name = os.path.basename(data_path.split(".csv")[0])
            print(file_name)
            if file_name in index:
                continue

            # load dataset
            df = pd.read_csv(file_path, thousands=",")
            # check if we need to make from time_series
            if config_df[file_name]["is_time_series"]:
                answer = []
                for row, row_index in df.iterrows():
                    ts_row = Main._to_time_series(list(row_index), 4)
                    answer.append(ts_row)
                df = pd.concat(answer)
                #df = Main._to_time_series(data=df, n_in=config_df[file_name]["window"])

            # make sure everything in the dataset is legit
            df.dropna(inplace=True)
            for col in list(df):
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.reset_index(inplace=True, drop=True)

            # prepare where to save results
            row_fit = []
            row_train = []
            row_test = []

            # put the target column in the start
            #y = df[config_df[file_name]["y_target_col"]]
            df.insert(0,
                      config_df[file_name]["y_target_col"],
                     df.pop(config_df[file_name]["y_target_col"]))
            #x = df.drop(config_df[file_name]["y_target_col"])
            # split to x and y
            x = df.iloc[:, 0:]
            #copy_x = x.copy()
            y = df.iloc[:, 0]
            #copy_y = y.copy()

            # split to train and test for later
            x_train, x_test, y_train, y_test = train_test_split(x,
                                                                y,
                                                                test_size=0.2,
                                                                random_state=RANDOM_STATE)
            # train models #
            # TODO: david - add logic for all the models here
            for model in [LinearRegression()]:
                #run model on all data
                fit_model = model
                fit_model.fit(x, y)
                y_pred = fit_model.predict(x)
                row_fit.append(metric(y_pred=y_pred, y=y))
                #run model on train
                real_model = model
                real_model.fit(x_train, y_train)
                y_pred_train = fit_model.predict(x_train)
                row_train.append(metric(y_pred=y_pred_train, y=y_train))
                #run model on test
                y_pred_test = fit_model.predict(x_test)
                row_test.append(metric(y_pred=y_pred_test, y=y_test))

            # store results for this dataset
            fit_performance.append(row_fit)
            train_performance.append(row_train)
            test_performance.append(row_test)

            # recall the file name so we will use it later, making sure the order is not important
            index.append(file_name)

            # save all three datasets of the answers - this is not optimal but helps if something clashes
            pd.DataFrame(data=fit_performance,
                         index=index,
                         columns=columns).to_csv(os.path.join(RESULTS_FOLDER_PATH, META_DATASET_FIT))
            pd.DataFrame(data=train_performance,
                         index=index,
                         columns=columns).to_csv(os.path.join(RESULTS_FOLDER_PATH, META_DATASET_TRAIN))
            pd.DataFrame(data=test_performance,
                         index=index,
                         columns=columns).to_csv(os.path.join(RESULTS_FOLDER_PATH, META_DATASET_TEST))

    @staticmethod
    def analyze_meta_datasets():
        # TODO: add later
        pass

    @staticmethod
    def _to_time_series(data: pd.DataFrame,
                        n_in: int,
                        n_out: int = 1,
                        dropnan: bool = True):
        """
        Frame a time series as a supervised learning dataset.
        Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
        Pandas DataFrame of series framed for supervised learning.
        """
        n_vars = 1 if type(data) is list else data.shape[1]
        if not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data)
        else:
            df = data
        cols = []
        names = []
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols,
                        axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg


if __name__ == '__main__':
    Main.run(prepare_meta_dataset=True,
             analyze_meta_dataset=False)
