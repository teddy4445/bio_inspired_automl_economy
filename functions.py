import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error



def r2_score(y_true,y_pred):
    r2 = (1 - (((y_true - y_pred) * 2).sum()) / (((y_true - y_true.mean()) * 2).sum()))
    return r2


def model_setup(x,y):
    model = LinearRegression()
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=73)
    fit_model = model
    #fit_model.fit(x, y)
    #print("fit: ")
    #model_fit_predict(model,x,y,y,x)
    #print("train: ")
    #model_fit_predict(model,x_train,y_train,y_train,x_train)
    print("test ")
    model_fit_predict(model,x_train,y_train,y_test,x_test)


def model_fit_predict(model,x,y,yForPredict,xForPredict):
    fit_model = model.fit(x,y)
    yPred = fit_model.predict(xForPredict)
    print(f"MSE: {mean_squared_error(yForPredict,yPred)} MAE: {mean_absolute_error(yForPredict,yPred)} R2: {r2_score(yForPredict,yPred)}")


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


df = pd.read_csv("data/phl-cityandmunicipallevelpovertyestimates-csv-1noNa.csv",thousands=",")
#df=pd.read_csv("data/2012_Industry_Data_by_Industry_and_StatenoNa.csv",thousands = ',')
#df = pd.read_csv("data/BSE_30_ADANIPORTSnoNa.csv")

answer = []
#for row, row_index in df.iterrows():
#    ts_row = _to_time_series(list(row_index), 3)
#    answer.append(ts_row)
#ts_df = pd.concat(answer)
#ts_df.to_csv("temp.csv", index=False)
#print(df)
#for col in list(df):
#    df[col] = df[col].astype('int')
y = df["Pov_2012"]
x = df.drop(columns=["Pov_2012"])
model_setup(x,y)
#print(x)
#print(y)
