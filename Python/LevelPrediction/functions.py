from numpy import array, stack, amin, amax, zeros, hstack, concatenate
from pandas import read_csv, datetime, concat, DataFrame, Series
from keras.models import  Sequential
from keras.layers import LSTM, Dense, TimeDistributed, SimpleRNN, Dropout
from sklearn.preprocessing import MinMaxScaler
import argparse
import os

#Commentin according to PEP257 docstring
# date-time parsing function for loading the dataset
def parser(x):
    return datetime.strptime(x,'%Y-%m-%d')

def data_preparation_level(df, input_vars, target_var, length,  max_days_left, return_y_sequence = False):
    """Data Preparation Function for Price Level Prediction

    This function converts the data as downloaded from Thomson Reuters Eikon and converts it into X, y np.arrays
    which can be used to train keras models. The input variables are shifted by one timestep into the past such that
    the prediction is made one day into the future.

    Args:
        df (pandas.DataFrame): The Thomson Reuters Data containing the columns 'Date', 'CLOSE' and 'name'
        input_vars (list(str)): List of parameter names appearing in df.name to select input variables
        target_var (str): Name of the target variable as it appears in df.name
        length (int): Selecting the length of each sequence for the training of recurrent models in Keras
        max_days_left (int): Maximum number of trading days to choose for each month
        return_y_sequence (bool):


    Returns:
        X (np.ndarray): Array of input variables shifted by one
        y (np.ndarray): Array of target variable values
        all_data (pd.DataFrame): Input and target variables in one dataframe
        reference_series (pd.Series): Reference prediction (Lagged Value)

    """
    input_data = [df.loc[df['name'] == var].CLOSE for var in input_vars]
    input_data = concat(input_data, axis=1)
    input_data.columns = input_vars
    new_input_vars = [var + "_Lag" for var in input_vars]

    input_data = input_data.fillna(method='ffill')
    input_data_shifted = input_data.shift(1)
    input_data_shifted = input_data_shifted.dropna()
    input_data_shifted.columns = new_input_vars

    target_series = df.loc[df['name'] == target_var].CLOSE
    target_data = DataFrame({target_var: target_series}, index= target_series.index,)
    target_data['days_left'] = [sum(target_data.index >= day) for day in target_data.index]
    # target_data_lag =  target_data.shift(1)
    # target_data_lag.columns = [target_var_dir + '_Lag']
    #
    # target_data_complete = target_data.merge(target_data_lag, how = 'inner', left_index = True, right_index = True)
    target_data_complete = target_data
    all_data = input_data_shifted.merge(target_data_complete, how='inner', left_index=True, right_index=True)
    all_data = all_data.loc[all_data.days_left <= max_days_left]
    if return_y_sequence:
        y = array(all_data[target_var])
        reference_series = all_data[target_var + '_Lag']
    else:
        y = array(all_data[target_var])[length - 1:]
        reference_series = all_data[target_var + '_Lag'].iloc[length - 1:]

    # shifted = array(all_data[target_var_dir + '_Lag'])[length-1:]
    input_data_filtered = all_data[new_input_vars].values
    # input_data_filtered['month'] = input_data_filtered.index.month
    # print(target_data_complete.head(10))
    # print(input_data_shifted.head(10))
    # print(all_data.head(10))
    # print(input_data_filtered.shape)

    X_list = list()
    for i in range(length, input_data_filtered.shape[0] + 1):
        new_row = input_data_filtered[i - length:i, :]
        X_list.append(new_row)

    X = stack(X_list)
    return X, y, all_data, reference_series

def create_model_LSTM(neurons_list, y, X, output_activation, loss, hidden_activation = 'tanh', recurrent_activation = 'hard_sigmoid', optimizer='rmsprop', metrics=[], dropout = 0, recurrent_dropout=0, **kwargs):
    output = 1 if len(y.shape) == 1 else y.shape[1]
    model = Sequential()
    if len(neurons_list) == 1:
        model.add(LSTM(neurons_list[0], input_shape=(X.shape[1], X.shape[2]), activation = hidden_activation, recurrent_activation=recurrent_activation, return_sequences=False, dropout = dropout, recurrent_dropout=recurrent_dropout, use_bias=True))
    else:
        model.add(LSTM(neurons_list[0], input_shape=(X.shape[1], X.shape[2]), activation = hidden_activation, recurrent_activation=recurrent_activation, return_sequences=True, dropout = dropout, recurrent_dropout=recurrent_dropout, use_bias = True))
        for i in range(1, len(neurons_list)):
            model.add(LSTM(neurons_list[i], activation = hidden_activation, recurrent_activation=recurrent_activation, recurrent_dropout=recurrent_dropout, use_bias=True))
    model.add(Dense(output, activation=output_activation))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    print(model.summary())
    return(model)

def create_model_simpleRNN(neurons_list, y, X, output_activation, loss, hidden_activation = 'tanh',  optimizer='rmsprop', metrics=[], dropout = 0, recurrent_dropout = 0):
    output = 1 if len(y.shape) == 1 else y.shape[1]
    model = Sequential()
    if len(neurons_list) == 1:
        model.add(SimpleRNN(neurons_list[0], input_shape=(X.shape[1], X.shape[2]), activation = hidden_activation,return_sequences=False, dropout = dropout, recurrent_dropout=recurrent_dropout))
    else:
        model.add(SimpleRNN(neurons_list[0], input_shape=(X.shape[1], X.shape[2]), activation = hidden_activation, return_sequences=True, dropout = dropout, recurrent_dropout=recurrent_dropout))
        for i in range(1, len(neurons_list)):
            model.add(SimpleRNN(neurons_list[i], activation = hidden_activation, recurrent_dropout=recurrent_dropout))
    model.add(Dense(output, activation=output_activation))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    print(model.summary())
    return(model)

def create_model_FFNN(neurons_list, y, X,  loss, optimizer='rmsprop', metrics=[], hidden_activation= 'tanh', output_activation  = 'linear', dropout = 0):
    assert len(y.shape) == 2, "y should be two dimensional but has shape: " + str(y.shape)
    assert len(X.shape) == 2, "X should be two dimensional but has shape: " + str(X.shape)
    output = y.shape[1]
    model = Sequential()
    #Case of Linear regression
    if neurons_list == [0]:
        if dropout == 0:
            model.add(
                Dense(output, input_dim=X.shape[1], kernel_initializer='normal', activation=output_activation))
        else:
            model.add(Dropout(input_shape=(X.shape[1],), rate=dropout))
            model.add(Dense(output, kernel_initializer='normal', activation=output_activation))
    #proper mlp / FFNN
    else:
        if dropout == 0:
            model.add(Dense(neurons_list[0], input_dim=X.shape[1],  kernel_initializer='normal', activation=hidden_activation))
        else:
            model.add(Dropout(input_shape=(X.shape[1],), rate = dropout))
            model.add(Dense(neurons_list[0],  kernel_initializer='normal', activation=hidden_activation))
        for i in range(1,len(neurons_list)):
            model.add(Dense(neurons_list[i], kernel_initializer='normal', activation=hidden_activation))
        model.add(Dense(output, kernel_initializer='normal', activation=output_activation))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    print(model.summary())
    return(model)

def convert_to_min_sep(series, target_var):
    ismin_series = Series(index=series.index,  name = target_var + '_ismin')
    min_series = Series(index=series.index,  name = target_var + '_min')
    reference_series = Series(index=series.index, name = target_var + '_ref')
    days_left_series = Series(index=series.index, name='days_left')
    for curr_day in series.index:
        curr_val = series[curr_day]
        curr_min = min(series.loc[(series.index >= curr_day)])
        curr_ismin = curr_val == curr_min
        ismin_series[curr_day] = curr_ismin
        min_series[curr_day] = curr_min
        days_left = (series.index >= curr_day).sum()
        reference_series[curr_day] = 1 / days_left
        days_left_series[curr_day] = days_left
    return ismin_series, reference_series, days_left_series, min_series

def data_preparation_binary(df, input_vars, target_var, length,  max_days_left, return_y_sequence = False):
    target_var_min = target_var + '_ismin'
    input_data = [df.loc[df['name'] == var].CLOSE.rename(var) for var in input_vars]
    input_data = concat(input_data, axis=1)
    input_data.columns = input_vars
    new_input_vars = [var + "_Lag" for var in input_vars]

    input_data = input_data.fillna(method='ffill')
    input_data_shifted = input_data.shift(1)
    input_data_shifted = input_data_shifted.dropna()
    input_data_shifted.columns = new_input_vars

    target_series, reference_series, days_left_series, min_series = convert_to_min_sep(
        df.loc[df['name'] == target_var].CLOSE, target_var)
    target_data = DataFrame({target_series.name: target_series, reference_series.name: reference_series,
                             days_left_series.name: days_left_series, min_series.name: min_series})
    # target_data_lag =  target_data.shift(1)
    # target_data_lag.columns = [target_var_dir + '_Lag']
    #
    # target_data_complete = target_data.merge(target_data_lag, how = 'inner', left_index = True, right_index = True)
    target_data_complete = target_data
    all_data = input_data_shifted.merge(target_data_complete, how='inner', left_index=True, right_index=True)
    all_data = all_data.loc[all_data.days_left <= max_days_left]
    if return_y_sequence:
        y = array(all_data[target_var_min])
        reference_series = all_data[target_var+'_ref']
    else:
        y = array(all_data[target_var_min])[length - 1:]
        reference_series = all_data[target_var + '_ref'].iloc[length - 1:]

    # shifted = array(all_data[target_var_dir + '_Lag'])[length-1:]
    input_data_filtered = all_data[new_input_vars + ['days_left']].values
    # input_data_filtered['month'] = input_data_filtered.index.month



    X_list = list()
    for i in range(length, input_data_filtered.shape[0] + 1):
        new_row = input_data_filtered[i - length:i, :]
        X_list.append(new_row)

    X = stack(X_list)
    return X, y, all_data, reference_series