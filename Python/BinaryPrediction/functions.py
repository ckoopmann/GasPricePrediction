"""Utility functions for all steps

This script contains a variety of functions to be used in parameter tuning, variable selection and evaluation of models
for both prediction problems

"""
from numpy import array, stack, amin, amax, zeros, hstack, concatenate
from pandas import read_csv, datetime, concat, DataFrame, Series
from keras.models import  Sequential
from keras.layers import LSTM, Dense, TimeDistributed, SimpleRNN, Dropout
from sklearn.preprocessing import MinMaxScaler
import os

def parser(x):
    """Datetime Parser

    This function is used to parse the dates from strings when reading in the input data .csv file.
    This function is passed to the 'pd.read_csv' function as the 'date_parser' function

    Args:
        x (str): String containing a date in the format %Y-%m-%d'


    Returns:
        date (datetime): x converted to date

        """
    date = datetime.strptime(x,'%Y-%m-%d')
    return date

def data_preparation_level(df, input_vars, target_var, length,  max_days_left):
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


    Returns:
        X (np.ndarray): Array of input variables shifted by one
        y (np.ndarray): Array of target variable values
        all_data (pd.DataFrame): Input and target variables in one dataframe
        reference_series (pd.Series): Reference prediction (Lagged Value)

    """
    #Convert different variables from long format (different rows) to wide format (different columns)
    input_data = [df.loc[df['name'] == var].CLOSE for var in input_vars]
    input_data = concat(input_data, axis=1)
    input_data.columns = input_vars
    new_input_vars = [var + "_Lag" for var in input_vars]

    #Shift input variables one step into the past
    input_data = input_data.fillna(method='ffill')
    input_data_shifted = input_data.shift(1)
    input_data_shifted = input_data_shifted.dropna()
    input_data_shifted.columns = new_input_vars

    #Extract target variable
    target_series = df.loc[df['name'] == target_var].CLOSE
    target_data = DataFrame({target_var: target_series}, index= target_series.index,)
    target_data['days_left'] = [sum(target_data.index >= day) for day in target_data.index]
    target_data_complete = target_data

    #Merge target variable with input variables using inner join
    all_data = input_data_shifted.merge(target_data_complete, how='inner', left_index=True, right_index=True)
    all_data = all_data.loc[all_data.days_left <= max_days_left]

    #Extract target variable as array y
    y = array(all_data[target_var])[length - 1:]
    #Extract reference prediction (lagged value of target variable)
    reference_series = all_data[target_var + '_Lag'].iloc[length - 1:]

    #Extract input variables
    input_data_filtered = all_data[new_input_vars].values

    #Create X array with dimensions [n.obs, length, n.variables]
    X_list = list()
    #Loop through the data and get the last 'lenght' observations of each variable for each time point as one array
    for i in range(length, input_data_filtered.shape[0] + 1):
        new_row = input_data_filtered[i - length:i, :]
        #Append each array to list
        X_list.append(new_row)
    #Stack list to get array of the required shape
    X = stack(X_list)
    return X, y, all_data, reference_series

def create_model_LSTM(neurons_list, y, X, output_activation, loss, hidden_activation = 'tanh', recurrent_activation = 'hard_sigmoid', optimizer='rmsprop', metrics=[], dropout = 0, recurrent_dropout=0, **kwargs):
    """Create Keras LSTM Model

    This function creates and compiles a sequential Keras model containing at least one LSTM layer and one fully connected
    output layer

    Args:
        neurons_list (list(int)): List of number of neurons in each hidden layer. Length of list controls depth of model
        y (np.array): Example data of output data to specify output shape
        X (np.array): Example data of input data to specify input shape
        output_activation(str): Name of output layer activation function as accepted by Keras model classes
        loss(str): Name of loss function
        hidden_activation(str): Name of hidden layer activation function as accepted by Keras model classes
        recurrent_activation (int): Name of activation function of LSTM gates as accepted by Keras model classes
        optimizer(str / keras.Optimizer): Choice of optimizer as string or optimizer object
        metrics(list(str)): Names of additional metrics to record in training history
        dropout(int): Dropout probability
        recurrent_dropout(int): Dropout probability for recurrent connections


    Returns:
        model (keras.Sequential): Compiled model ready to be trained
    """
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
    """Create Keras simple RNN Model

    This function creates and compiles a sequential Keras model containing at least one simple RNN layer and one fully connected
    output layer

    Args:
        neurons_list (list(int)): List of number of neurons in each hidden layer. Length of list controls depth of model
        y (np.array): Example data of output data to specify output shape
        X (np.array): Example data of input data to specify input shape
        output_activation(str): Name of output layer activation function as accepted by Keras model classes
        loss(str): Name of loss function
        hidden_activation(str): Name of hidden layer activation function as accepted by Keras model classes
        optimizer(str / keras.Optimizer): Choice of optimizer as string or optimizer object
        metrics(list(str)): Names of additional metrics to record in training history
        dropout(int): Dropout probability
        recurrent_dropout(int): Dropout probability for recurrent connections


    Returns:
        model (keras.Sequential): Compiled model ready to be trained
    """
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
    """Create Keras FFNN Model

    This function creates and compiles a sequential Keras feed forward neural network with or without hidden layers

    Args:
        neurons_list (list(int)): List of number of neurons in each hidden layer.[0] creates model without hidden layer
        y (np.array): Example data of output data to specify output shape
        X (np.array): Example data of input data to specify input shape
        output_activation(str): Name of output layer activation function as accepted by Keras model classes
        loss(str): Name of loss function
        hidden_activation(str): Name of hidden layer activation function as accepted by Keras model classes
        optimizer(str / keras.Optimizer): Choice of optimizer as string or optimizer object
        metrics(list(str)): Names of additional metrics to record in training history
        dropout(int): Dropout probability


    Returns:
        model (keras.Sequential): Compiled model ready to be trained
    """
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
    """ Convert a numerical target variable into the binary variable in the second prediction problem

        This function takes in a pandas Series of a numerical variable with date index and returns a binary series
        indicating wether each value is minimal among the remaining days of the month as well as the reference prediciton
        series based on an equal distribution, a series indicating the number of remaining trading days as well as the series
        containing the minimum among all remaining trading days at each point.


        Args:
            series (pd.Series): Series of numerical variable with date index
            target_var (str): String containing the name of the original target variable


        Returns:
            ismin_series (pd.Series): Binary Series indicating wether each value is minimal among the remaining trading days
            reference_series (pd.Series): Series containing the reference prediction: 1/days_left
            days_left_series (pd.Series): Number of trading days left in the same month at each time point
            min_series (pd.Series): Minimum price among remaining trading days in that month for each trading day
        """
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

def data_preparation_binary(df, input_vars, target_var, length,  max_days_left):
    """Data Preparation Function for Binary Prediction

    This function converts the data as downloaded from Thomson Reuters Eikon and converts it into X, y np.arrays
    which can be used to train keras models. The input variables are shifted by one timestep into the past such that
    the prediction is made one day into the future. Unlike in the price level case y contains the binary variable indicating
    wether the current price is minimal for that month and the reference series is based on the equal distribution assumption.

    Args:
        df (pandas.DataFrame): The Thomson Reuters Data containing the columns 'Date', 'CLOSE' and 'name'
        input_vars (list(str)): List of parameter names appearing in df.name to select input variables
        target_var (str): Name of the target variable as it appears in df.name
        length (int): Selecting the length of each sequence for the training of recurrent models in Keras
        max_days_left (int): Maximum number of trading days to choose for each month


    Returns:
        X (np.ndarray): Array of input variables shifted by one
        y (np.ndarray): Array of binary target variable values
        all_data (pd.DataFrame): Input and binary target variables in one dataframe
        reference_series (pd.Series): Reference prediction (Equal Distribution)

    """
    #Convert different variables from long format (different rows) to wide format (different columns)
    target_var_min = target_var + '_ismin'
    input_data = [df.loc[df['name'] == var].CLOSE.rename(var) for var in input_vars]
    input_data = concat(input_data, axis=1)
    input_data.columns = input_vars
    new_input_vars = [var + "_Lag" for var in input_vars]

    #Shift input variables one step into the past
    input_data = input_data.fillna(method='ffill')
    input_data_shifted = input_data.shift(1)
    input_data_shifted = input_data_shifted.dropna()
    input_data_shifted.columns = new_input_vars

    #Apply convert_to_min_sep function to get binary target variable and reference predictions
    target_series, reference_series, days_left_series, min_series = convert_to_min_sep(
        df.loc[df['name'] == target_var].CLOSE, target_var)
    target_data = DataFrame({target_series.name: target_series, reference_series.name: reference_series,
                             days_left_series.name: days_left_series, min_series.name: min_series})

    #Merge target and input variables
    target_data_complete = target_data
    all_data = input_data_shifted.merge(target_data_complete, how='inner', left_index=True, right_index=True)
    all_data = all_data.loc[all_data.days_left <= max_days_left]

    #Limit data to observations where at least `length` number of past values are available for prediction
    y = array(all_data[target_var_min])[length - 1:]
    reference_series = all_data[target_var + '_ref'].iloc[length - 1:]
    input_data_filtered = all_data[new_input_vars + ['days_left']].values

    X_list = list()
    #Loop through the data and get the last 'lenght' observations of each variable for each time point as one array
    for i in range(length, input_data_filtered.shape[0] + 1):
        new_row = input_data_filtered[i - length:i, :]
        X_list.append(new_row)
    # Stack list to get array of the required shape
    X = stack(X_list)
    y = y.astype(int)
    return X, y, all_data, reference_series