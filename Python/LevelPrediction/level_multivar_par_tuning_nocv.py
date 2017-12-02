"""Multivariate Parameter Tuning for the Price Level Prediction problem

This script contains the code that was used to tune the hyper parameters of multivariate models in the price level prediction problem
This should be the third script to run among the scripts in this directory

"""
from numpy.random import seed
seed(1234)
from tensorflow import set_random_seed
set_random_seed(1234)
import signal
if hasattr(signal, 'SIGPIPE'):
    signal.signal(signal.SIGPIPE,signal.SIG_DFL)
from numpy import  concatenate, repeat
import numpy as np
from pandas import read_csv,  DataFrame, concat
import re
from pickle import dump
from functions import *
import sys
from keras.optimizers import RMSprop, SGD
from keras import backend as K

from sklearn.metrics import mean_absolute_error, mean_squared_error, log_loss, roc_auc_score
#Dictonary containing loss functions to be chosen by string `loss`
loss_functions_dict = {'mae':mean_absolute_error,  'mse': mean_squared_error, 'binary_crossentropy': log_loss, 'auc' : roc_auc_score}
#Select paths for model output and input data
output_path = "../../Data/Output/LevelPrediction/level_multivar_par_tuning"
data_path = '../../Data/Input/InputData.csv'
#Path of model output from previous step (variable selection) to select input variables
variable_selection_path = "../../Data/Output/LevelPrediction/level_var_selection/evaluation.csv"
#Select fixed hyper parameters
length_passed = 20
n_epochs = 300
batch= 20
output_activation= 'tanh'
loss='mse'
scaling = True
#Verbosity during model training
verbosity = 0
#Maximum days included for each month
max_days_left_passed=30
#Maximum days included for each month
regex_testmonth= '16'
#All years NOT matching this regex will be selected as training_months
regex_trainmonths= '16|17'
#Read in results from variable selection
var_selection_df = read_csv(variable_selection_path,header=0)
models = list(var_selection_df.Model.unique())
#Select the variable combination with lowest error for each model based on the results from variable tuning step
additional_input_vars_dict  = {model_name: var_selection_df.iloc[var_selection_df.loc[var_selection_df.Model == model_name, loss].idxmin()].Vars.split('_') for model_name in models}
target_type  = 'TTF'
#Select tuning grid
learningrates = [0.0001,0.001, 0.01, 0.1]
dropouts = [0, 0.25, 0.5]
architectures = [[8],[16], [32]]
#Create list of all parameter combinations
par_combs = [( l, d, a)  for l in learningrates for d in dropouts for a in architectures]

output = 1
#Complete regex to match all monthly futures of the target type
regex = target_type + '\d'
#Read in input data
df = read_csv(data_path,header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
#Empty lists to store results
eval_list = []
pred_df_list = []
hist_df_list = []
loss_function = loss_functions_dict[loss]

for model_name in models:

    months = [var for var in df.name.unique() if re.search(regex, var) is not None]
    test_months = [month for month in months if re.search(regex_testmonth, month) is not None]
    train_months_candidates = [month for month in months if re.search(regex_trainmonths, month) is None]
    train_months = []
    # Adjust length parameter according to model type
    if 'ffnn' in model_name:
        length = 1
    else:
        length = length_passed

    max_days_left = max_days_left_passed + length


    target_data_list = []
    all_data_list = []
    X_sep = []
    y_sep = []
    ref_sep = []
    ref_series_list = []
    # Create X and y arrays for each month separately (to avoid problem of changing delivery period at the turn of the month)
    for target_var in months:
        input_vars = [target_var] + additional_input_vars_dict[model_name]
        try:
            X_curr, y_curr, all_data, reference_series = data_preparation_level(df, input_vars, target_var, length,
                                                                              max_days_left)
            if 'ffnn' in model_name:
                X_curr = X_curr.reshape(X_curr.shape[0], -1)
                y_curr = y_curr.reshape(y_curr.shape[0], -1)
            X_sep.append(X_curr)
            y_sep.append(y_curr)
            ref_sep.append(reference_series)
            all_data_list.append(all_data)
            train_months.append(target_var)
        except Exception as e:
            print('No Data for: ' + target_var)
            print(e)
            pass

    #Create lists to seperate test and train data
    test_selection = [i for i in range(len(train_months)) if train_months[i] in test_months]
    train_selection = [i for i in range(len(train_months)) if train_months[i] in train_months_candidates]

    #Divide data in train and test
    X_train_list = [X_sep[i] for i in train_selection]
    y_train_list = [y_sep[i] for i in train_selection]
    ref_train_list = [ref_sep[i] for i in train_selection]
    X_train = concatenate(X_train_list)
    y_train = concatenate(y_train_list)
    reference_train = concat(ref_train_list)


    X_test_list = [X_sep[i] for i in test_selection]
    y_test_list = [y_sep[i] for i in test_selection]
    ref_test_list = [ref_sep[i] for i in test_selection]
    X_test = concatenate(X_test_list)
    y_test = concatenate(y_test_list)
    reference_test = concat(ref_test_list)
    months_test_list = [repeat(train_months[i], len(ref_sep[i])) for i in test_selection]
    months_test = concatenate(months_test_list)

    # If selected above scale input and target variables using MinMaxScaler
    if scaling:
        scaler_y = MinMaxScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape((-1, 1)))
        y_train = y_train_scaled.reshape(-1,1)

        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
        y_test_unscaled = y_test
        y_test = y_test_scaled.reshape(-1,1)


        scaler_X = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train.reshape((-1, X_train.shape[-1])))
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_train = X_train_scaled

        X_test_scaled = scaler_X.transform(X_test.reshape((-1, X_test.shape[-1])))
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        X_test = X_test_scaled
    # Begin parameter tuning by looping through combinations
    for (learningrate, dropout, architecture) in par_combs:
        try:
            # Create the model
            if model_name == 'lstm':
                model = create_model_LSTM(architecture, y_sep[0], X_sep[0], output_activation=output_activation, loss=loss,
                                          recurrent_dropout=dropout, optimizer=RMSprop(lr=learningrate), use_bias=False)
            elif model_name == 'rnn':
                model = create_model_simpleRNN(architecture, y_sep[0], X_sep[0], output_activation=output_activation,
                                               loss=loss, recurrent_dropout=dropout, optimizer=RMSprop(lr=learningrate))
            elif model_name == 'ffnn':
                model = create_model_FFNN(architecture, y_sep[0], X_sep[0], output_activation=output_activation,
                                          loss=loss,
                                          dropout=dropout, optimizer=SGD(lr=learningrate))
            elif model_name == 'ffnn_regression':
                if architecture == architectures[0]:
                    architecture = [0]
                    model = create_model_FFNN(architecture, y_sep[0], X_sep[0], output_activation=output_activation,
                                              loss=loss,
                                              dropout=dropout, optimizer=SGD(lr=learningrate))
                else:
                    continue

            #Train model
            history = model.fit(X_train, y_train, batch_size=batch, epochs=n_epochs,
                                validation_data=(X_test, y_test), verbose=verbosity)
            y_hat_test = model.predict(X_test_scaled)
            if scaling:
                y_hat_test = scaler_y.inverse_transform(y_hat_test.reshape(-1, 1))
                y_test = y_test_unscaled
            y_hat_test = y_hat_test.reshape(-1)
            y_test = y_test.reshape(-1)

            #Save predictions and actuals
            new_predictions = DataFrame(
                {'Model': model_name, 'Month_Traded': months_test, 'Batchsize': batch, 'Epochs': n_epochs, 'Length': length,  'LearningRate': learningrate, 'Dropout': dropout, 'Architecture': '_'.join(str(i) for i in architecture), 'Variables': '_'.join(str(i) for i in additional_input_vars_dict[model_name]),
                 'Prediction': y_hat_test, 'Actual': y_test, 'Reference': reference_test},
                index=reference_test.index)
            pred_df_list.append(new_predictions)

            #Calculate loss function for this combination and this month
            mean_loss = loss_function(y_test, y_hat_test)
            ref_loss = loss_function(y_test, reference_test)
            trainable_count = int(
                np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
            new_eval = DataFrame.from_records(
                [{'Model': model_name, 'Batchsize': batch, 'Epochs': n_epochs, 'Length': length, 'LearningRate': learningrate, 'Dropout': dropout, 'Architecture': '_'.join(str(i) for i in architecture), 'Variables': '_'.join(str(i) for i in additional_input_vars_dict[model_name]), loss: mean_loss, loss+'ref': ref_loss, 'TrainObs': X_train.shape[0], 'TrainableParams': trainable_count}])
            eval_list.append(new_eval)
            # Save training history
            new_hist = DataFrame(
                {'Model': model_name, 'Batchsize': batch, 'Epochs': n_epochs, 'Length': length, 'LearningRate': learningrate, 'Dropout': dropout,
                 'Architecture': '_'.join(str(i) for i in architecture), 'Variables': '_'.join(str(i) for i in additional_input_vars_dict[model_name]),
                 'TrainLoss': history.history['loss'], 'TestLoss': history.history['val_loss'],
                 'Iteration': [i for i in range(len(history.history['loss']))]})
            hist_df_list.append(new_hist)
        except Exception as e:
            print('No training possible for parameter combination: ' + '_'.join(
                [str(learningrate), str(dropout), '_'.join(str(int(i)) for i in architecture)]))
            print(e)
            continue


#Collapse list of prediction data and evaluation data in single dataframes
predictions_df = concat(pred_df_list)
eval_df = concat(eval_list)
hist_df = concat(hist_df_list)

if not os.path.exists(output_path):
    os.makedirs(output_path)
#Save predictions and evaluation
predictions_df.to_csv(output_path + "/predictions.csv", index=True)
eval_df.to_csv(output_path + "/evaluation.csv", index=False)
hist_df.to_csv(output_path + "/history.csv", index=False)