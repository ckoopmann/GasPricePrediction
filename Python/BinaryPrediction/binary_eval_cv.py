"""Model evaluation for the Binary Prediction problem

This script contains the code that was used to evaluate uni- and multivariate models in the binary prediction problem
This should be the fourth and last script to run among the scripts in this directory

"""
from numpy.random import seed
seed(1234)
from tensorflow import set_random_seed
set_random_seed(1234)
import signal
if hasattr(signal, 'SIGPIPE'):
    signal.signal(signal.SIGPIPE,signal.SIG_DFL)
from numpy import  concatenate, repeat
from pandas import read_csv,  DataFrame, concat
import re
import numpy as np
from pickle import dump
from functions import *
import sys
from keras.optimizers import RMSprop, SGD
from keras import backend as K

from sklearn.metrics import mean_absolute_error, mean_squared_error, log_loss, roc_auc_score
#Dictonary containing loss functions to be chosen by string `loss`
loss_functions_dict = {'mae':mean_absolute_error,  'mse': mean_squared_error, 'binary_crossentropy': log_loss, 'auc' : roc_auc_score}
#Select paths for model output and input data
output_path = "../../Data/Output/BinaryPrediction/binary_eval"
data_path = '../../Data/Input/InputData.csv'
#Path of model output from univariate parameter tuning step
parameter_selection_path_univar = "../../Data/Output/BinaryPrediction/binary_par_tuning/evaluation.csv"
#Path of model output from multivariate parameter tuning step
parameter_selection_path_multivar = "../../Data/Output/BinaryPrediction/binary_multivar_par_tuning/evaluation.csv"
#Select fixed hyper parameters
length_passed = 20
n_epochs = 300
batch= 20
output_activation= 'sigmoid'
loss='binary_crossentropy'
scaling = True
#Verbosity during model training
verbosity = 0
#Maximum days included for each month
max_days_left_passed=30
#All years matching this regex will be selected as test_months (Rolling prediction)
regex_testmonth= '17'
target_type  = 'TTF'

#Read in evaluation data from univariate tuning step
par_selection_df_univar = read_csv(parameter_selection_path_univar,header=0)
models = list(par_selection_df_univar.Model.unique())
#Select optimal parameter values for univariate models
architecture_dict_univar = {model_name: [int(s) for s in str(par_selection_df_univar.iloc[par_selection_df_univar.loc[par_selection_df_univar.Model == model_name, loss].idxmin()].Architecture).split('_')] for model_name in models}
learningrate_dict_univar = {model_name: par_selection_df_univar.iloc[par_selection_df_univar.loc[par_selection_df_univar.Model == model_name, loss].idxmin()].LearningRate for model_name in models}
dropout_dict_univar = {model_name: par_selection_df_univar.iloc[par_selection_df_univar.loc[par_selection_df_univar.Model == model_name, loss].idxmin()].Dropout for model_name in models}
#Read in evaluation data from multivariate tuning step
par_selection_df_multivar= read_csv(parameter_selection_path_multivar,header=0)
#Select optimal parameter values  and input variables for multivariate models
architecture_dict_multivar = {model_name: [int(s) for s in str(par_selection_df_multivar.iloc[par_selection_df_multivar.loc[par_selection_df_multivar.Model == model_name, loss].idxmin()].Architecture).split('_')] for model_name in models}
learningrate_dict_multivar = {model_name: par_selection_df_multivar.iloc[par_selection_df_multivar.loc[par_selection_df_multivar.Model == model_name, loss].idxmin()].LearningRate for model_name in models}
dropout_dict_multivar = {model_name: par_selection_df_multivar.iloc[par_selection_df_multivar.loc[par_selection_df_multivar.Model == model_name, loss].idxmin()].Dropout for model_name in models}
additional_input_vars_dict  = {model_name: par_selection_df_multivar.iloc[par_selection_df_multivar.loc[par_selection_df_multivar.Model == model_name, loss].idxmin()].Variables.split('_') for model_name in models}

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

for type in ['univar', 'multivar']:
    for model_name in models:

        months = [var for var in df.name.unique() if re.search(regex, var) is not None]
        test_months = [month for month in months if re.search(regex_testmonth, month) is not None]
        train_months = []
        train_months_sel = [month for month in months if re.search(regex_testmonth, month) is None]
        # Adjust length parameter according to model type
        if 'ffnn' in model_name:
            length = 1
        else:
            length = length_passed

        max_days_left = max_days_left_passed + length
        # Set hyper parameter depending on model type
        if type == 'univar':
            architecture = architecture_dict_univar[model_name]
            learningrate = learningrate_dict_univar[model_name]
            dropout = dropout_dict_univar[model_name]
            additional_input_vars = []
        elif type == 'multivar':
            architecture = architecture_dict_multivar[model_name]
            learningrate = learningrate_dict_multivar[model_name]
            dropout = dropout_dict_multivar[model_name]
            additional_input_vars = additional_input_vars_dict[model_name]

        target_data_list = []
        all_data_list = []
        X_sep = []
        y_sep = []
        ref_sep = []
        ref_series_list = []
        # Create X and y arrays for each month separately (to avoid problem of changing delivery period at the turn of the month)
        for target_var in months:
            input_vars = [target_var] + additional_input_vars
            try:
                X_curr, y_curr, all_data, reference_series = data_preparation_binary(df, input_vars, target_var, length,
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
                # print('No Data for: ' + target_var)
                # print('Original Error Message:' + str(e))
                pass
        test_months = sorted(test_months)
        # Loop through Test month for rolling prediction and testing
        for test_month in test_months:
            # Create lists to seperate test and train data
            test_selection = [months.index(test_month)]
            train_selection = [i for i in range(len(train_months)) if train_months[i] in train_months_sel or months[i] < test_month]

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

            # If selected above scale input variables using MinMaxScaler
            if scaling:
                scaler_X = MinMaxScaler()
                X_train_scaled = scaler_X.fit_transform(X_train.reshape((-1, X_train.shape[-1])))
                X_train_scaled = X_train_scaled.reshape(X_train.shape)
                X_train = X_train_scaled

                X_test_scaled = scaler_X.transform(X_test.reshape((-1, X_test.shape[-1])))
                X_test_scaled = X_test_scaled.reshape(X_test.shape)
                X_test = X_test_scaled

            decay_rate = learningrate/n_epochs
            # Create the model
            if model_name == 'lstm':
                model = create_model_LSTM(architecture, y_sep[0], X_sep[0], output_activation=output_activation,
                                          loss=loss, recurrent_dropout=dropout, optimizer=RMSprop(lr=learningrate),
                                          use_bias=False)
            elif model_name == 'rnn':
                model = create_model_simpleRNN(architecture, y_sep[0], X_sep[0], output_activation=output_activation,
                                               loss=loss, recurrent_dropout=dropout, optimizer=RMSprop(lr=learningrate))
            elif model_name == 'ffnn':
                model = create_model_FFNN(architecture, y_sep[0], X_sep[0], output_activation=output_activation,
                                          loss=loss,
                                          dropout=dropout, optimizer=SGD(lr=learningrate))
            elif model_name == 'ffnn_regression':
                model = create_model_FFNN(architecture, y_sep[0], X_sep[0], output_activation=output_activation,
                                          loss=loss,
                                          dropout=dropout, optimizer=SGD(lr=learningrate))
            #Train model
            history = model.fit(X_train, y_train, batch_size=batch, epochs=n_epochs,
                                validation_data=(X_test, y_test), verbose=verbosity)

            #Create predictions and reshape into one dimensional arrays
            y_hat_test = model.predict(X_test)
            y_hat_test  = y_hat_test.reshape(y_hat_test.shape[0])
            #Reshape actuals into one dimensional arrays
            y_test = y_test.reshape(-1)

            #Save predictions and actuals
            new_predictions = DataFrame(
                {'Model': model_name, 'Month_Traded': months_test, 'Batchsize': batch, 'Epochs': n_epochs, 'Length': length,  'LearningRate': learningrate, 'Dropout': dropout, 'Architecture': '_'.join(str(i) for i in architecture),
                 'Variables':'_'.join(str(i) for i in additional_input_vars), 'Prediction': y_hat_test, 'Actual': y_test, 'Reference': reference_test},
                index=reference_test.index)
            pred_df_list.append(new_predictions)

            #Calculate loss function for this combination and this month
            mean_loss = loss_function(y_test, y_hat_test)
            ref_loss = loss_function(y_test, reference_test)
            trainable_count = int(
                np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
            new_eval = DataFrame.from_records(
                [{'Model': model_name, 'TestMonth': test_month, 'Batchsize': batch, 'Epochs': n_epochs, 'Length': length, 'LearningRate': learningrate, 'Dropout': dropout, 'Architecture': '_'.join(str(i) for i in architecture),'Variables':'_'.join(str(i) for i in additional_input_vars), loss: mean_loss, loss+'ref': ref_loss, 'TrainObs': X_train.shape[0], 'TrainableParams': trainable_count}])
            eval_list.append(new_eval)
            # Save training history
            new_hist = DataFrame(
                {'Model': model_name,'TestMonth': test_month, 'Batchsize': batch, 'Epochs': n_epochs, 'Length': length, 'LearningRate': learningrate, 'Dropout': dropout,
                 'Architecture': '_'.join(str(i) for i in architecture), 'Variables':'_'.join(str(i) for i in additional_input_vars),
                 'TrainLoss': history.history['loss'], 'TestLoss': history.history['val_loss'],
                 'Iteration': [i for i in range(len(history.history['loss']))]})
            hist_df_list.append(new_hist)


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