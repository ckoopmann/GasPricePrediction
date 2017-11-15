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

loss_functions_dict = {'mae':mean_absolute_error,  'mse': mean_squared_error, 'binary_crossentropy': log_loss, 'auc' : roc_auc_score}
output_path = "../../Data/Output/LevelPrediction/level_par_tuning"
data_path = '../../Data/Input/InputData.csv'
length_passed = 20
n_epochs = 500
batch= 10

verbosity = 0
max_days_left_passed=30
regex_testmonth= '16'
regex_trainmonths= '16|17'
output_activation= 'linear'
loss='mse'
models = ['lstm', 'rnn', 'ffnn', 'ffnn_regression']
additional_input_vars_dict ={model_name: [] for model_name in models}
target_type  = 'TTF'

learningrates = [0.0001, 0.001, 0.01, 0.1]
dropouts = [0]
architectures = [[8],[16], [32]]

par_combs = [( l, d, a)  for l in learningrates for d in dropouts for a in architectures]

output = 1

regex = target_type + '\d'

df = read_csv(data_path,header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

eval_list = []
pred_df_list = []
hist_df_list = []
loss_function = loss_functions_dict[loss]

for model_name in models:

    months = [var for var in df.name.unique() if re.search(regex, var) is not None]
    test_months = [month for month in months if re.search(regex_testmonth, month) is not None]
    train_months_candidates = [month for month in months if re.search(regex_trainmonths, month) is None]
    train_months = []

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

                #Create predictions and reshape into one dimensional arrays
                y_hat_test = model.predict(X_test)
                y_hat_test  = y_hat_test.reshape(y_hat_test.shape[0])
                #Reshape actuals into one dimensional arrays
                y_test = y_test.reshape(-1)

                #Save predictions and actuals
                new_predictions = DataFrame(
                    {'Model': model_name, 'Month_Traded': months_test, 'Batchsize': batch, 'Epochs': n_epochs, 'Length': length,  'LearningRate': learningrate, 'Dropout': dropout, 'Architecture': '_'.join(str(int(i)) for i in architecture),
                     'Prediction': y_hat_test, 'Actual': y_test, 'Reference': reference_test},
                    index=reference_test.index)
                pred_df_list.append(new_predictions)

                #Calculate loss function for this combination and this month
                mean_loss = loss_function(y_test, y_hat_test)
                ref_loss = loss_function(y_test, reference_test)
                trainable_count = int(
                    np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
                new_eval = DataFrame.from_records(
                    [{'Model': model_name, 'Batchsize': batch, 'Epochs': n_epochs, 'Length': length, 'LearningRate': learningrate, 'Dropout': dropout, 'Architecture': '_'.join(str(int(i)) for i in architecture), loss: mean_loss, loss+'ref': ref_loss, 'TrainObs': X_train.shape[0], 'TrainableParams': trainable_count}])
                eval_list.append(new_eval)

                new_hist = DataFrame(
                    {'Model': model_name, 'Batchsize': batch, 'Epochs': n_epochs, 'Length': length, 'LearningRate': learningrate, 'Dropout': dropout,
                     'Architecture': '_'.join(str(int(i)) for i in architecture),
                     'TrainLoss': history.history['loss'], 'TestLoss': history.history['val_loss'],
                     'Iteration': [i for i in range(len(history.history['loss']))]})
                hist_df_list.append(new_hist)
            except Exception as e:
                print('No training possible for parameter combination: ' + '_'.join([str(learningrate), str(dropout), '_'.join(str(int(i)) for i in architecture)]))
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