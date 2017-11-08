from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import signal
if hasattr(signal, 'SIGPIPE'):
    signal.signal(signal.SIGPIPE,signal.SIG_DFL)
from numpy import  concatenate, repeat
from pandas import read_csv,  DataFrame, concat
import re
from pickle import dump
from functions import *
import sys
from keras.optimizers import RMSprop, SGD
from sklearn.metrics import mean_absolute_error, mean_squared_error, log_loss, roc_auc_score

loss_functions_dict = {'mae':mean_absolute_error,  'mse': mean_squared_error, 'binary_crossentropy': log_loss, 'auc' : roc_auc_score}

output_path = "../../Data/Output/LevelPrediction/level_var_selection"
data_path = '../../Data/Input/InputData.csv'
parameter_selection_path = "../../Data/Output/LevelPrediction/level_par_tuning/evaluation.csv"

length_passed = 20
n_epochs= 300
batch= 20
verbosity = 0

max_days_left_passed=30
regex_testmonth= '16'
filename='lstm_min_var_selection.py'
output_activation= 'linear'
loss='mse'
all_input_vars=["ConLDZNL", "ConLDZEU", "ConNLDZNL", "ConNLDZEU",  "LNGStockEU", "ProdNL", "ProdUKCS", "StorageNL", "StorageUK", "StorageEU", "TradeBBL", "TradeIUK", "TradeNONWE", "TradeNOUK", "TradeRUNWE", "TTFDA", "NBPFM", "OilFM", "ElectricityBaseFM", "ElectricityPeakFM", "EURUSDFX", "EURGBPFX"]
target_type = 'TTF'
max_iteration = 5
#sys.stdout.flush()


par_selection_df = read_csv(parameter_selection_path,header=0)
models = list(par_selection_df.Model.unique())
#Enter results of parameter tuning hereW
architecture_dict = {model_name: [int(s) for s in str(par_selection_df.iloc[par_selection_df.loc[par_selection_df.Model == model_name, loss].idxmin()].Architecture).split('_')] for model_name in models}
learningrate_dict = {model_name: par_selection_df.iloc[par_selection_df.loc[par_selection_df.Model == model_name, loss].idxmin()].LearningRate for model_name in models}
dropout_dict = {model_name: par_selection_df.iloc[par_selection_df.loc[par_selection_df.Model == model_name, loss].idxmin()].Dropout for model_name in models}


output = 1

regex = target_type + '\d'

df = read_csv(data_path,header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

eval_list = []
pred_df_list = []
hist_df_list = []
loss_function = loss_functions_dict[loss]

for model_name in models:
    architecture = architecture_dict[model_name]
    learningrate = learningrate_dict[model_name]
    dropout = dropout_dict[model_name]

    remaining_inputs = [i for i in all_input_vars]
    selected_input_vars = []

    if 'ffnn' in model_name:
        length = 1
    else:
        length = length_passed

    max_days_left = max_days_left_passed + length

    for iteration in range(1,max_iteration+1):
        eval_list_iter = []
        for curr_input_var in remaining_inputs:

            months = [var for var in df.name.unique() if re.search(regex, var) is not None]
            test_months = [month for month in months if re.search(regex_testmonth, month) is not None]
            train_months = []


            additional_input_vars = selected_input_vars + [curr_input_var]

            #Initialise
            target_data_list = []
            all_data_list = []
            X_sep = []
            y_sep = []
            ref_sep = []
            ref_series_list = []

            for target_var in months:
                input_vars = [target_var] + additional_input_vars
                try:
                    X_curr, y_curr, all_data, reference_series = data_preparation_min(df, input_vars, target_var, length, max_days_left)
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

            # Display error message if error was caught for every month and skip to next variable
            if len(X_sep) == 0:
                print("No Data available for combination: " + str(additional_input_vars))
                continue

            #Create the model
            if model_name == 'lstm':
                model = create_model_LSTM(architecture, y_sep[0], X_sep[0], output_activation=output_activation, loss = loss, recurrent_dropout=dropout, optimizer= RMSprop(lr=learningrate), use_bias = False)
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
            #Leave one out loop


            #Create lists to seperate test and train data
            test_selection = [i for i in range(len(train_months)) if train_months[i] in test_months]
            train_selection = [i for i in range(len(train_months)) if train_months[i] not in test_months]

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
                {'Model': model_name, 'MonthTraded': months_test, 'Iteration': iteration, 'NewVar': curr_input_var,'Vars': '_'.join(additional_input_vars),'Prediction': y_hat_test, 'Actual': y_test, 'Reference': reference_test},
                index=reference_test.index)
            pred_df_list.append(new_predictions)

            new_hist = DataFrame(
                {'Model': model_name, 'SelectionIteration': iteration, 'NewVar': curr_input_var, 'Vars': '_'.join(additional_input_vars),
                 'TrainLoss': history.history['loss'], 'TestLoss': history.history['val_loss'],
                 'TrainingIteration': [i for i in range(len(history.history['loss']))]})
            hist_df_list.append(new_hist)

            #Calculate loss function for this combination and this month
            mean_loss = loss_function(list(y_test), y_hat_test)
            ref_loss = loss_function(list(y_test), reference_test)

            new_eval = DataFrame.from_records(
                [{'Model': model_name, 'Iteration': iteration, 'NewVar': curr_input_var,'Vars': '_'.join(additional_input_vars), loss: mean_loss, loss+'_ref':ref_loss}])
            eval_list_iter.append(new_eval)
        #Get variable combination with minimum loss in this iteration
        eval_iter = concat(eval_list_iter).reset_index()
        min_index = eval_iter[loss].idxmin()
        min_var = eval_iter.loc[min_index].NewVar
        #Add selected variable to selected vars and remove from remaining vars
        selected_input_vars.append(min_var)
        remaining_inputs.remove(min_var)
        eval_iter['selected'] = min_var
        eval_list.append(eval_iter)


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