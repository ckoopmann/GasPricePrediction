rm(list = ls())
library(data.table)
library(tidyr)
library(ggplot2)
library(ggsci)
library(MLmetrics)
library(plyr)
library(xtable)

#Select directory paths for Output data from each step of the binary prediction process
chr.par_tuning_directory = "../Data/Output/BinaryPrediction/binary_par_tuning"
chr.var_selection_directory = "../Data/Output/BinaryPrediction/binary_var_selection"
chr.multivar_par_tuning_directory = "../Data/Output/BinaryPrediction/binary_multivar_par_tuning"
chr.evaluation_directory = "../Data/Output/BinaryPrediction/binary_eval"

#Analysis of Par-Tuning
df.par_tuning = as.data.table(read.csv(file = paste(chr.par_tuning_directory, "evaluation.csv", sep = "/")))
df.par_tuning[, Model:= toupper(Model)]
#Minor changes in spelling of some model types and column names
df.par_tuning[, Model := gsub('FFNN_REGRESSION','Regression',Model)]
df.par_tuning = rename(df.par_tuning, c(Architecture = "HiddenNeurons", binary_crossentropy = 'BCE'))
#Get Minimum values of target function by model
df.par_tuning_mins = df.par_tuning[,.SD[which.min(BCE)],by = Model]
#Get full data on each minimal parameter combination. (Assumes that minimum values are unique)
df.par_tuning_mins = df.par_tuning_mins[,.(Model, HiddenNeurons, Dropout, LearningRate, BCE)]

#Save data on minimal parameter combinations as latex tabel
Filename = '../Latex/tables/binary_par_tuning_short.tex'
caption = 'Selected parameter combinations in univariate tuning step of binary prediction'
label = 'tab:binary.par.tuning.short'
latex = xtable(df.par_tuning_mins, label = label, caption = caption, digits = 4)
latex = print(latex, include.rownames = F, table.placement = 'h!')
latex = gsub('\\begin{tabular}', ' \\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}', latex, fixed = T)
latex = gsub('\\end{tabular}', ' \\end{tabular}\n\\end{adjustbox}', latex, fixed = T)
writeLines(latex, con = Filename)

#Save data on all parameter combinations as latex tables
Filename = '../Latex/tables/binary_par_tuning_full.tex'
caption = 'Full Results in univariate tuning step of binary prediction'
label = 'tab:binary.par.tuning.full'
latex = xtable(df.par_tuning[,.(Model, HiddenNeurons, Dropout, LearningRate, BCE)], label = label, caption = caption, digits = 4)
latex = print(latex, include.rownames = F, table.placement = 'h!')
latex = gsub('\\begin{tabular}', ' \\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}', latex, fixed = T)
latex = gsub('\\end{tabular}', ' \\end{tabular}\n\\end{adjustbox}', latex, fixed = T)
writeLines(latex, con = Filename)


#Analysis of Var-Selection
df.var_selection = as.data.table(read.csv(file = paste(chr.var_selection_directory, "evaluation.csv", sep = "/")))
df.var_selection[, Model:= toupper(Model)]
#Minor changes in spelling of some model types and column names
df.var_selection[, Model := gsub('FFNN_REGRESSION','Regression',Model)]
df.var_selection[, Variables := paste0('TTFFM ',gsub('_', ' ', Vars))]
df.var_selection = rename(df.var_selection, c( binary_crossentropy = 'BCE'))
#Get data on minimal variable combinations
df.var_selection_mins = df.var_selection[,.SD[which.min(BCE)],by = Model]
df.var_selection_mins = df.var_selection_mins[,.(Model, Variables, BCE)]

#Save minimal variable combinations as latex table
Filename = '../Latex/tables/binary_var_selection_short.tex'
caption = 'Selected variable combinations in variable selection step of binary prediction'
label = 'tab:binary.var.selection.short'
latex = xtable(df.var_selection_mins, label = label, caption = caption, digits = 4)
latex = print(latex, include.rownames = F, table.placement = 'h!')
latex = gsub('\\begin{tabular}', ' \\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}', latex, fixed = T)
latex = gsub('\\end{tabular}', ' \\end{tabular}\n\\end{adjustbox}', latex, fixed = T)
writeLines(latex, con = Filename)

#Save data on all variable combinations as latex table
Filename = '../Latex/tables/binary_var_selection_full.tex'
caption = 'Full Results in variable selection step of binary prediction'
label = 'tab:binary.var.selection.full'
latex = xtable(df.var_selection[,.(Model, Variables, BCE)], label = label, caption = caption, digits = 4)
latex = print(latex, include.rownames = F, table.placement = 'h!')
latex = gsub('\\begin{tabular}', ' \\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}', latex, fixed = T)
latex = gsub('\\end{tabular}', ' \\end{tabular}\n\\end{adjustbox}', latex, fixed = T)
writeLines(latex, con = Filename)


#Analysis of Par-Tuning (Same as for univariate case above)
df.multivar_par_tuning = as.data.table(read.csv(file = paste(chr.multivar_par_tuning_directory, "evaluation.csv", sep = "/")))
df.multivar_par_tuning[, Model:= toupper(Model)]
#Minor changes in spelling of some model types and column names
df.multivar_par_tuning[, Model := gsub('FFNN_REGRESSION','Regression',Model)]
df.multivar_par_tuning = rename(df.multivar_par_tuning, c(Architecture = "HiddenNeurons", binary_crossentropy = 'BCE'))
df.multivar_par_tuning_mins = df.multivar_par_tuning[,.SD[which.min(BCE)],by = Model]
df.multivar_par_tuning_mins = df.multivar_par_tuning_mins[,.(Model, HiddenNeurons, Dropout, LearningRate, BCE)]

#Save data on minimal parameter combinations as latex tabel
Filename = '../Latex/tables/binary_multivar_par_tuning_short.tex'
caption = 'Selected parameter combinations in multivariate tuning step of binary prediction'
label = 'tab:binary.multivar.par.tuning.short'
latex = xtable(df.multivar_par_tuning_mins, label = label, caption = caption, digits = 4)
latex = print(latex, include.rownames = F, table.placement = 'h!')
latex = gsub('\\begin{tabular}', ' \\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}', latex, fixed = T)
latex = gsub('\\end{tabular}', ' \\end{tabular}\n\\end{adjustbox}', latex, fixed = T)
writeLines(latex, con = Filename)


#Save data on all parameter combinations as latex tabel
Filename = '../Latex/tables/binary_multivar_par_tuning_full.tex'
caption = 'Full Results in multivariate tuning step of binary prediction'
label = 'tab:binary.multivar.par.tuning.full'
latex = xtable(df.multivar_par_tuning[,.(Model, HiddenNeurons, Dropout, LearningRate, BCE)], label = label, caption = caption, digits = 4)
latex = print(latex, include.rownames = F, table.placement = 'h!')
latex = gsub('\\begin{tabular}', ' \\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}', latex, fixed = T)
latex = gsub('\\end{tabular}', ' \\end{tabular}\n\\end{adjustbox}', latex, fixed = T)
writeLines(latex, con = Filename)


#Analysis of Model Evaluation
df.evaluation = as.data.table(read.csv(file = paste(chr.evaluation_directory, "evaluation.csv", sep = "/")))
df.evaluation[, Model:= toupper(Model)]
#Minor changes in spelling of some model types and column names
df.evaluation[, Model := gsub('FFNN_REGRESSION','Regression',Model)]
df.evaluation[, Variables := paste0('TTFFM ',gsub('_', ' ', Variables))]
df.evaluation = rename(df.evaluation, c(Architecture = "HiddenNeurons", binary_crossentropy = 'BCE'))
#Convert the reference value for the equal distribution model from a column to an additional row in the evalaution data.table
df.evaluation = rbind(df.evaluation[,.(Model, Variables,TestMonth = gsub('TTF','',TestMonth), BCE)],  df.evaluation[Model == 'LSTM' & Variables == "TTFFM ",.(Model = 'EqualDistribution', Variables = "",TestMonth = gsub('TTF','',TestMonth), BCE = binary_crossentropyref)])
#Calculate mean BCE across Test Months and sort accordingly
df.evaluation_mean = df.evaluation[,.(BCE = mean(BCE)), by = .(Model, Variables)]
df.evaluation_mean = df.evaluation_mean[order(BCE),]

#Save data on mean BCE across Test Months
Filename = '../Latex/tables/binary_eval_short.tex'
caption = 'Average BCE across months for each model in evaluation step'
label = 'tab:binary.eval.short'
latex = xtable(df.evaluation_mean, label = label, caption = caption, digits = 4)
latex = print(latex, include.rownames = F, table.placement = 'h!')
latex = gsub('\\begin{tabular}', ' \\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}', latex, fixed = T)
latex = gsub('\\end{tabular}', ' \\end{tabular}\n\\end{adjustbox}', latex, fixed = T)
writeLines(latex, con = Filename)

#Save data on month wise BCE
Filename = '../Latex/tables/binary_eval_full.tex'
caption = 'Full results by testing month for each model in evaluation step'
label = 'tab:binary.eval.short'
latex = xtable(df.evaluation, label = label, caption = caption, digits = 4)
latex = print(latex, include.rownames = F, table.placement = 'h!')
latex = gsub('\\begin{tabular}', ' \\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}', latex, fixed = T)
latex = gsub('\\end{tabular}', ' \\end{tabular}\n\\end{adjustbox}', latex, fixed = T)
writeLines(latex, con = Filename)


#Graph performance over test month
df.evaluation[, Multivar := Variables != "TTFFM "]
df.evaluation[Multivar == T, Type := "Multivariate"]
df.evaluation[Multivar != T, Type := "Univariate"]

plt.evaluation_months = ggplot(data = df.evaluation, aes(x = as.numeric(TestMonth), y = BCE)) + geom_line(aes(col = Model, linetype = Type)) + scale_x_continuous(breaks=c(117, 217,317,417,517,617,717), labels=c("January 17", "February 17", "March 17", "April 17", "May 17", "June 17", "July 17")) + theme_classic() + scale_color_uchicago("dark") + labs(title = "Test BCE across Test Months for all", x = "Test Month")

ggsave( "../Plots/binary_evaluation_monthly_all.png",plt.evaluation_months, width = 8.11, height = 4.98)

#Multivariate models plus EqualDistribution 
plt.evaluation_months_multivar = ggplot(data = df.evaluation[Type == "Multivariate" | Model == "EqualDistribution"], aes(x = as.numeric(TestMonth), y = BCE)) + geom_line(aes(col = Model)) + scale_x_continuous(breaks=c(117, 217,317,417,517,617,717), labels=c("January 17", "February 17", "March 17", "April 17", "May 17", "June 17", "July 17")) + theme_classic() + scale_color_uchicago("dark") + labs(title = "Test BCE across test months for multivariate models", x = "Test Month")

ggsave( "../Plots/binary_evaluation_monthly_multivar.png",plt.evaluation_months_multivar, width = 8.11, height = 4.98)

#Univariate Models
plt.evaluation_months_univar = ggplot(data = df.evaluation[Type == "Univariate" | Model == "EqualDistribution"], aes(x = as.numeric(TestMonth), y = BCE)) + geom_line(aes(col = Model)) + scale_x_continuous(breaks=c(117, 217,317,417,517,617,717), labels=c("January 17", "February 17", "March 17", "April 17", "May 17", "June 17", "July 17")) + theme_classic() + scale_color_uchicago("dark") + labs(title = "Test BCE across test months for univariate models", x = "Test Month")

ggsave( "../Plots/binary_evaluation_monthly_univar.png",plt.evaluation_months_univar, width = 8.11, height = 4.98)


##Trading Strategy Analysis
df.predictions = as.data.table(read.csv(file = paste(chr.evaluation_directory, "predictions.csv", sep = "/")))
#Set last prediction always to one to ensure sum of trades to be 1
df.predictions[, Variables := paste0('TTFFM ',gsub('_', ' ', Variables))]
df.predictions[, Model:= toupper(Model)]
#Minor changes in spelling of some model types and column names
df.predictions[, Model := gsub('FFNN_REGRESSION','Regression',Model)]
df.predictions_long = df.predictions
df.predictions_long[Reference == 1, Prediction := 1]
df.predictions_long = rbind(df.predictions_long[,.(Date, Month_Traded, Model, Variables, Prediction)], df.predictions_long[Model == 'LSTM' & Variables == 'TTFFM ',.(Date, Month_Traded, Model = 'EqualDistribution', Variables = '', Prediction = Reference)])
#Get TTFFM daily closing price from predictio data frame of price level prediction data
df.prices = as.data.table(read.csv(file = "../Data/Output/LevelPrediction/level_eval/predictions.csv"))[Model == 'lstm' & Variables == '',.(Date, Month_Traded, Price = Actual)]

df.predictions_prices = merge(df.predictions_long, df.prices,by = c("Date", "Month_Traded"), all.x = T, all.y = F)


df.predictions_prices[,DaysLeft := .N:1, by = .( Model, Month_Traded, Variables)]

#Evaluation of Trading Strategies resulting from different models
#Length of trading period. Change this to evaluate models for different periods
num.daysleft_cutoff = 14
#Keep only Days in trading period
df.predictions_prices_reduced = df.predictions_prices[DaysLeft <= num.daysleft_cutoff,]
#Create 2 columns: Remaining Share left to be traded and Proportion Traded on each day
df.predictions_prices_reduced[,RemainingShare := cumprod(1-Prediction), by = .(Month_Traded,Model, Variables)]
df.predictions_prices_reduced[,RemainingShare := shift(RemainingShare), by = .(Month_Traded,Model, Variables)]
df.predictions_prices_reduced[is.na(RemainingShare),RemainingShare := 1]
df.predictions_prices_reduced[,ProportionTraded := RemainingShare*Prediction]
#Check that proportions traded add up to one for each model and month
df.predictions_prices_reduced[,sum(ProportionTraded), by = .(Month_Traded,Model, Variables)]
#Get average price paid with each trading strategy both month wise and averaged across months
df.trade_evaluation = df.predictions_prices_reduced[,.(AveragePrice = sum(ProportionTraded*Price)), by = .(Month_Traded,Model, Variables)]
df.trade_evaluation_mean = df.trade_evaluation[,.(AveragePrice = mean(AveragePrice)), by = .(Model, Variables)]
df.trade_evaluation_mean = df.trade_evaluation_mean[order(AveragePrice),]
#Save results as latex table
Filename = '../Latex/tables/binary_trade_short.tex'
caption = 'Average price per MWh across months for simple trading strategy based on each model'
label = 'tab:binary.trade.short'
latex = xtable(df.trade_evaluation_mean, label = label, caption = caption, digits = 4)
latex = print(latex, include.rownames = F, table.placement = 'h!')
latex = gsub('\\begin{tabular}', ' \\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}', latex, fixed = T)
latex = gsub('\\end{tabular}', ' \\end{tabular}\n\\end{adjustbox}', latex, fixed = T)
writeLines(latex, con = Filename)

#Trading evaluation graphed over time
#Graph performance over test month
#Create Column indicating wether model is multivariate or univariate
df.trade_evaluation[, Multivar := Variables != "TTFFM "]
df.trade_evaluation[Multivar == T, Type := "Multivariate"]
df.trade_evaluation[Multivar != T, Type := "Univariate"]

#Plot for all models (Uni and Multivariate)
plt.trade_evaluation_months = ggplot(data = df.trade_evaluation, aes(x = as.numeric(Month_Traded), y = AveragePrice)) + geom_line(aes(col = Model, linetype = Type)) + scale_x_continuous(breaks=c(117, 217,317,417,517,617,717), labels=c("January 17", "February 17", "March 17", "April 17", "May 17", "June 17", "July 17")) + theme_classic() + scale_color_uchicago("dark") + labs(title = "Test AveragePrice across Test Months for all", x = "Test Month")

ggsave( "../Plots/binary_trade_evaluation_monthly_all.png",plt.trade_evaluation_months, width = 8.11, height = 4.98)

#Multivariate models plus EqualDistribution 
plt.trade_evaluation_months_multivar = ggplot(data = df.trade_evaluation[Type == "Multivariate" | Model == "EqualDistribution"], aes(x = as.numeric(Month_Traded), y = AveragePrice)) + geom_line(aes(col = Model)) + scale_x_continuous(breaks=c(117, 217,317,417,517,617,717), labels=c("January 17", "February 17", "March 17", "April 17", "May 17", "June 17", "July 17")) + theme_classic() + scale_color_uchicago("dark") + labs(title = "Average price per MWh for trading strategies based on multivariate models", x = "Month Traded", y = "Price[EUR/MW]")

ggsave( "../Plots/binary_trade_evaluation_monthly_multivar.png",plt.trade_evaluation_months_multivar, width = 8.11, height = 4.98)

#Univariate models plus EqualDistribution 
plt.trade_evaluation_months_univar = ggplot(data = df.trade_evaluation[Type == "Univariate" | Model == "EqualDistribution"], aes(x = as.numeric(Month_Traded), y = AveragePrice)) + geom_line(aes(col = Model)) + scale_x_continuous(breaks=c(117, 217,317,417,517,617,717), labels=c("January 17", "February 17", "March 17", "April 17", "May 17", "June 17", "July 17")) + theme_classic() + scale_color_uchicago("dark") + labs(title = "Average price per MWh for trading strategies based on univariate models", x = "Month Traded", y = "Price[EUR/MW]")

ggsave( "../Plots/binary_trade_evaluation_monthly_univar.png",plt.trade_evaluation_months_univar, width = 8.11, height = 4.98)

#Plot predictions for example month
df.prediction_plot_data = df.predictions_long[Month_Traded == "TTF0717" & Model %in% c("LSTM", "EqualDistribution") & Variables %in% c("TTFFM ", "")]

plt.predictions = ggplot(data = df.prediction_plot_data, aes(x = as.Date(Date), y = Prediction)) + geom_line(aes(col = Model)) + theme_classic() + scale_color_uchicago("dark") + labs(title = "Predictions LSTM vs. EqualDistribution", subtitle = "Month Traded July 2017", x = "Trading Day", y = "Binary Prediction")


#Analyse relative savings of LSTM model compared to equal distributions for different lengths of trading periods
df.eval_cutoff = data.table()
for(num.daysleft_cutoff in 1:15){
    #Almost identical to above code (lines 180 - 190) comparing strategies of different models
    df.predictions_prices_reduced_iteration = df.predictions_prices[DaysLeft <= num.daysleft_cutoff,]
    
    df.predictions_prices_reduced_iteration[,RemainingShare := cumprod(1-Prediction), by = .(Month_Traded,Model, Variables)]
    df.predictions_prices_reduced_iteration[,RemainingShare := shift(RemainingShare), by = .(Month_Traded,Model, Variables)]
    df.predictions_prices_reduced_iteration[is.na(RemainingShare),RemainingShare := 1]
    df.predictions_prices_reduced_iteration[,ProportionTraded := RemainingShare*Prediction]
    
    df.predictions_prices_reduced_iteration[,sum(ProportionTraded), by = .(Month_Traded,Model, Variables)]
    
    df.trade_evaluation = df.predictions_prices_reduced_iteration[,.(AveragePrice = sum(ProportionTraded*Price)), by = .(Month_Traded,Model, Variables)]
    df.trade_evaluation_mean = df.trade_evaluation[,.(AveragePrice = mean(AveragePrice)), by = .(Model, Variables)]
    #Extract average prices of univariate lstm and equal distribution
    num.price_lstm_univar = df.trade_evaluation_mean[Model == 'LSTM' & Variables == "TTFFM ",AveragePrice]
    num.price_equal = df.trade_evaluation_mean[Model == 'EqualDistribution',AveragePrice]
    #Calculate savings
    num.savings = num.price_equal - num.price_lstm_univar
    #Save ressult in data.table
    df.eval_cutoff = rbind(df.eval_cutoff, data.table(TradingPeriod = num.daysleft_cutoff, LSTMSavings = num.savings))
}
#Export results as latex table
Filename = '../Latex/tables/binary_eval_cutoff.tex'
caption = 'Savings in EUR/MWh of univariate LSTM model relative to equal distribution benchmark for differenttrading periods'
label = 'tab:binary.eval.cutoff'
latex = xtable(df.eval_cutoff, label = label, caption = caption, digits = 4)
latex = print(latex, include.rownames = F, table.placement = 'h!')
latex = gsub('\\begin{tabular}', ' \\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}', latex, fixed = T)
latex = gsub('\\end{tabular}', ' \\end{tabular}\n\\end{adjustbox}', latex, fixed = T)
writeLines(latex, con = Filename)
