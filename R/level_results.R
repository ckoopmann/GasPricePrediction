rm(list = ls())
library(data.table)
library(tidyr)
library(ggplot2)
library(ggsci)
library(MLmetrics)
library(plyr)
library(xtable)
library(scales)

#Select directory paths for Output data from each step of the price level prediction process
chr.par_tuning_directory = "../Data/Output/LevelPrediction/level_par_tuning"
chr.var_selection_directory = "../Data/Output/LevelPrediction/level_var_selection"
chr.multivar_par_tuning_directory = "../Data/Output/LevelPrediction/level_multivar_par_tuning"
chr.evaluation_directory = "../Data/Output/LevelPrediction/level_eval"

#Analysis of Par-Tuning
df.par_tuning = as.data.table(read.csv(file = paste(chr.par_tuning_directory, "evaluation.csv", sep = "/")))
df.par_tuning[, Model:= toupper(Model)]
#Minor changes in spelling of some model types and column names
df.par_tuning[, Model := gsub('FFNN_REGRESSION','Regression',Model)]
df.par_tuning = rename(df.par_tuning, c(Architecture = "HiddenNeurons", mse = 'MSE'))
#Get Minimum values of target function by model
df.par_tuning_mins = df.par_tuning[,.SD[which.min(MSE)],by = Model]
#Get full data on each minimal parameter combination. (Assumes that minimum values are unique)
df.par_tuning_mins = df.par_tuning_mins[,.(Model, HiddenNeurons, Dropout, LearningRate, MSE)]

#Save data on minimal parameter combinations as latex table
Filename = '../Latex/tables/level_par_tuning_short.tex'
caption = 'Selected parameter combinations in univariate tuning step of price level prediction'
label = 'tab:level.par.tuning.short'
latex = xtable(df.par_tuning_mins, label = label, caption = caption, digits = 4)
latex = print(latex, include.rownames = F, table.placement = 'h!')
latex = gsub('\\begin{tabular}', ' \\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}', latex, fixed = T)
latex = gsub('\\end{tabular}', ' \\end{tabular}\n\\end{adjustbox}', latex, fixed = T)
writeLines(latex, con = Filename)

#Save data on all parameter combinations as latex tables
Filename = '../Latex/tables/level_par_tuning_full.tex'
caption = 'Full Results in univariate tuning step of price level prediction'
label = 'tab:level.par.tuning.full'
latex = xtable(df.par_tuning[,.(Model, HiddenNeurons, Dropout, LearningRate, MSE)], label = label, caption = caption, digits = 4)
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
#Get minimum variable combinations
df.var_selection = rename(df.var_selection, c( mse = 'MSE'))
df.var_selection_mins = df.var_selection[,.SD[which.min(MSE)],by = Model]
df.var_selection_mins = df.var_selection_mins[,.(Model, Variables, MSE)]

#Save minimum variable combinations as latex
Filename = '../Latex/tables/level_var_selection_short.tex'
caption = 'Selected variable combinations in variable selection step of price level prediction'
label = 'tab:level.var.selection.short'
latex = xtable(df.var_selection_mins, label = label, caption = caption, digits = 4)
latex = print(latex, include.rownames = F, table.placement = 'h!')
latex = gsub('\\begin{tabular}', ' \\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}', latex, fixed = T)
latex = gsub('\\end{tabular}', ' \\end{tabular}\n\\end{adjustbox}', latex, fixed = T)
writeLines(latex, con = Filename)


#Save data on all variable combiantions as latex
Filename = '../Latex/tables/level_var_selection_full.tex'
caption = 'Full Results in variable selection step of price level prediction'
label = 'tab:level.var.selection.full'
latex = xtable(df.var_selection[,.(Model, Variables, MSE)], label = label, caption = caption, digits = 4)
latex = print(latex, include.rownames = F, table.placement = 'h!')
latex = gsub('\\begin{tabular}', ' \\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}', latex, fixed = T)
latex = gsub('\\end{tabular}', ' \\end{tabular}\n\\end{adjustbox}', latex, fixed = T)
writeLines(latex, con = Filename)


#Analysis of multivariate Par-Tuning (Same as univariate case above)
df.multivar_par_tuning = as.data.table(read.csv(file = paste(chr.multivar_par_tuning_directory, "evaluation.csv", sep = "/")))
df.multivar_par_tuning[, Model:= toupper(Model)]
#Minor changes in spelling of some model types and column names
df.multivar_par_tuning[, Model := gsub('FFNN_REGRESSION','Regression',Model)]
df.multivar_par_tuning = rename(df.multivar_par_tuning, c(Architecture = "HiddenNeurons", mse = 'MSE'))
#Get minimal parameter combinations
df.multivar_par_tuning_mins = df.multivar_par_tuning[,.SD[which.min(MSE)],by = Model]
df.multivar_par_tuning_mins = df.multivar_par_tuning_mins[,.(Model, HiddenNeurons, Dropout, LearningRate, MSE)]

#Save minimal parameter combinations as latex
Filename = '../Latex/tables/level_multivar_par_tuning_short.tex'
caption = 'Selected parameter combinations in multivariate tuning step of price level prediction'
label = 'tab:level.multivar.par.tuning.short'
latex = xtable(df.multivar_par_tuning_mins, label = label, caption = caption, digits = 4)
latex = print(latex, include.rownames = F, table.placement = 'h!')
latex = gsub('\\begin{tabular}', ' \\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}', latex, fixed = T)
latex = gsub('\\end{tabular}', ' \\end{tabular}\n\\end{adjustbox}', latex, fixed = T)
writeLines(latex, con = Filename)

#Save data on all parameter combinations as latex
Filename = '../Latex/tables/level_multivar_par_tuning_full.tex'
caption = 'Full Results in multivariate tuning step of price level prediction'
label = 'tab:level.multivar.par.tuning.full'
latex = xtable(df.multivar_par_tuning[,.(Model, HiddenNeurons, Dropout, LearningRate, MSE)], label = label, caption = caption, digits = 4)
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
df.evaluation = rename(df.evaluation, c(Architecture = "HiddenNeurons", mse = 'MSE'))
#Convert the reference value for the lagged value model from a column to an additional row in the evalaution data.table
df.evaluation = rbind(df.evaluation[,.(Model, Variables,TestMonth = gsub('TTF','',TestMonth), MSE)],  df.evaluation[Model == 'LSTM' & Variables == "TTFFM ",.(Model = 'LaggedValue', Variables = "TTFFM ",TestMonth = gsub('TTF','',TestMonth), MSE = mseref)])

#Calculate mean MSE across test months for each model
df.evaluation_mean = df.evaluation[,.(MSE = mean(MSE)), by = .(Model, Variables)]
df.evaluation_mean = df.evaluation_mean[order(MSE),]

#Save mean MSEs across months as latex table
Filename = '../Latex/tables/level_eval_short.tex'
caption = 'Average MSE across months for each model in evaluation step'
label = 'tab:level.eval.short'
latex = xtable(df.evaluation_mean, label = label, caption = caption, digits = 4)
latex = print(latex, include.rownames = F, table.placement = 'h!')
latex = gsub('\\begin{tabular}', ' \\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}', latex, fixed = T)
latex = gsub('\\end{tabular}', ' \\end{tabular}\n\\end{adjustbox}', latex, fixed = T)
writeLines(latex, con = Filename)

#Save month wise MSE as latex table
Filename = '../Latex/tables/level_eval_full.tex'
caption = 'Full results by testing month for each model in evaluation step'
label = 'tab:level.eval.short'
latex = xtable(df.evaluation, label = label, caption = caption, digits = 4)
latex = print(latex, include.rownames = F, table.placement = 'h!')
latex = gsub('\\begin{tabular}', ' \\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}', latex, fixed = T)
latex = gsub('\\end{tabular}', ' \\end{tabular}\n\\end{adjustbox}', latex, fixed = T)
writeLines(latex, con = Filename)

#Graph performance over test month
df.evaluation[, Multivar := Variables != "TTFFM "]
df.evaluation[Multivar == T, Type := "Multivariate"]
df.evaluation[Multivar != T, Type := "Univariate"]

plt.evaluation_months = ggplot(data = df.evaluation, aes(x = as.numeric(TestMonth), y = MSE)) + geom_line(aes(col = Model, linetype = Type)) + scale_x_continuous(breaks=c(117, 217,317,417,517,617,717), labels=c("January 17", "February 17", "March 17", "April 17", "May 17", "June 17", "July 17")) + theme_classic() + scale_color_uchicago("dark") + labs(title = "Test MSE across Test Months for all", x = "Test Month")

ggsave( "../Plots/level_evaluation_monthly_all.png",plt.evaluation_months, width = 8.11, height = 4.98)

#Multivariate models plus LaggedValue 
plt.evaluation_months_multivar = ggplot(data = df.evaluation[Type == "Multivariate" | Model == "LaggedValue"], aes(x = as.numeric(TestMonth), y = MSE)) + geom_line(aes(col = Model)) + scale_x_continuous(breaks=c(117, 217,317,417,517,617,717), labels=c("January 17", "February 17", "March 17", "April 17", "May 17", "June 17", "July 17")) + theme_classic() + scale_color_uchicago("dark") + labs(title = "Test MSE across test months for multivariate models", x = "Test Month")

ggsave( "../Plots/level_evaluation_monthly_multivar.png",plt.evaluation_months_multivar, width = 8.11, height = 4.98)

#Univariate Models
plt.evaluation_months_univar = ggplot(data = df.evaluation[Type == "Univariate" | Model == "LaggedValue"], aes(x = as.numeric(TestMonth), y = MSE)) + geom_line(aes(col = Model)) + scale_x_continuous(breaks=c(117, 217,317,417,517,617,717), labels=c("January 17", "February 17", "March 17", "April 17", "May 17", "June 17", "July 17")) + theme_classic() + scale_color_uchicago("dark") + labs(title = "Test MSE across test months for univariate models", x = "Test Month")

ggsave( "../Plots/level_evaluation_monthly_univar.png",plt.evaluation_months_univar, width = 8.11, height = 4.98)

#Compare LSTM and Lagged Value predictions
df.predictions = as.data.table(read.csv(file = "../Data/Output/LevelPrediction/level_eval/predictions.csv"))[,.SD[which.min(Month_Traded)],by = Date]
df.predictions[,Date := as.Date(Date)]
df.predictions = df.predictions[Model == 'lstm' & Variables == '',.(Date, LSTM = Prediction, LaggedValue = Reference, Month_Traded)]
df.predictions_long = as.data.table(gather(df.predictions, Type, Value,  LSTM, LaggedValue))


plt.predictions = ggplot(data = df.predictions_long[Date >= as.Date('2017-03-01')], aes(x = Date, y = Value)) + geom_line(aes(col = Type)) + theme_classic() + scale_color_uchicago("dark") + labs(title = "Predictions LSTM vs. LaggedValue vs. Actual", x = "Trading Day", y = "Price [EUR/MWh]")+ scale_x_date(labels = date_format("%m-%Y"))

ggsave( "../Plots/level_predictions.png",plt.predictions, width = 8.11, height = 4.98)

#Get mean absolute difference betwen predictions
df.predictions[,mean(abs(LSTM - LaggedValue)),by = Month_Traded]
df.predictions[,mean(abs(LSTM - LaggedValue))]
#Mean Absolute relative difference
df.predictions[,mean(abs(LSTM - LaggedValue)/LaggedValue), by = Month_Traded]
df.predictions[,mean(abs(LSTM - LaggedValue)/LaggedValue)]
#Mean Difference
df.predictions[,mean(LSTM - LaggedValue)]
