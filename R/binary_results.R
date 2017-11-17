rm(list = ls())
library(data.table)
library(tidyr)
library(ggplot2)
library(ggsci)
library(MLmetrics)
library(plyr)
library(xtable)

chr.par_tuning_directory = "../Data/Output/BinaryPrediction/binary_par_tuning"
chr.var_selection_directory = "../Data/Output/BinaryPrediction/binary_var_selection"
chr.multivar_par_tuning_directory = "../Data/Output/BinaryPrediction/binary_eval"
chr.evaluation_directory = "../Data/Output/BinaryPrediction/binary_eval"

#Analysis of Par-Tuning
df.par_tuning = as.data.table(read.csv(file = paste(chr.par_tuning_directory, "evaluation.csv", sep = "/")))
df.par_tuning[, Model:= toupper(Model)]
df.par_tuning[, Model := gsub('FFNN_REGRESSION','Regression',Model)]
df.par_tuning = rename(df.par_tuning, c(Architecture = "HiddenNeurons", binary_crossentropy = 'BCE'))
df.par_tuning_mins = df.par_tuning[,.SD[which.min(BCE)],by = Model]

df.par_tuning_mins = df.par_tuning_mins[,.(Model, HiddenNeurons, Dropout, LearningRate, BCE)]

Filename = '../Latex/tables/binary_par_tuning_short.tex'
caption = 'Selected parameter combinations in univariate tuning step of binary prediction'
label = 'tab:binary.par.tuning.short'
latex = xtable(df.par_tuning_mins, label = label, caption = caption, digits = 4)
print(latex, file = Filename)


Filename = '../Latex/tables/binary_par_tuning_full.tex'
caption = 'Full Results in univariate tuning step of binary prediction'
label = 'tab:binary.par.tuning.full'
latex = xtable(df.par_tuning[,.(Model, HiddenNeurons, Dropout, LearningRate, BCE)], label = label, caption = caption, digits = 4)
print(latex, file = Filename)

#Analysis of Var-Selection
df.var_selection = as.data.table(read.csv(file = paste(chr.var_selection_directory, "evaluation.csv", sep = "/")))
df.var_selection[, Model:= toupper(Model)]
df.var_selection[, Model := gsub('FFNN_REGRESSION','Regression',Model)]
df.var_selection[, Variables := paste0('TTFFM ',gsub('_', ' ', Vars))]
df.var_selection = rename(df.var_selection, c( binary_crossentropy = 'BCE'))
df.var_selection_mins = df.var_selection[,.SD[which.min(BCE)],by = Model]

df.var_selection_mins = df.var_selection_mins[,.(Model, Variables, BCE)]

Filename = '../Latex/tables/binary_var_selection_short.tex'
caption = 'Selected variable combinations in variable selection step of binary prediction'
label = 'tab:binary.var.selection.short'
latex = xtable(df.var_selection_mins, label = label, caption = caption, digits = 4)
print(latex, file = Filename)


Filename = '../Latex/tables/binary_var_selection_full.tex'
caption = 'Full Results in variable selection step of binary prediction'
label = 'tab:binary.var.selection.full'
latex = xtable(df.var_selection[,.(Model, Variables, BCE)], label = label, caption = caption, digits = 4)
print(latex, file = Filename)

#Analysis of Par-Tuning
df.multivar_par_tuning = as.data.table(read.csv(file = paste(chr.multivar_par_tuning_directory, "evaluation.csv", sep = "/")))
df.multivar_par_tuning[, Model:= toupper(Model)]
df.multivar_par_tuning[, Model := gsub('FFNN_REGRESSION','Regression',Model)]
df.multivar_par_tuning = rename(df.multivar_par_tuning, c(Architecture = "HiddenNeurons", binary_crossentropy = 'BCE'))
df.multivar_par_tuning_mins = df.multivar_par_tuning[,.SD[which.min(BCE)],by = Model]

df.multivar_par_tuning_mins = df.multivar_par_tuning_mins[,.(Model, HiddenNeurons, Dropout, LearningRate, BCE)]

Filename = '../Latex/tables/binary_multivar_par_tuning_short.tex'
caption = 'Selected parameter combinations in multivariate tuning step of binary prediction'
label = 'tab:binary.multivar.par.tuning.short'
latex = xtable(df.multivar_par_tuning_mins, label = label, caption = caption, digits = 4)
print(latex, file = Filename)


Filename = '../Latex/tables/binary_multivar_par_tuning_full.tex'
caption = 'Full Results in multivariate tuning step of binary prediction'
label = 'tab:binary.multivar.par.tuning.full'
latex = xtable(df.multivar_par_tuning[,.(Model, HiddenNeurons, Dropout, LearningRate, BCE)], label = label, caption = caption, digits = 4)
print(latex, file = Filename)

#Analysis of Model Evaluation
df.evaluation = as.data.table(read.csv(file = paste(chr.evaluation_directory, "evaluation.csv", sep = "/")))
df.evaluation[, Model:= toupper(Model)]
df.evaluation[, Model := gsub('FFNN_REGRESSION','Regression',Model)]
df.evaluation[, Variables := paste0('TTFFM ',gsub('_', ' ', Variables))]
df.evaluation = rename(df.evaluation, c(Architecture = "HiddenNeurons", binary_crossentropy = 'BCE'))
df.evaluation = rbind(df.evaluation[,.(Model, Variables,TestMonth = gsub('TTF','',TestMonth), BCE)],  df.evaluation[Model == 'LSTM' & Variables == "TTFFM ",.(Model = 'LaggedValue', Variables = "TTFFM ",TestMonth = gsub('TTF','',TestMonth), BCE = binary_crossentropyref)])

df.evaluation_mean = df.evaluation[,.(BCE = mean(BCE)), by = .(Model, Variables)]
df.evaluation_mean = df.evaluation_mean[order(BCE),]


Filename = '../Latex/tables/binary_eval_short.tex'
caption = 'Average BCE across months for each model in evaluation step'
label = 'tab:binary.eval.short'
latex = xtable(df.evaluation_mean, label = label, caption = caption, digits = 4)
print(latex, file = Filename)

Filename = '../Latex/tables/binary_eval_full.tex'
caption = 'Full results by testing month for each model in evaluation step'
label = 'tab:binary.eval.short'
latex = xtable(df.evaluation, label = label, caption = caption, digits = 4)
print(latex, file = Filename)



