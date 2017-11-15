rm(list = ls())
library(data.table)
library(tidyr)
library(ggplot2)
library(ggsci)
library(MLmetrics)


chr.results_directory = "../Data/Output/LevelPrediction/level_eval/"

df.evaluation = as.data.table(read.csv(file = paste(chr.results_directory, "evaluation.csv", sep = "/")))

df.evaluation = df.evaluation[,.(MSE = mean(mse), MSEReference = mean(mseref)), by = .(Model, Variables, LearningRate, Dropout) ]

df.evaluation = df.evaluation[order(MSE),]


Filename = '../../Latex/tables/val_eval_cv.tex'
caption = 'Test Results using monthly cross validation of tuned models for data 01 - 07/2017'
label = 'tab:min.eval.cv'
latex = xtable(df.evaluation, label = label, caption = caption, digits = 3, row.names = FALSE)
print(latex, file = Filename)

