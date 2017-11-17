rm(list = ls())
library(data.table)
library(tidyr)
library(ggplot2)
library(ggsci)
library(MLmetrics)

chr.results_directory =  "../Data/Output/BinaryPrediction/binary_eval/"

df.evaluation = as.data.table(read.csv(file = paste(chr.results_directory, "evaluation.csv", sep = "/")))

df.evaluation = df.evaluation[,.(CrossEntropy = mean(binary_crossentropy), CrossEntropyReference = mean(binary_crossentropyref)), by = .(Model, Variables, LearningRate, Dropout) ]

df.evaluation = df.evaluation[order(CrossEntropy),]

Filename = '../../Latex/tables/min_eval_cv.tex'
caption = 'Test Results using monthly cross validation of tuned models for data 01 - 08/2017'
label = 'tab:min.eval.cv'
latex = xtable(df.evaluation, label = label, caption = caption, digits = 4)
print(latex, file = Filename)

