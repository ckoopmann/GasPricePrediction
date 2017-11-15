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


