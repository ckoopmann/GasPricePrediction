automatic_data = read.csv('AllData.csv')
manual_data = read.csv('manual_cleanup.csv')
all_data = rbind(automatic_data, manual_data)
all_data = all_data[!is.na(all_data$Date),]
write.csv(all_data, 'Data.csv', row.names = FALSE)