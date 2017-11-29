rm(list = ls())
library(lubridate)
library(ggsci)
library(data.table)
library(ggplot2)
library(xtable)

#Directory to save the plots
plot_directory = '../Plots/'
#Read in Data
data_path = '../Data/Input/InputData.csv'
df = read.csv(data_path)
df = as.data.table(df)
df[,Date := as.Date(Date)]

#Abbrevations of each variable to be plotted
abbrevations = c("TTFFM", "ConLDZNL", "ConNLDZNL",  "ProdNL", "ProdUKCS", "StorageNL", "StorageUK", "TradeBBL", "TradeIUK", "TTFDA", "NBPFM", "OilFM", "ElectricityBaseFM", "ElectricityPeakFM", "EURUSDFX", "EURGBPFX")

#Reduce data to those variables that are to be plotted
df = df[name %in% abbrevations,]

#Full names of variables
names = c("TTF Front Month Closing Price", "Dutch LDZ Consumption", "Dutch Non-LDZ Consumption", "Dutch Production", "UK Continental Shelf Production", "Dutch Storage Level", "UK Storage Level", "BBL Pipeline Flow", "Interconnector Pipeline Flow", "TTF Day Ahead Closing Price", "NBP Front Month Closing Price", "Brent Oil Front Month Closing Price",  "Phelix Base Load Front Month Closing Price", "Phelix Peak Load Front Month Closing Price", "USD / Euro Exchange Rate Closing", "GBP / EUR Exchange Rate Closing")

#Units of measurement for each variable
units = c("[EUR/MWh]", "[GWh]", "[GWh]", "[GWh]", "[million cubic meters]", "[% Capacity]", "[% Capacity]", "[GWh]", "[GWh]", "[EUR/MWh]", "[pence/Therm]", "[USD/Barrel]",  "[EUR/MWh]", "[EUR/MWh]", "[USD/EUR]", "[GBP/EUR]" )

#Plot each variable, while automatically creating labels, titles and filenames from the above data
for(i in 1:length(abbrevations)){
    print(abbrevations[i])
    print(names[i])
    plot_data = df[name == abbrevations[i]]
    plot =  ggplot(data = plot_data, aes(x = Date, y = CLOSE)) + geom_line() + theme_classic() + labs(title = names[i], subtitle = abbrevations[i], y =  units[i])
    ggsave(filename = paste0(plot_directory, abbrevations[i], 'level.png'), width = 8.11, height = 4.98)
}

##Data Availability Overview:
##only include dates where the target variable is available
ttf_fm_dates = unique(df[name == "TTFFM" & !is.na(CLOSE),]$Date)
df = df[Date %in% ttf_fm_dates,]

#Create Data.table containing the minimal and maximal date for each variable and number of observations where the Value (saved in Column CLOSE) is not NA
data_availability = df[!is.na(CLOSE),.(MinDate = as.character(min(Date)), MaxDate = as.character(max(Date)), NumObs = .N), by = .(Variable = name)]

#Save results as latex table
data_availability_latex = xtable(data_availability, label = 'tab:data-availability', caption = 'Data Availability of all Input Variables', digits = 0)
print(data_availability_latex, file = '../Latex/tables/data_availability.tex', include.rownames=FALSE)

#Create binary target variable for visualisation
binary_plot_data = df[name == "TTFFM"]
#Loop through all observations of TTF Front month
for(i in 1:nrow(binary_plot_data)){
    #Get Current Date, Month and Year
    currDate = binary_plot_data$Date[i]
    currMonth = month(currDate)
    currYear = year(currDate)
    #Get minimum value of TTF Front month for all observations in this Month/Year where the date is larger than the current date
    currMin = min(binary_plot_data [Date >= currDate & month(Date) == currMonth & year(Date) == currYear,]$CLOSE)
    binary_plot_data[Date == currDate,MinRemaining := currMin]
}
#Check for each observations if the current value is equal to the minimum across the remaining observations to get binary target variable
binary_plot_data$BinaryVar = as.numeric(binary_plot_data$CLOSE == binary_plot_data$MinRemaining)

#Drop September first 2017 since there is only one observation in this month
binary_plot_data = binary_plot_data[ Date != as.Date("2017-09-01")]

#Get positive shares of binary target variable for each month, to find months with many / few positive observations
binary_plot_data[,.(PositiveShare = mean(BinaryVar)), by = .(Year = as.character(year(Date)), Month = month(Date))]

#Plot for month with few positive observations
binary_plot_few_positive = ggplot(data = binary_plot_data[year(Date) == 2017 & month(Date) == 2], aes(x = Date)) + geom_line(aes(y = CLOSE)) + geom_point(aes(y = BinaryVar*max(binary_plot_data[year(Date) == 2017 & month(Date) == 2]$CLOSE))) + scale_y_continuous(sec.axis = sec_axis(~.*1/max(binary_plot_data[year(Date) == 2017 & month(Date) == 2]$CLOSE), name = "Binary Target")) + theme_classic() + labs(title = "Price level vs. binary target variable 02/2017", y = "Price Level [EUR/MWh]")

ggsave( "../Plots/binary_plot_few_positive.png",binary_plot_few_positive, width = 8.11, height = 4.98)

#Plot for month with many positive observations
binary_plot_many_positive = ggplot(data = binary_plot_data[year(Date) == 2016 & month(Date) == 12], aes(x = Date)) + geom_line(aes(y = CLOSE)) + geom_point(aes(y = BinaryVar*max(binary_plot_data[year(Date) == 2016 & month(Date) == 12]$CLOSE))) + scale_y_continuous(sec.axis = sec_axis(~.*1/max(binary_plot_data[year(Date) == 2016 & month(Date) == 12]$CLOSE), name = "Binary Target")) + theme_classic() + labs(title = "Price level vs. binary target variable 12/2016", y = "Price Level [EUR/MWh]")

ggsave( "../Plots/binary_plot_many_positive.png",binary_plot_many_positive, width = 8.11, height = 4.98)

#Balance overview table with annual share
balance_overview = binary_plot_data[,.(PositiveShare = mean(BinaryVar)), by = .(Year = as.character(year(Date)))]
balance_overview = rbind(balance_overview,binary_plot_data[,.(Year = "Total",PositiveShare = mean(BinaryVar))])

balance_overview_latex = xtable(balance_overview, label = 'tab:balance-overview', caption = 'Annual share of positive observation for binary target variable', digits = 3)
print(balance_overview_latex, file = '../Latex/tables/balance_overview.tex', include.rownames=FALSE)


