rm(list = ls())
library(lubridate)
library(ggsci)
library(data.table)
library(ggplot2)
library(xtable)

plot_directory = '../Plots/'
#Read in Data
df = read.csv('../Data/Input/InputData.csv')
df = as.data.table(df)
df[,Date := as.Date(Date)]


abbrevations = c("TTFFM", "ConLDZNL", "ConNLDZNL",  "ProdNL", "ProdUKCS", "StorageNL", "StorageUK", "TradeBBL", "TradeIUK", "TTFDA", "NBPFM", "OilFM", "ElectricityBaseFM", "ElectricityPeakFM", "EURUSDFX", "EURGBPFX")

df = df[name %in% abbrevations,]

names = c("TTF Front Month Closing Price", "Dutch LDZ Consumption", "Dutch Non-LDZ Consumption", "Dutch Production", "UK Continental Shelf Production", "Dutch Storage Level", "UK Storage Level", "BBL Pipeline Flow", "Interconnector Pipeline Flow", "TTF Day Ahead Closing Price", "NBP Front Month Closing Price", "Brent Oil Front Month Closing Price",  "Phelix Base Load Front Month Closing Price", "Phelix Peak Load Front Month Closing Price", "USD / Euro Exchange Rate Closing", "GBP / EUR Exchange Rate Closing")

units = c("[EUR/MWh]", "[GWh]", "[GWh]", "[GWh]", "[million cubic meters]", "[% Capacity]", "[% Capacity]", "[GWh]", "[GWh]", "[EUR/MWh]", "[pence/Therm]", "[USD/Barrel]",  "[EUR/MWh]", "[EUR/MWh]", "[USD/EUR]", "[GBP/EUR]" )

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

data_availability = df[!is.na(CLOSE),.(MinDate = as.character(min(Date)), MaxDate = as.character(max(Date)), NumObs = .N), by = .(Variable = name)]

data_availability_latex = xtable(data_availability, label = 'tab:data-availability', caption = 'Data Availability of all Input Variables', digits = 0)
print(data_availability_latex, file = '../Latex/tables/data_availability.tex', include.rownames=FALSE)


binary_plot_data = df[name == "TTFFM"]
for(i in 1:nrow(binary_plot_data)){
    currDate = binary_plot_data$Date[i]
    currMonth = month(currDate)
    currYear = year(currDate)
    currMin = min(binary_plot_data [Date >= currDate & month(Date) == currMonth & year(Date) == currYear,]$CLOSE)
    binary_plot_data[Date == currDate,MinRemaining := currMin]
}

binary_plot_data$BinaryVar = as.numeric(binary_plot_data$CLOSE == binary_plot_data$MinRemaining)

#Drop September first 2017 since there is only one observation in this month
binary_plot_data = binary_plot_data[ Date != as.Date("2017-09-01")]

binary_plot_data[,.(PositiveShare = mean(BinaryVar)), by = .(Year = as.character(year(Date)), Month = month(Date))]

#Plot for month with few positive observations
binary_plot_few_positive = ggplot(data = binary_plot_data[year(Date) == 2017 & month(Date) == 2], aes(x = Date)) + geom_line(aes(y = CLOSE)) + geom_point(aes(y = BinaryVar*max(binary_plot_data[year(Date) == 2017 & month(Date) == 2]$CLOSE))) + scale_y_continuous(sec.axis = sec_axis(~.*1/max(binary_plot_data[year(Date) == 2017 & month(Date) == 2]$CLOSE), name = "Binary Target")) + theme_classic() + labs(title = "Price level vs. binary target variable 02/2017", y = "Price Level [EUR/MWh]")

ggsave( "../Plots/binary_plot_few_positive.png",binary_plot_few_positive, width = 8.11, height = 4.98)

binary_plot_many_positive = ggplot(data = binary_plot_data[year(Date) == 2016 & month(Date) == 12], aes(x = Date)) + geom_line(aes(y = CLOSE)) + geom_point(aes(y = BinaryVar*max(binary_plot_data[year(Date) == 2016 & month(Date) == 12]$CLOSE))) + scale_y_continuous(sec.axis = sec_axis(~.*1/max(binary_plot_data[year(Date) == 2016 & month(Date) == 12]$CLOSE), name = "Binary Target")) + theme_classic() + labs(title = "Price level vs. binary target variable 12/2016", y = "Price Level [EUR/MWh]")

ggsave( "../Plots/binary_plot_many_positive.png",binary_plot_many_positive, width = 8.11, height = 4.98)

#Balance overview table with annual share
balance_overview = binary_plot_data[,.(PositiveShare = mean(BinaryVar)), by = .(Year = as.character(year(Date)))]
balance_overview = rbind(balance_overview,binary_plot_data[,.(Year = "Total",PositiveShare = mean(BinaryVar))])

balance_overview_latex = xtable(balance_overview, label = 'tab:balance-overview', caption = 'Annual share of positive observation for binary target variable', digits = 3)
print(balance_overview_latex, file = '../Latex/tables/balance_overview.tex', include.rownames=FALSE)


