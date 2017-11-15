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



