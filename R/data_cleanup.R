rm(list = ls())
library(data.table)
library(lubridate)

df.template_default = read.csv('GasFundamentals/AllData.csv', nrows = 1)





fun.read_long <- function(str.colnames, str.filepath, df.template = df.template_default, ls.sum_cols = NULL){
    df.Input = read.csv2(str.filepath, stringsAsFactors = FALSE, nrows = length(readLines(str.filepath))-2)
    print(head(df.Input))
    if(!is.null(ls.sum_cols)){
        for(chr.sum_cols in ls.sum_cols){
            df.Input[,chr.sum_cols[1]] = rowSums(df.Input[,chr.sum_cols], na.rm = TRUE)
            df.Input[,chr.sum_cols[-1]] = NULL
        }
    }
    names(df.Input) = str.colnames
    if('Date' %in% names(df.Input)){
        df.Input$Date = as.Date(df.Input$Date, format = "%d.%m.%Y")
    }
    df.output = df.template[0,]
    df.output$Date = as.Date(df.output$Date)
    print(names(df.Input))
    for(var in setdiff(names(df.Input), 'Date')){
        df.newdata = data.table(Date = df.Input[!is.na(df.Input[,var]),]$Date, CLOSE = df.Input[!is.na(df.Input[,var]),var], COUNT = NA, HIGH = NA, LOW = NA, OPEN = NA, VOLUME = NA, name = var)
        df.output = rbind(df.output, df.newdata)
    }
    return(df.output)
}

df.ConNWELDZ = fun.read_long(str.colnames = c('Date','ConLDZDE','ConLDZFR', 'ConLDZNL','ConLDZBE'), 
                             str.filepath = 'GasFundamentals/ConsumptionEULDZ.csv')

df.ConNWELDZ_agg = df.ConNWELDZ[,.(CLOSE = sum(CLOSE), COUNT = NA, HIGH = NA, LOW = NA, OPEN = NA, VOLUME = NA, name = "ConLDZEU"), by = Date]

df.ConNWENLDZ = fun.read_long(str.colnames = c('Date','ConNLDZDE','ConNLDZFR', 'ConNLDZNL','ConNLDZBE'), 
                              str.filepath = 'GasFundamentals/ConsumptionEUNLDZ.csv',
                              ls.sum_cols = list(c('France.Ind','France.Gas.for.Power'), c('Belgium.Ind', 'Belgium.Gsa.for.Power')))

df.ConNWENLDZ_agg = df.ConNWELDZ[,.(CLOSE = sum(CLOSE), COUNT = NA, HIGH = NA, LOW = NA, OPEN = NA, VOLUME = NA, name = "ConNLDZEU"), by = Date]

df.ConUK = fun.read_long(str.colnames = c('Date','ConIndUK','ConPowUK', 'ConLDZUK'), 
                         str.filepath = 'GasFundamentals/ConsumptionUK.csv')

df.ProdNL = fun.read_long(str.colnames = c('Date','ProdNL'), 
                          str.filepath = 'GasFundamentals/DailyProdNL.csv')

df.LNGStockNWE = fun.read_long(str.colnames = c('Date','LNGStockNL','LNGStockBE', 'LNGStockFR', 'LNGStockIT'), 
                          str.filepath = 'GasFundamentals/LNGStockNWE.csv')

df.LNGStockNWE_agg = df.LNGStockNWE[,.(CLOSE = sum(CLOSE), COUNT = NA, HIGH = NA, LOW = NA, OPEN = NA, VOLUME = NA, name = "LNGStockEU"), by = Date]

df.LNGStockUK = fun.read_long(str.colnames = c('Date','LNGStockUK','LNGStorCapUK'), 
                               str.filepath = 'GasFundamentals/LNGStockUKreduced.csv')

df.ProdUKCS = fun.read_long(str.colnames = c('Date','ProdUKCS'), 
                              str.filepath = 'GasFundamentals/ProductionUKCS.csv')

df.TradeBBL = fun.read_long(str.colnames = c('Date','TradeBBL', 'CapBBL'), 
                            str.filepath = 'GasFundamentals/TradeBBL.csv')

df.TradeIUK = fun.read_long(str.colnames = c('Date','TradeIUK', 'SpreadNBPZEE'), 
                            str.filepath = 'GasFundamentals/TradeIUk.csv')

df.TradeNOEU = fun.read_long(str.colnames = c('Date','TradeNONWE', 'TradeNOUK','CapNONWE'), 
                             str.filepath = 'GasFundamentals/TradeNOEU.csv')

df.TradeRUNWE = fun.read_long(str.colnames = c('Date','TradeRUNWE'), 
                              ls.sum_cols = list(c("Velke.Kapusany", 'Mallnow', 'Nord.Stream.OPAL', 'Nord.Stream.NEL','Poland','Beregovo..HU.')),
                              str.filepath = 'GasFundamentals/TradeRUNWE.csv')

fun.read_wide <- function(str.filepath, chr.varname, chr.date_format = '%d/%m %H:%M:%S', df.template = df.template_default){
    
    df.Input = read.csv2(str.filepath, stringsAsFactors = FALSE, nrows = length(readLines(str.filepath))-2)
    df.Input[[1]] = as.Date(df.Input[[1]], format = chr.date_format)
    names(df.Input)[1] = "Date"
    df.output = df.template[0]
    for(var in names(df.Input)[-1]){
        num.year = as.numeric(substr(var,2,5))
        df.newdata = data.table(Date = df.Input[!is.na(df.Input[,var]),]$Date, CLOSE = df.Input[!is.na(df.Input[,var]),var], COUNT = NA, HIGH = NA, LOW = NA, OPEN = NA, VOLUME = NA, name = chr.varname)
        year(df.newdata$Date) = num.year
        df.output = rbind(df.output, df.newdata)
    }
    return(df.output)
}


df.StorageAT = fun.read_wide(str.filepath = 'GasFundamentals/StorageAT.csv',
                             chr.varname = 'StorageAT')
df.Storage = df.template_default[0]
chr.storage_files = list.files('GasFundamentals/', full.names = TRUE)[grep('Storage',list.files('GasFundamentals/'))]
chr.storage_names = gsub(pattern = '.csv', replacement = '', x = list.files('GasFundamentals/')[grep('Storage',list.files('GasFundamentals/'))])

for(i in 1:length(chr.storage_names)){
    chr.storage_file = chr.storage_files[i]
    chr.storage_name = chr.storage_names[i]
    df.new_storage = fun.read_wide(str.filepath = chr.storage_file,
                                   chr.varname = chr.storage_name)
    df.Storage = rbind(df.Storage, df.new_storage)
}
rm(list = c("df.new_storage", "df.template_default"))

df.Storage_agg = df.Storage[name != "StorageUK",.(CLOSE = sum(CLOSE), COUNT = NA, HIGH = NA, LOW = NA, OPEN = NA, VOLUME = NA, name = "StorageEU"), by = Date]

fun.append_data <- function(){
    ls.data = list()
    for(name in ls(name = ".GlobalEnv" , pattern = '^df')){
        print(name)
        new_df = eval(parse(text = name))
        print(class(new_df))
        ls.data = c(ls.data,list(new_df))
    }
    
    return(rbindlist(ls.data))
}

cleaned_up = fun.append_data()

write.csv(cleaned_up, "manual_cleanup.csv", row.names = FALSE)

