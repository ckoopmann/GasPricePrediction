import eikon as ek
import pandas as pd
import os
ek.set_app_id('D301711ACA93A1D7E41BDA7')
# test = ek.get_news_headlines('R:LHAG.DE', date_from='2017-04-05T09:00:00', date_to='2017-04-05T18:00:00')
#
# gas = ek.get_timeseries(["TRNLTTFMF7"], fields=["Close","Open"], start_date = "2016-03-01", end_date = "2017-03-27",
#                             interval="daily")
#
# front_month = ek.get_timeseries(["TRDEBMQ7^1"], start_date = "2014-01-01",
#                             interval="daily")

def read_data(data_list_path, output_directory, avoid_replacement = False):
    data_list = pd.read_csv(data_list_path, ";")
    df_list = []
    for index, row in data_list.iterrows():
        try:
            if avoid_replacement and os.path.isfile(os.path.join(output_directory,row['Name'] + '.csv')):
                continue
            df = ek.get_timeseries([row['Id']], start_date="2010-01-01")
            df['name'] = row['Name']
            df.to_csv(os.path.join(output_directory,row['Name'] + '.csv'))
            df_list.append(df)
        except:
            print("Error on reading in:" + row['Name'])
    if len(df_list) > 0:
        all_data = pd.concat(df_list)
    else:
        all_data = None
    return(all_data)

DataList_Directory = "../Data/DataLists"
Output_Directory = "../Data/Input"
DF_List = []
for Data_List in os.listdir(DataList_Directory):
    DataList_Path = os.path.join(DataList_Directory, Data_List)
    DF = read_data(DataList_Path,Output_Directory)
    DF_List.append(DF)
AllData = pd.concat(DF_List)
AllData.to_csv(os.path.join(Output_Directory,'InputData' + '.csv'))
