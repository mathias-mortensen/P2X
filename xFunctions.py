from pathlib import Path
import pandas as pd
import numpy as np
import csv
# ---------------------- GENERAL FUNCTIONS --------------------------------
def assign_var(df,var_name):
    look = df['Variable'].str.find(var_name)
    for i in range(0,len(df)):
        if look[i] == 0:
            var_name = df['Value'][i]
    return var_name 

# Auto-detect the delimiter of a csv file
def get_delimiter(file_path, bytes = 4096):
    sniffer = csv.Sniffer()
    data = open(file_path, "r").read(bytes)
    delimiter = sniffer.sniff(data).delimiter
    return delimiter

# ----------------------- READING DATA -------------------------------------


## PV INPUT DATA PROCESSING - can be changed to match the data file
def import_PV(filename, Start_date, End_date, Start_date_scen, End_date_scen):
    file_to_open = Path("Data/") / filename
    df_solar_prod = pd.read_excel(file_to_open)

    TimeRangePV = (df_solar_prod['Hour UTC'] >= Start_date) & (df_solar_prod['Hour UTC']  <= End_date)
    TimeRangePV_scen = (df_solar_prod['Hour UTC'] >= Start_date_scen) & (df_solar_prod['Hour UTC']  <= End_date_scen)

    df_solar_prod_time = df_solar_prod[TimeRangePV]
    PV_scen = df_solar_prod[TimeRangePV_scen]

    PV_scenPower = PV_scen['Power [MW]'].tolist() #Convert from pandas data series to list

    PV = df_solar_prod_time['Power [MW]'].tolist() #Convert from pandas data series to list

    P_PV_max = dict(zip(np.arange(1,len(PV)+1),PV))

    return P_PV_max

##  DAY-AHEAD MARKET DATA PROCESSING
def import_DA(file_name, Start_date, End_date, Start_date_scen, End_date_scen):
    file_to_open = Path("Data/") / file_name
    df_DKDA_raw = pd.read_csv(file_to_open,sep=';',decimal=',')

    #Converting to datetime
    df_DKDA_raw[['HourUTC','HourDK']] = df_DKDA_raw[['HourUTC','HourDK']].apply(pd.to_datetime)
    
    #Input for model
    TimeRange2020DA = (df_DKDA_raw['HourDK'] >= Start_date) & (df_DKDA_raw['HourDK']  <= End_date)
    TimeRangeScenarioDA = (df_DKDA_raw['HourDK'] >= Start_date_scen) & (df_DKDA_raw['HourDK']  <= End_date_scen)

    df_DKDA_raw2020 = df_DKDA_raw[TimeRange2020DA]
    df_DKDA_rawScen = df_DKDA_raw[TimeRangeScenarioDA]

    DA_list = df_DKDA_raw2020['SpotPriceEUR,,'].tolist()
    DA_list_scen = df_DKDA_rawScen['SpotPriceEUR,,'].tolist()


    DA = dict(zip(np.arange(1,len(DA_list)+1),DA_list))
    #print(DA,Start_date,End_date)

    #Getting time range
    DateRange = df_DKDA_raw2020['HourDK']

    return DA,DateRange


# FCR data import
def import_FCR(file_name, price_column,time_column, Start_date, End_date, Start_date_scen, End_date_scen):
    file_to_open = Path("Data/") / file_name
    df_FCR_DE_raw = pd.read_csv(file_to_open,sep=',',low_memory=False)
    df_FCR_DE_raw[price_column] = df_FCR_DE_raw[price_column].astype(float)

    #Input for model
    TimeRange_FCR = (df_FCR_DE_raw[time_column] >= Start_date) & (df_FCR_DE_raw['DATE_FROM']  <= End_date)
    TimeRangeFCR_Scen = (df_FCR_DE_raw[time_column] >= Start_date_scen) & (df_FCR_DE_raw['DATE_FROM']  <= End_date_scen)

    df_FCR_DE = df_FCR_DE_raw[TimeRange_FCR]
    df_FCR_DE_scen = df_FCR_DE_raw[TimeRangeFCR_Scen]

    #Convert from pandas data series to list
    list_FCR = df_FCR_DE[price_column].tolist() 
    c_FCR = dict(zip(np.arange(1,len(list_FCR)+1),list_FCR))

    return c_FCR

# aFRR data import function 
def import_aFRR(file_name, Start_date, End_date, Start_date_scen, End_date_scen):
    file_to_open = Path("Data/") / file_name
    df_aFRR_raw = pd.read_excel(file_to_open)

    #reduce data point to the chosen time period
    TimeRange_aFRR = (df_aFRR_raw['Period'] >= Start_date) & (df_aFRR_raw['Period']  <= End_date)
    TimeRange_aFRR_Scen = (df_aFRR_raw['Period'] >= Start_date_scen) & (df_aFRR_raw['Period']  <= End_date_scen)

    df_aFRR = df_aFRR_raw[TimeRange_aFRR]
    df_aFRR_scen = df_aFRR_raw[TimeRange_aFRR_Scen]

    #convert to list
    list_aFRR_up = df_aFRR['aFRR Upp Pris (EUR/MW)'].tolist() #Convert from pandas data series to list
    list_aFRR_down = df_aFRR['aFRR Ned Pris (EUR/MW)'].tolist() #Convert from pandas data series to list

    #convert to dict
    c_aFRR_up = dict(zip(np.arange(1,len(list_aFRR_up)+1),list_aFRR_up))
    c_aFRR_down = dict(zip(np.arange(1,len(list_aFRR_down)+1),list_aFRR_down))

    return c_aFRR_up,c_aFRR_down

# mFRR data import function
def import_mFRR(file_name, Start_date, End_date, Start_date_scen, End_date_scen):

    file_to_open = Path("Data/") / file_name
    df_DKmFRR_raw = pd.read_csv(file_to_open,sep=';', decimal=',')

    #Converting to datetime
    df_DKmFRR_raw[['HourUTC','HourDK']] =  df_DKmFRR_raw[['HourUTC','HourDK']].apply(pd.to_datetime)
    df_mFRR = df_DKmFRR_raw.iloc[0:24095,:]
    df_mFRR_raw = df_mFRR[::-1]
    sum(df_mFRR['mFRR_UpPriceEUR'])

    TimeRange_mFRR = (df_mFRR_raw['HourDK'] >= Start_date) & (df_mFRR_raw['HourDK']  <= End_date)
    TimeRange_mFRR_Scen = (df_mFRR_raw['HourDK'] >= Start_date_scen) & (df_mFRR_raw['HourDK']  <= End_date_scen)

    df_mFRR = df_mFRR_raw[TimeRange_mFRR]
    df_mFRR_scen = df_mFRR_raw[TimeRange_mFRR_Scen]

    #convert to list
    list_mFRR_up = df_mFRR['mFRR_UpPriceEUR'].tolist() #Convert from pandas data series to list

    #convert to dict
    c_mFRR_up = dict(zip(np.arange(1,len(list_mFRR_up)+1),list_mFRR_up))

    return c_mFRR_up


# Function designed for importing and converting hourly time series csv file (delimiter-flexible))
def import_generic(file_name, price_column,time_column, Start_date, End_date, Start_date_scen, End_date_scen):
    file_to_open = Path("Data/") / file_name
    if '.csv' in file_name:
        df_raw = pd.read_csv(file_to_open,sep=get_delimiter(file_to_open),decimal = ',',low_memory=False)
        df_raw[price_column] = df_raw[price_column].astype(float)
    elif '.xlsx' in file_name:
        df_raw = pd.read_excel(file_to_open)

    #Input for model
    TimeRange = (df_raw[time_column] >= Start_date) & (df_raw[time_column]  <= End_date)
    TimeRange_Scen = (df_raw[time_column] >= Start_date_scen) & (df_raw[time_column]  <= End_date_scen)

    df_generic = df_raw[TimeRange]
    df_generic_scen = df_raw[TimeRange_Scen]

    #Convert from pandas data series to list
    list_generic = df_generic[price_column].tolist() 
    c_generic = dict(zip(np.arange(1,len(list_generic)+1),list_generic))
    return c_generic

def demand_assignment(Demand_pattern,TimeRange,k_d):
    demand = list(0 for i in range(0,len(TimeRange)))
    
    if Demand_pattern == 'Hourly':
        for i in range(0,len(demand),1):
            demand[i] = k_d
    
    if Demand_pattern == 'Daily':
        for i in range(0,len(demand),24):
            demand[i+23] = k_d*24    
    
    if Demand_pattern == 'Weekly':
        for i in range(1,1+int(len(TimeRange)/(24*7))):
            demand[i*24*7-1] = k_d*24*7
        dw = len(TimeRange)/(24*7) - int(len(TimeRange)/(24*7))
        if dw > 0:
            demand[len(TimeRange)-1] = dw*7*24*k_d     
    
    demand = dict(zip(np.arange(1,len(demand)+1),demand))
    
    return demand