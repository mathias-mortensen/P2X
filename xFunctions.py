from pathlib import Path
import pandas as pd
import numpy as np
import csv
import random
from sklearn_extra.cluster import KMedoids # for Kmedoids function

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

    DA_list = df_DKDA_raw2020['SpotPriceEUR'].tolist()
    DA_list_scen = df_DKDA_rawScen['SpotPriceEUR'].tolist()


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
def import_generic(file_name, price_column,time_column, Start_date, End_date, format):
    file_to_open = Path("Data/") / file_name
    if '.csv' in file_name:
        df_raw = pd.read_csv(file_to_open,sep=get_delimiter(file_to_open),decimal = ',',low_memory=False)
        df_raw[price_column] = df_raw[price_column].astype(float)
    elif '.xlsx' in file_name:
        df_raw = pd.read_excel(file_to_open)

    #Input for model
    TimeRange = (df_raw[time_column] >= Start_date) & (df_raw[time_column]  <= End_date)
    #TimeRange_Scen = (df_raw[time_column] >= Start_date_scen) & (df_raw[time_column]  <= End_date_scen)

    df_generic = df_raw[TimeRange]
    #df_generic_scen = df_raw[TimeRange_Scen]

    #Convert from pandas data series to list
    list_generic = df_generic[price_column].tolist()
    if format == 'list':
        return list_generic
    elif format == 'dict': 
        c_generic = dict(zip(np.arange(1,len(list_generic)+1),list_generic))
        return c_generic
    else:
        return 0

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


def Bootsrap(Type,Data,Data_names,n_samples,blocksize,sample_length):

    if Type == 'single':
        DA_block = []
        FCR_block = []
        aFRR_up_block = []
        aFRR_down_block = []
        mFRR_block = []


        for x in range(0,len(Data)):
            #Sample length
            n = len(Data[x])

            #Split sample in blocks of length blocksize

            blocks = [Data[x][i:i + blocksize] for i in range (0,n,blocksize)]

            #Delete last block if length differs from blocksize 
            if len(blocks[-1]) != blocksize:
                del blocks[-1]


            samples = np.zeros((n_samples,sample_length))

            for i in range(0,n_samples):
                t = 0
                while t < sample_length:

                    r = random.randrange(0,len(blocks))
                   

                    for j in range(0,blocksize):
                        samples[i,t+j] = blocks[r][j]

                    t = t + blocksize
        
            if Data_names[x] == 'DA':
                DA_block = samples
            if Data_names[x] == 'FCR':
                FCR_block = samples
            if Data_names[x] == 'aFRR Up':
                aFRR_up_block = samples
            if Data_names[x] == 'aFRR Down':
                aFRR_down_block = samples
            if Data_names[x] == 'mFRR':
                mFRR_block = samples


                
    if Type == 'combined':

        ########## Multi Blocks ######## 
        ### Multi ### 
        df = pd.DataFrame({'DA':  np.array(Data[0]) , 'aFRR Up:':  np.array(Data[1]), 'aFRR Down': np.array(Data[2]), 'mFRR':  np.array(Data[3])})

        data = df.values.tolist() #Acces element by  data[0][0]

        n = len(data)

        #Split sample in blocks of length blocksize

        blocks = [data[i:i + blocksize ] for i in range (0,n,blocksize)]#Acces element by blocks[0][0][0]

        #Delete last block if length differs from blocksize 
        if len(blocks[-1]) != blocksize:
            del blocks[-1]

        len_element = len(blocks[1][1])

        ## creating an array with zeros and same dimensions as blocks 
        samples = np.zeros((n_samples,sample_length,len_element))

        for i in range(0,n_samples):
            t = 0
            while t < sample_length:

                r = random.randrange(0,len(blocks))
                

                for j in range(0,blocksize):
                
                    samples[i,t+j] = blocks[r][j]

                t = t + blocksize

        Combined_blocks = samples

        
    if Type == 'single':
        
            return  DA_block,FCR_block,aFRR_up_block,aFRR_down_block,mFRR_block

    if Type == 'combined': 
        return Combined_blocks


def K_Medoids(scenarios,n_clusters): #Scenario reduction from all scenarios to n_clusters 
    Red_Scen = []   ## Red_Scen[0] = DA scenarios, Red_Scen[1] = FCR scenarios, Red_Scen[2] = aFRR_up scenarios, Red_Scen[3] = aFRR_Down scenarios, Red_Scen[4] = mFRR scenarios 
    Prob = np.zeros((n_clusters,len(scenarios))) # Prob scenario 1 in DA = Prob[0,0], Prob scenario 2 in DA = Prob[1,0] osv... Prob scenario 1 FCR = Prob[0,1] .....   

    for i in range(0,len(scenarios)):
    
        kmedoids = KMedoids(n_clusters=n_clusters,metric='euclidean').fit(scenarios[i])
        Red_Scen.append(kmedoids.cluster_centers_) # Calculating scenario probability ## 
       
        for j in range(0,n_clusters):
            Prob[j,i] = np.count_nonzero(kmedoids.labels_ == j)/len(kmedoids.labels_)
    
    return Red_Scen,Prob

def SingleInputData(Rep_scen,Prob):

    x = len(Rep_scen[0]) # same as n_clusters
    hours = len(Rep_scen[0][0]) # same as sample_length
    Ω = x**4 #number of "reserve scenarios"
    Φ = x # number of DA scenarios

    c_FCRs = {}
    c_aFRR_ups = {}
    c_aFRR_downs = {}
    c_mFRR_ups = {}
    π_r = {} #probability?

    for a in range(1,x+1):
        for b in range(1,x+1):
            for c in range(1,x+1):
                for d in range(1,x+1):
                
                    w = (a-1)*x**3 + (b-1)*x**2 + (c-1)*x + d
                    π_r[w] = Prob[a-1,1] * Prob[b-1,2] * Prob[c-1,3] * Prob[d-1,4] 
                    
                    for t in range(1,hours+1):

                        c_FCRs[(w,t)] = Rep_scen[1][a-1][t-1]
                        c_aFRR_ups[(w,t)] = Rep_scen[2][b-1][t-1]
                        c_aFRR_downs[(w,t)] = Rep_scen[3][c-1][t-1]
                        c_mFRR_ups[(w,t)] = Rep_scen[4][d-1][t-1]
    

    c_DAs = {}
    π_DA = {}
    for i in range(1,x+1):
        π_DA[(i)] = Prob[i-1,0] 
        for t in range(1,hours+1):
            c_DAs[(i,t)] = Rep_scen[0][i-1][t-1]

    return Φ, Ω,c_FCRs,c_aFRR_ups,c_aFRR_downs,c_mFRR_ups,c_DAs,π_r,π_DA


def GenAverage(scenarios,n_samples,sample_length):
    Avg_scenarios = np.zeros((n_samples,sample_length))

    for i in range(0,n_samples):
        for j in range(0,sample_length):
            Avg_scenarios[i][j] = scenarios[i][j].mean()
    return Avg_scenarios

def AvgKmedReduction(Avg_scenarios,scenarios,n_clusters,n_samples,sample_length):

    Red_Scen = []   ## Red_Scen[0] = DA scenarios, Red_Scen[1] = FCR scenarios, Red_Scen[2] = aFRR_up scenarios, Red_Scen[3] = aFRR_Down scenarios, Red_Scen[4] = mFRR scenarios 
    Prob = np.zeros(n_clusters) # Prob scenario 1 in DA = Prob[0,0], Prob scenario 2 in DA = Prob[1,0] osv... Prob scenario 1 FCR = Prob[0,1] .....   
    kmedoids = KMedoids(n_clusters=n_clusters,metric='euclidean').fit(Avg_scenarios)

    ## Calculating scenario probability ## 
    Red_Scen.append(kmedoids.cluster_centers_) 
    for j in range(0,n_clusters):
        Prob[j] = np.count_nonzero(kmedoids.labels_ == j)/len(kmedoids.labels_)

    true = 0
    index = []

    for j in range(0,len(Red_Scen[0])):   
        for x in range(0,n_samples):     
            for i in range(0,sample_length):
            
                if Red_Scen[0][j][i] == scenarios[x][i].mean():
                    true = true+1
                    
                    if true == sample_length:
                        index.append(x)
                        
                if Red_Scen[0][j][i] != scenarios[x][i].mean():
                    true = 0
        else:
            continue
        
    rep_senc1 = []
    for i in index:
        rep_senc1.append(scenarios[i]) 

    Rep_scen1 = np.zeros((len(rep_senc1[0][0]),n_clusters,sample_length))

    for i in range(0,len(rep_senc1[0][0])): # nr markets
        for j in range(0,n_clusters):
            Rep_scen1[i][j] = rep_senc1[j][:,i]

    return Rep_scen1, Prob

# Combined scenario generation
def scenario_comb(scenarios,n_samples,sample_length,n_clusters,c_FCR_scen,blocksize):
    Avg_scenarios = GenAverage(scenarios,n_samples,sample_length)
    Rep_scen_comb, Prob_comb = AvgKmedReduction(Avg_scenarios,scenarios,n_clusters,n_samples,sample_length) 

    Data_FCR = [c_FCR_scen]
    Data_names_FCR = ['FCR']
    scenarios_FCR = Bootsrap('single',Data_FCR,Data_names_FCR,n_samples,blocksize,sample_length)
    kmedoids = KMedoids(n_clusters=n_clusters,metric='euclidean').fit(scenarios_FCR[1])

    FCR_red_scen = kmedoids.cluster_centers_
    Prob_FCR = np.zeros((n_clusters))
    for j in range(0,n_clusters):
                Prob_FCR[j] = np.count_nonzero(kmedoids.labels_ == j)/len(kmedoids.labels_)

    Rep_scen_combALL = np.zeros((5,n_clusters,len(Rep_scen_comb[0][0])))
    Rep_scen_combALL[0] = Rep_scen_comb[0]  ## Day ahead 
    Rep_scen_combALL[1] = FCR_red_scen  ## FCR
    Rep_scen_combALL[2] = Rep_scen_comb[1]  ## aFRR up 
    Rep_scen_combALL[3] = Rep_scen_comb[2]  ## aFRR down 
    Rep_scen_combALL[4] = Rep_scen_comb[3]  ## mFRR  


    Prob_comb_all = np.zeros((n_clusters,len(Rep_scen_combALL)))

    for i in range(0,len(Rep_scen_combALL)):
        for j in range(0,n_clusters):
            if i != 1:
                Prob_comb_all[j][i] = Prob_comb[j]
            if i == 1:
                Prob_comb_all[j][i] = Prob_FCR[j]
    
    return Rep_scen_combALL,Prob_comb_all



#def H2_bounds(sEfficiency)
#    if sEfficiency == 'pw'

#
def write_1d_to_2d(d1):
    d2 = {}
    for t in range(1,len(d1)+1):
        d2[(1,t)] = d1[t]
    return d2



#Export to Excel 
def WriteToExcel(version,df_results_values,Start_date,End_date): 
    print(df_results_values)
    DateRange = pd.date_range(start=Start_date, end=End_date, freq='H')
    # Convert date range to a list of strings
    date_strings = [d.strftime('%Y-%m-%d %H:%M:%S') for d in DateRange]
    #Creating dataframe
    columns = ['P_PEM', 'P_import', 'P_export', 'P_grid', 'z_grid', 'P_PV',
           'b_FCR', 'beta_FCR', 'r_FCR', 'c_FCR', 'b_mFRR_up', 'beta_mFRR_up',
           'r_mFRR_up', 'c_mFRR_up', 'b_aFRR_up', 'beta_aFRR_up', 'r_aFRR_up',
           'c_aFRR_up', 'b_aFRR_down', 'beta_aFRR_down', 'r_aFRR_down',
           'c_aFRRdown', 'Raw Storage', 'Pure Storage', 'CO2', 'm_Raw_In',
           'DA_clearing', 'vOPEX']
    
    Results = pd.DataFrame(columns=columns, index=date_strings)


    if version == 1: 
        #Define which variables should be printed
        Results['P_PEM'] = df_results_values['P_PEM'].tolist()
        Results['P_import'] = df_results_values['P_import'].tolist()
        Results['P_export'] = df_results_values['P_export'].tolist()
        Results['P_grid'] = df_results_values['P_grid'].tolist()
        Results['z_grid'] = df_results_values['z_grid'].tolist()
        Results['P_PV'] = df_results_values['P_PV'].tolist()
        Results['s_raw'] = df_results_values['s_raw'].tolist()
        Results['s_pu'] = df_results_values['s_pu'].tolist()
        Results['m_CO2'] = df_results_values['m_CO2'].tolist()
        Results['m_ri'] = df_results_values['m_ri'].tolist()
        Results['DA'] = df_results_values['DA'].tolist()
        Results['vOPEX'] = df_results_values['vOPEX'].tolist()
        
    if version == 2: 
        #Define which variables should be printed
        print('MONS')
        Results['P_PEM'] = df_results_values['P_PEM'].tolist()
        Results['P_import'] = df_results_values['P_import'].tolist()
        Results['P_export'] = df_results_values['P_export'].tolist()
        Results['P_grid'] = df_results_values['P_grid'].tolist()
        Results['z_grid'] = df_results_values['z_grid'].tolist()
        Results['P_PV'] = df_results_values['P_PV'].tolist()
        Results['s_raw'] = df_results_values['s_raw'].tolist()
        Results['s_pu'] = df_results_values['s_pu'].tolist()
        Results['m_CO2'] = df_results_values['m_CO2'].tolist()
        Results['m_ri'] = df_results_values['m_ri'].tolist()
        Results['DA'] = df_results_values['DA'].tolist()
        Results['vOPEX'] = df_results_values['vOPEX'].tolist()

    if version == 3: 
        #Define which variables should be printed
        Results = pd.DataFrame({#Col name : Value(list)
                                'P_PEM' : df_results_values['P_PEM'],
                                'P_import' : df_results_values['P_import'],
                                'P_export' : df_results_values['P_export'],
                                'P_grid' : df_results_values['P_grid'],
                                'z_grid' : df_results_values['z_grid'],
                                'P_PV' : df_results_values['P_PV'],
                                'b_FCR': nan_list,
                                'beta_FCR': nan_list,
                                'r_FCR' : nan_list,
                                'c_FCR' : nan_list,
                                'b_mFRR_up': nan_list,
                                'beta_mFRR_up': nan_list,
                                'r_mFRR_up': nan_list,
                                'c_mFRR_up' : nan_list,
                                'b_aFRR_up': nan_list,
                                'beta_aFRR_up': nan_list,
                                'r_aFRR_up': nan_list,
                                'c_aFRR_up' : nan_list,
                                'b_aFRR_down': nan_list,
                                'beta_aFRR_down': nan_list,
                                'r_aFRR_down': nan_list,
                                'c_aFRRdown' : nan_list,
                                'Raw Storage' : df_results_values['s_raw'],
                                'Pure Storage' : df_results_values['s_pu'],
                                'm_Raw_In' : df_results_values['m_ri'],
                                'DA_clearing' : df_results_values['DA'],
                                'vOPEX' : df_results_values['vOPEX']},
                                    index=DateRange,)
    
    Results.to_excel("Version_"+str(version)+ "_" + Start_date[:10]+"_"+End_date[:10]+ ".xlsx")
    return Results

    
