#PACKAGES
import pyomo.environ as pe
import pyomo.opt as po
from pyomo.core import *
import pandas as pd
import numpy as np
#---------------------------------------------------------------------------
from xFunctions import *  # Contains all custom functions created for the model
#---------------------------------------------------------------------------
# Input Data (from master excel file)
df_run = pd.read_excel('Opt_X.xlsx','Run')
df_param = pd.read_excel('Opt_X.xlsx','parameter settings')
df_pw = pd.read_excel('Opt_X.xlsx','Efficiency breakpoints')

# Assigning model input values to variables
version = assign_var(df_run,'model_version')
Start_date = assign_var(df_run,'Start_date_sim') 
End_date = assign_var(df_run,'End_date_sim')
Demand_pattern = assign_var(df_run,'Demand_pattern')
Start_date_scen = assign_var(df_run,'Start_date_scen')
End_date_scen = assign_var(df_run,'End_date_scen')
sEfficiency = assign_var(df_run,'sEfficiency')
n_samples = int(assign_var(df_run,'n_samples'))
blocksize = int(assign_var(df_run,'block size'))
PV_Cluster = assign_var(df_run,'PV_Cluster')
n_clusters_PV = int(assign_var(df_run,'n_clusters_PV'))
blocksize_PV = int(assign_var(df_run,'blocksize_PV'))
weeks = int(assign_var(df_run,'weeks'))

# constants assignment from 'Parameter settings'
hourly_demand = assign_var(df_param,'k_d')

#PV data import
file_PV = assign_var(df_run,'Solar irradiance')
power_column_PV = assign_var(df_run,'PV power column')
time_column_PV = assign_var(df_run,'PV time column')
P_PV_max = import_generic(file_PV,power_column_PV, time_column_PV, Start_date, End_date,'dict')

# importing day ahead prices as well as the applicaple time range detemrined by the date settings
# TIMERANGE MAY NOT BE NEEDED, 'DA' is obsolete, use c_DA
DA,TimeRange = import_DA('Elspotprices_RAW.csv',Start_date, End_date, Start_date_scen, End_date_scen)
file_DA = assign_var(df_run, 'Day-ahead file name')
price_column_DA = assign_var(df_run,'DA price column')
time_column_DA = assign_var(df_run,'DA time column')
c_DA = import_generic(file_DA,price_column_DA, time_column_DA, Start_date, End_date,'dict')

#Generating demand profile
Demand = demand_assignment(Demand_pattern,TimeRange,hourly_demand)

# importing FCR data
file_FCR = assign_var(df_run,'FCR price file name')
price_column_FCR = assign_var(df_run,'FCR price column')
time_column_FCR = assign_var(df_run,'FCR time column')
c_FCR = import_generic(file_FCR,price_column_FCR, time_column_FCR, Start_date, End_date,'dict')
# importing mFRR data
c_mFRR = import_mFRR("MfrrReservesDK1.csv", Start_date, End_date, Start_date_scen, End_date_scen)
file_mFRR = assign_var(df_run,'mFRR price file name')
price_column_mFRR = assign_var(df_run,'mFRR price column')
time_column_mFRR = assign_var(df_run,'mFRR time column')
c_mFRR = import_generic(file_mFRR,price_column_mFRR, time_column_mFRR, Start_date, End_date,'dict')
# importing aFRR data using the generic function
file_aFRR_up = assign_var(df_run,'aFRR_up price file name')
price_column_aFRR_up = assign_var(df_run,'aFRR_up price column')
time_column_aFRR_up = assign_var(df_run,'aFRR_up time column')
c_aFRR_up = import_generic(file_aFRR_up,price_column_aFRR_up, time_column_aFRR_up, Start_date, End_date,'dict')
file_aFRR_down = assign_var(df_run,'aFRR_down price file name')
price_column_aFRR_down = assign_var(df_run,'aFRR_down price column')
time_column_aFRR_down = assign_var(df_run,'aFRR_down time column')
c_aFRR_down = import_generic(file_aFRR_down,price_column_aFRR_down, time_column_aFRR_down, Start_date, End_date,'dict')




# Converting the efficiency breakpoints to respective lists for setpoints and resulting hydrogen mass flow
if sEfficiency == 'pw':
    pem_setpoint = df_pw['p_pem'].tolist() 
    hydrogen_mass_flow = df_pw['m'].tolist()


if version == 3:
    c_DA_scen = import_generic(file_DA,price_column_DA, time_column_DA, Start_date_scen, End_date_scen,'list')
    c_FCR_scen = import_generic(file_FCR,price_column_FCR, time_column_FCR, Start_date_scen, End_date_scen,'list')
    c_aFRR_up_scen = import_generic(file_aFRR_up,price_column_aFRR_up, time_column_aFRR_up, Start_date_scen, End_date_scen,'list')
    c_aFRR_down_scen = import_generic(file_aFRR_down,price_column_aFRR_down, time_column_aFRR_down, Start_date_scen, End_date_scen,'list')
    c_mFRR_scen = import_generic(file_mFRR,price_column_mFRR, time_column_mFRR, Start_date_scen, End_date_scen,'list')

    Data = [c_DA_scen, c_FCR_scen, c_aFRR_up_scen, c_aFRR_down_scen, c_mFRR_scen]
    Data_names = ['DA','FCR','aFRR Up','aFRR Down','mFRR']

    Data_comb = [c_DA_scen, c_aFRR_up_scen, c_aFRR_down_scen, c_mFRR_scen]
    Data_comb_names = ['DA','aFRR Up','aFRR Down','mFRR']

    sampling_method = assign_var(df_run,'scenario sampling method')
    block_size = assign_var(df_run,'block size') #number of hours for each 
    sample_length = assign_var(df_run,'sample length') #number of hours for each scenario - should depend on time period ?
    n_clusters = int(assign_var(df_run,'number of clusters'))

    if sampling_method == 'single':
        scenarios = Bootsrap(sampling_method,Data,Data_names,n_samples,blocksize,sample_length)
        Rep_scen,Prob = K_Medoids(scenarios,n_clusters) #generating representative scenarios in "Rep_scen" from 'scenarios' with probabilities in "Prob"
        Φ, Ω,c_FCRs,c_aFRR_ups,c_aFRR_downs,c_mFRR_ups,c_DAs,π_r,π_DA = SingleInputData(Rep_scen,Prob)

    if sampling_method == 'combined':
        ## Generate Average Price for all markets for each time (Only for "Combined scenario generation"!!) ##
        ## For DA, aFRR_up & down and mFRR 
        scenarios = Bootsrap('combined',Data_comb,Data_comb_names,n_samples,blocksize,sample_length)
        Rep_scen_combALL,Prob_comb_all = scenario_comb(scenarios,n_samples,sample_length,n_clusters,c_FCR_scen,blocksize)
        Φ, Ω,c_FCRs,c_aFRR_ups,c_aFRR_downs,c_mFRR_ups,c_DAs,π_r,π_DA = SingleInputData(Rep_scen_combALL,Prob_comb_all)
