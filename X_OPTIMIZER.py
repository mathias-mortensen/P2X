#PACKAGES
import pyomo.environ as pe
import pyomo.opt as po
from pyomo.core import *
import pandas as pd 
import numpy as np
#---------------------------------------------------------------------------
from xFunctions import *  # Contains all custom function created for the model
#---------------------------------------------------------------------------
# Input Data (from master excel file)
df_run = pd.read_excel('Opt_X.xlsx','Run')
df_param = pd.read_excel('Opt_X.xlsx','parameter settings')
df_pw = pd.read_excel('Opt_X.xlsx','Efficiency breakpoints')

# Assigning model input values to variables
Start_date = assign_var(df_run,'Start_date_sim') 
End_date = assign_var(df_run,'End_date_sim')
Demand_pattern = assign_var(df_run,'Demand_pattern')
Start_date_scen = assign_var(df_run,'Start_date_scen')
End_date_scen = assign_var(df_run,'End_date_scen')
sEfficiency = assign_var(df_run,'sEfficiency')
n_samples = int(assign_var(df_run,'n_samples'))
blocksize = int(assign_var(df_run,'blocksize'))
n_clusters = int(assign_var(df_run,'n_clusters'))
PV_Cluster = assign_var(df_run,'PV_Cluster')
n_clusters_PV = int(assign_var(df_run,'n_clusters_PV'))
blocksize_PV = int(assign_var(df_run,'blocksize_PV'))
weeks = int(assign_var(df_run,'weeks'))

# constants assignment from 'Parameter settings'
hourly_demand = assign_var(df_param,'k_d')

#Importing solar data
P_PV_max = import_PV('PV_data.xlsx',Start_date, End_date, Start_date_scen, End_date_scen)
# importing day ahead prices as well as the applicaple time range detemrined by the date settings
DA,TimeRange = import_DA('Elspotprices_RAW.csv',Start_date, End_date, Start_date_scen, End_date_scen)

Demand = demand_assignment(Demand_pattern,TimeRange,hourly_demand)

# importing FCR data
file_FCR = assign_var(df_run,'FCR price file name')
price_column_FCR = assign_var(df_run,'FCR price column')
time_column_FCR = assign_var(df_run,'FCR time column')
c_FCR = import_generic(file_FCR,price_column_FCR, time_column_FCR, Start_date, End_date, Start_date_scen, End_date_scen)
# importing mFRR data
c_mFRR = import_mFRR("MfrrReservesDK1.csv", Start_date, End_date, Start_date_scen, End_date_scen)
file_mFRR = assign_var(df_run,'mFRR price file name')
price_column_mFRR = assign_var(df_run,'mFRR price column')
time_column_mFRR = assign_var(df_run,'mFRR time column')
c_mFRR_2 = import_generic(file_mFRR,price_column_mFRR, time_column_mFRR, Start_date, End_date, Start_date_scen, End_date_scen)
# importing aFRR data using the generic function
file_aFRR_up = assign_var(df_run,'aFRR_up price file name')
price_column_aFRR_up = assign_var(df_run,'aFRR_up price column')
time_column_aFRR_up = assign_var(df_run,'aFRR_up time column')
c_aFRR_upX = import_generic(file_aFRR_up,price_column_aFRR_up, time_column_aFRR_up, Start_date, End_date, Start_date_scen, End_date_scen)
file_aFRR_down = assign_var(df_run,'aFRR_down price file name')
price_column_aFRR_down = assign_var(df_run,'aFRR_down price column')
time_column_aFRR_down = assign_var(df_run,'aFRR_down time column')
c_aFRR_down = import_generic(file_aFRR_down,price_column_aFRR_down, time_column_aFRR_down, Start_date, End_date, Start_date_scen, End_date_scen)

#PV data import
file_PV = assign_var(df_run,'Solar irradiance')
power_column_PV = assign_var(df_run,'PV power column')
time_column_PV = assign_var(df_run,'PV time column')
P_PV_max = import_generic(file_PV,power_column_PV, time_column_PV, Start_date, End_date, Start_date_scen, End_date_scen)


# Converting the efficiency breakpoints to respective lists for setpoints and resulting hydrogen mass flow
if sEfficiency == 'pw':
    pem_setpoint = df_pw['p_pem'].tolist() 
    hydrogen_mass_flow = df_pw['m'].tolist()
